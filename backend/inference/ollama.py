import json

from backend.nli_labels import ExNLILabel

from .inference import Inference
from prompts import prompts
from prompts.output_formats import *

import logging

import ollama


class OllamaInference(Inference):
    def __init__(self, config, class_names=None, client=None):
        self.client = ollama if client is None else client
        self.config = config
        self.class_names = class_names

    def prompt_model(self, prompt: str , think: bool=False) -> tuple[dict|str, str|None]:
        # get values from config
        model = self.config["model"]
        options = self.config["options"]
        structure = self.config["structure"]
        top_logprobs = self.config["n_logprobs"]

        # inference step
        if structure is not None:
            structure = eval(structure)
        

        output = self.client.generate(model, 
                        prompt,
                        options=options,
                        logprobs=(top_logprobs > 0),
                        top_logprobs=top_logprobs,
                        format = structure.model_json_schema() if structure is not None else None,
                        think=think)
        response = output.response
        thinking = output.thinking
        
        # failsafe added for reasoning models (specifically qwen): if response is empty but thinking is not, use the thinking as the response
        if len(response.strip()) == 0 and len(thinking) > 0:
            response = thinking

        if structure is not None:
            # structure successfully followed
            try:
                output.response = dict(structure.model_validate_json(response))
            except Exception as e:
                logging.info(f"Conversion with dict() failed, attempting json.loads()")
                try:
                    # attempt to get a json directly from the response
                    output.response = json.loads(response)
                # structure not followed successfully, let it through and attempt to salvage from string
                except Exception as e:
                    logging.warning(f"Conversion to JSON failed - Structure not followed successfully! \n {e}")

        return output

    def get_prediction(self, nli_response: str) -> str:
        
        # relevant for some thinking models
        clean_response = self._remove_between(nli_response, "<think>", "</think>")

        class_mentioned = None
        for cn in self.class_names:
            cm = self._check_word(cn, clean_response)
            
            # first class mention
            if cm and class_mentioned is None:
                class_mentioned = cn
            # multiple classes mentioned (invalid)
            elif cm and class_mentioned is not None:
                print("Multiple classes mentioned")
                return "InvalidAnswer"
        
        # no class found (invalid)
        if class_mentioned is None:
            return "InvalidAnswer"
        # exactly one class mentioned (valid)
        else:
            return class_mentioned

    def get_evidence(self, evidence_response: str) -> list[str]:
        """
        Unimplemented, included for future relevance
        """
        return None
        return refindall("^\*\s*(.*)", evidence_response, re.MULTILINE)

    def postprocess(self, response: dict|str) -> tuple[str, list[str]]:
        # Applies simple steps to find nli/evidence. works if instructions are followed

        if type(response) is dict:
            nli = response["classification"] if "classification" in response.keys() else ""
            evidence = response["evidence"] if "evidence" in response.keys() else ""
        else:
            nli, evidence = "",""

        if nli=="":
            nli = self.get_prediction(response)
        
        if evidence=="":
            evidence = self.get_evidence(response)

        return nli, evidence
    
    def _create_prefill(self, response):
        """
        Assumes the "explanation" "classification" structure. Won't work otherwise.
        """

        explanation = response["explanation"]

        prefill = f'{{"explanation": "{explanation}", "classification": "'        

        return prefill

    def classify_from_explanation(self, sample, original_response) -> str:

        prompt_template = self.config["prompt"]
        user_prompt = self.assemble_prompt(sample, prompt_template=prompt_template)

        model = self.config["model"]

        structure = self.config["structure"]
        if structure is not None:
            structure = eval(structure)

        options = self.config["options"].copy()
        options["temperature"] = 0.00001 # keep it low to ensure deterministic output
        options["num_predict"] = 100 # very generous

        # inference step
        
        # prefill = self._create_prefill(original_response)
        messages=[
                {
                    "role": "user",
                    "content": user_prompt 
                },
                {
                    "role": "assistant",
                    "content": original_response["explanation"] # prefill starts here
                },
                {
                    "role": "assistant",
                    "format": NLIClassification.model_json_schema()
                }
            ]

        completion_response = ollama.chat(
            model=model,
            messages=messages,
            # format=structure.model_json_schema(),
            think=False,
            options=options
        )
        
        prediction = completion_response.message.content
        # response = {
        #     "explanation": original_response["explanation"],
        #     "classification": prediction
        # }

        # clean up (necessary for qwen at least)
        prediction = self._remove_between(prediction, "<think>", "</think>")
        prediction = prediction.strip()

        # verify that prediction fits one of the labels
        try:
            ExNLILabel.from_str(prediction).value
            return prediction, None
        
        except ValueError:
            # contradiction = ExNLILabel.CONTRADICTION.to_anno_name().lower() in prediction.lower()
            # entailment = ExNLILabel.ENTAILMENT.to_anno_name().lower() in prediction.lower()
            # not_mentioned = ExNLILabel.NOT_MENTIONED.to_anno_name().lower() in prediction.lower()
            
            # if sum(contradiction, entailment, not_mentioned) == 1:
            #     return 

            logging.info("Sample produced Invalid Label!")

            return ExNLILabel.INVALID_ANSWER.to_anno_name(), prediction

    