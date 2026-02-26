from .inference import Inference
from prompts import prompts
from prompts.output_formats import *

import logging

import ollama


class OllamaInference(Inference):
    def __init__(self, config, class_names=None):
        self.config = config
        self.class_names = class_names

    def prompt_model(self, prompt: str) -> tuple[dict|str, str|None]:
        # get values from config
        model = self.config["model"]
        options = self.config["options"]
        structure = self.config["structure"]
        top_logprobs = self.config["n_logprobs"]

        # inference step
        if structure is not None:
            structure = eval(structure)

        output = ollama.generate(model, 
                        prompt,
                        options=options,
                        logprobs=(top_logprobs > 0),
                        top_logprobs=top_logprobs,
                        format = structure.model_json_schema() if structure is not None else None)
        
        response = output.response
        # thinking = output.thinking
        
        if structure is not None:
            # structure successfully followed
            try:
                output.response = dict(structure.model_validate_json(response))
            # structure not followed successfully, let it through and attempt to salvage from string
            except Exception as e:
                logging.warning(f"Structure not followed successfully! \n {e}")

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
            nli = self.get_prediction(response, self.class_names)
        
        if evidence=="":
            evidence = self.get_evidence(response)

        return nli, evidence