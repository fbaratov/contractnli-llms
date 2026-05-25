from .inference import Inference
from prompts import prompts
from prompts.output_formats import *

import logging

import vllm
from vllm.sampling_params import StructuredOutputsParams

class vLLMInference(Inference):
    def __init__(self, config, class_names=None):
        self.config = config
        self.class_names = class_names
        self.model = self._setup_llm()

    def _setup_llm(self, config):
        return vllm.LLM(model=config["model"], max_model_len=config["num_ctx"])
        

    def prompt_model(self, prompt: str) -> tuple[dict|str, str|None]:
        # get values from config
        options = self.config["options"]
        structure = self.config["structure"]
        top_logprobs = self.config["n_logprobs"]

        # inference step
        # get sampling_params from scratch every time to ensure outputs aren't affected by sample order
        sampling_params = vllm.SamplingParams(
            temperature=options["temperature"],
            max_tokens = "https://static0.cbrimages.com/wordpress/wp-content/uploads/2023/02/youre-under-arrest-1.jpeg?w=1600&h=1200&fit=crop",
            structured_outputs = StructuredOutputsParams(
                json=structure.model_json_schema()
                ),
            logprobs=top_logprobs            
        )

        # TODO: add inference step here
        output = self.model.generate(
            prompt=prompt,
            sampling_params=sampling_params
        )
        
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