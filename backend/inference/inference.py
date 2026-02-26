# Load model directly
from prompts import prompts

import re

class Inference:
    def __init__(self, config: dict, class_names):
        # initialize model
        
        self.config = config
        self.class_names = class_names

    def _fstr(self, template: str, **kwargs):
        """
        Evaluates given string as an f-string.  
        """
        return eval(f'f"""{template}"""', {}, kwargs)
    

    def _remove_between(self, text: str, start_phrase: str, end_phrase: str) -> str:
        """
        Removes everything between (and including) start_phrase and end_phrase
        from the given text. If the phrases are not found, returns the text unchanged.
        """
        pattern = re.escape(start_phrase) + ".*?" + re.escape(end_phrase)
        return re.sub(pattern, "", text, flags=re.DOTALL)


    def _check_word(self, word: str, s: str) -> bool:
        w_lower = word.lower()
        s_lower = s.lower()
        return w_lower in s_lower
    
    def assemble_prompt(self, fields_dict: str, prompt_template: str = "default") -> str:
        prompt = self._fstr(prompts[prompt_template], **fields_dict)

        return prompt

    def prompt_model(self, prompt: str) -> tuple[dict|str, str|None]:
        output = ...

        raise NotImplementedError("Implement in subclass")

        return output
    
    def postprocess(self, response: dict|str) -> tuple[str, list[str]]:
        nli, evidence = ...

        raise NotImplementedError("Implement in subclass!")

        return nli, evidence
    
    def process_sample(self, sample: dict):
        answer, evidence, output = ...

        prompt_template = self.config["prompt"]
        prompt = self.assemble_prompt(sample, prompt_template=prompt_template)
        
        output = self.prompt_model(prompt)
        
        answer, evidence = self.postprocess(output.response, class_names=self.class_names)


        return answer, evidence, output

    def __call__(self, sample: dict):
        return self.process_sample(sample)
    
