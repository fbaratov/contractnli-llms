# import ollama
from ollama import generate

# access current prompt library
from prompts import prompts

# other stuff
import logging
# logging.basicConfig(level=logging.INFO)
import re

from prompts.output_formats import *


def fstr(template: str, **kwargs):
    """
    Evaluates given string as an f-string.  
    """
    return eval(f'f"""{template}"""', {}, kwargs)

def assemble_prompt(fields_dict: str, prompt_template: str = "default") -> str:
    prompt = fstr(prompts[prompt_template], **fields_dict)

    return prompt

def prompt_model(prompt: str, config) -> tuple[dict|str, str|None]:
    # get values from config
    model = config["model"]
    options = config["options"]
    structure = config["structure"]
    top_logprobs = config["n_logprobs"]

    # inference step
    if structure is not None:
        structure = eval(structure)

    output = generate(model, 
                      prompt,
                      options=options,
                      logprobs=(top_logprobs > 0),
                      top_logprobs=top_logprobs,
                      format = structure.model_json_schema() if structure is not None else None)
    
    response = output.response
    # thinking = output.thinking
    
    if structure is not None:
        try: # structure successfully followed
            output.response = dict(structure.model_validate_json(response))
        except Exception as e: # structure not followed successfully, let it through and attempt to salvage from string
            logging.warning(f"Structure not followed successfully! \n {e}")

    return output


def remove_between(text: str, start_phrase: str, end_phrase: str) -> str:
    """
    Removes everything between (and including) start_phrase and end_phrase
    from the given text. If the phrases are not found, returns the text unchanged.
    """
    pattern = re.escape(start_phrase) + ".*?" + re.escape(end_phrase)
    return re.sub(pattern, "", text, flags=re.DOTALL)


def check_word(word: str, s: str) -> bool:
    w_lower = word.lower()
    s_lower = s.lower()
    return w_lower in s_lower

def get_prediction(nli_response: str, class_names: list[str]) -> str:
    
    # relevant for some thinking models
    clean_response = remove_between(nli_response, "<think>", "</think>")

    class_mentioned = None
    for cn in class_names:
        cm = check_word(cn, clean_response)
        
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

def get_evidence(evidence_response: str) -> list[str]:
    return None
    return re.findall("^\*\s*(.*)", evidence_response, re.MULTILINE)

def extract_answer(response: dict|str, class_names: list[str]) -> tuple[str, list[str]]:
    # Applies simple steps to find nli/evidence. works if instructions are followed

    if type(response) is dict:
        nli = response["classification"] if "classification" in response.keys() else ""
        evidence = response["evidence"] if "evidence" in response.keys() else ""
    else:
        nli, evidence = "",""

    if nli=="":
        nli = get_prediction(response, class_names)
    
    if evidence=="":
        evidence = get_evidence(response)

    return nli, evidence

def process_sample(example: dict, config, class_names: list[str]):

    prompt_template = config["prompt"]
    prompt = assemble_prompt(example, prompt_template=prompt_template)
    
    if config["backend"] == "ollama":
        output = prompt_model(prompt, config)
    elif config["backend"] == "hf":
        raise NotImplementedError()
    
    answer, evidence = extract_answer(output.response, class_names=class_names)

    return answer, evidence, output