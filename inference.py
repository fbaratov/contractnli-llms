# import ollama
from ollama import generate

# access current prompt library
from prompts import prompts

# other stuff
import logging
# logging.basicConfig(level=logging.INFO)
import re

# THIS IS JUST FOR THE TEST! DO NOT FORGET TO REMOVE!
from prompts.output_formats import BinaryNLIReversed
from pydantic import BaseModel


def fstr(template: str, **kwargs):
    """
    Evaluates given string as an f-string.  
    """
    return eval(f'f"""{template}"""', {}, kwargs)

def assemble_prompt(contract: str, hypothesis: str, prompt_template: str = "default") -> str:
    # added to avoid warnings
    contract = contract
    hypothesis = hypothesis 

    prompt = fstr(prompts[prompt_template], contract=contract, hypothesis=hypothesis)

    return prompt

def prompt_model(prompt: str, model:str ='gemma3', options:dict=None, structure:BaseModel|None=None) -> tuple[dict|str, str|None]:
    # inference step
    if structure is not None:
        structure = eval(structure)

    output = generate(model, 
                      prompt,
                      options=options,
                      format = structure.model_json_schema() if structure is not None else None)
    
    response = output.response
    # thinking = output.thinking
    
    if structure is not None:
        try: # structure successfully followed
            response = dict(structure.model_validate_json(response))
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

def get_prediction(nli_response: str) -> str:

    clean_response = remove_between(nli_response, "<think>", "</think>")

    contradiction = check_word("contradiction", clean_response)
    entailment = check_word("entailment", clean_response)

    if contradiction and entailment:
        return "InvalidAnswer"
    elif contradiction:
        return "Contradiction"
    elif entailment:
        return "Entailment"
    elif not (contradiction or entailment):
        return "InvalidAnswer"

def get_evidence(evidence_response: str) -> list[str]:
    return re.findall("^\*\s*(.*)", evidence_response, re.MULTILINE)

def extract_answer(response: dict|str) -> tuple[str, list[str]]:
    # Applies simple steps to find nli/evidence. works if instructions are followed

    if type(response) is dict:
        nli_response = response["classification"] if "classification" in response.keys() else ""
        evidence_response = response["evidence"] if "evidence" in response.keys() else ""
    else:
        nli_response, evidence_response = response, response
    
    nli = get_prediction(nli_response)
    
    evidence = get_evidence(evidence_response)

    return nli, evidence

def process_sample(example, config):
    logits = None
    model = config["model"]
    prompt_template = config["prompt"]
    options = config["options"]
    structure = config["structure"]

    contract = example.context_text
    hypothesis = example.hypothesis_text

    prompt = assemble_prompt(contract, hypothesis, prompt_template=prompt_template)
    
    if config["backend"] == "ollama":
        output = prompt_model(prompt, model=model, options=options, structure=structure)
    elif config["backend"] == "hf":
        raise NotImplementedError()
        response, logits = ...
        logits 

    answer, evidence = extract_answer(output.response)

    return answer, evidence, output