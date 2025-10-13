# import ollama
from ollama import generate

# access current prompt library
from prompts import prompts

# other stuff
import logging
logging.basicConfig(level=logging.INFO)
import re



def fstr(template: str, **kwargs):
    """
    Evaluates given string as an f-string.  
    """
    return eval(f'f"""{template}"""', {}, kwargs)

def assemble_prompt(contract, hypothesis, prompt_template="default"):
    # added to avoid warnings
    contract = contract
    hypothesis = hypothesis 

    prompt = fstr(prompts[prompt_template], contract=contract, hypothesis=hypothesis)

    return prompt

def prompt_model(prompt, model='gemma3', options=None):
    # inference step
    response = generate(model, prompt, options=options)
    return response


def remove_between(text: str, start_phrase: str, end_phrase: str) -> str:
    """
    Removes everything between (and including) start_phrase and end_phrase
    from the given text. If the phrases are not found, returns the text unchanged.
    """
    pattern = re.escape(start_phrase) + ".*?" + re.escape(end_phrase)
    return re.sub(pattern, "", text, flags=re.DOTALL)


def check_word(word: str, s: str):
    w_lower = word.lower()
    s_lower = s.lower()
    return w_lower in s_lower

def get_prediction(response):
    # step necessary for qwen, doesn't affect the rest
    clean_response = remove_between(response, "<think>", "</think>")

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

def extract_answer(response):
    # Applies simple steps to find nli/evidence. works if instructions are followed

    output = response.response
    
    nli = get_prediction(response)
    
    evidence = re.findall("^\*\s*(.*)", output, re.MULTILINE)

    return nli, evidence

def process_sample(example, config):
    model = config["model"]
    prompt_template = config["prompt"]
    options = config["options"]

    contract = example.context_text
    hypothesis = example.hypothesis_text

    prompt = assemble_prompt(contract, hypothesis, prompt_template=prompt_template)
    
    response = prompt_model(prompt, model=model, options=options)
    
    answer, evidence = extract_answer(response)
    thinking = response.thinking if hasattr(response, "thinking") else None
    return response.response, thinking, answer, evidence