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

def extract_answer(response):
    # postprocessing step
    
    output = response.response
    match = re.match(r"^Answer:\s*(.*)", output)
    match = re.sub(r'[^\w\s]', '', match)
    if match:
        nli = match.group(1)  # Output: This is the correct response.
    else:
        nli = None
    
    evidence = re.findall("^\*\s*(.*)", output, re.MULTILINE)
    
    if nli.lower() not in (s.lower() for s in ['True', 'False', 'Not Mentioned']):
        logging.warning(f"Model has provided invalid answer {nli}")

    if nli is None:
        logging.warning("Failure to extract answer")
    elif nli in ['True', 'False'] and evidence is None:
        logging.warning("Failure to extract evidence despite T/F answer")
    elif nli in ['Not Mentioned'] and evidence is not None:
        logging.warning(f"Evidence extracted despite Not Mentioned answer: {evidence}")

    return nli, evidence

def process_sample(example, config):
    model = config["model"]
    prompt_template = config["prompt"]
    options = config["options"]

    contract = example.context_text
    hypothesis = example.hypothesis_text

    prompt = assemble_prompt(contract, hypothesis, prompt_template=prompt_template)
    
    response = prompt_model(prompt, model=model, options=options)
    
    extraction = extract_answer(response)
    answer, evidence = extraction
    return response.response, answer, evidence