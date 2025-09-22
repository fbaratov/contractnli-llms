# import ollama
from ollama import generate

# access current prompt library
from prompts import prompts

# other stuff
import logging
logging.basicConfig(level=logging.INFO)
import re



def fstr(template: str):
    """
    Evaluates given string as an f-string.  
    """
    return eval(f'f"""{template}"""')

def assemble_prompt(contract, hypothesis, prompt_template="default"):
    # added to avoid warnings
    contract = contract
    hypothesis = hypothesis 

    prompt = fstr(prompts[prompt_template])

    return prompt

def prompt_model(prompt, model='gemma3'):
    # inference step
    response = generate(model, prompt)
    return response

def extract_answer(response):
    # postprocessing step
    
    output = response.response
    match = re.match(r"^Answer:\s*(.*)", output)
    if match:
        nli = match.group(1)  # Output: This is the correct response.
    else:
        nli = None
    
    evidence = re.findall("^\*\s*(.*)", output, re.MULTILINE)
    
    if nli is None:
        logging.warning("Failure to extract answer")
    if nli in ['True', 'False'] and evidence is None:
        logging.warning("Failure to extract evidence despite T/F answer")
    if nli in ['Not Mentioned'] and evidence is not None:
        logging.info(f"Evidence extracted despite Not Mentioned answer: {evidence}")

    return nli, evidence

def process_sample(example, model='gemma3', prompt_template="default"):
    contract = example.context_text
    hypothesis = example.hypothesis_text
    prompt = assemble_prompt(contract, hypothesis, prompt_template=prompt_template)
    response = prompt_model(prompt, model=model)
    extraction = extract_answer(response)
    answer, evidence = extraction
    return response.response, answer, evidence