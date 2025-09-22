# import ollama
from ollama import generate

# other stuff
import logging
logging.basicConfig(level=logging.INFO)
import re

def assemble_prompt(premise, hypothesis):
    # preprocessing step
    prompt = f"""
You are an attorney that is checking to see if a given contract satisfies certain conditions. For the given contract and hypothesis, complete the following two tasks:
Task 1. Based on the given contract and hypothesis, state whether this the hypothesis is 'True', 'False', or 'Not Mentioned'.
Task 2. If the hypothesis is 'True' or 'False', repeat verbatim evidence sentences from the contract that back up this conclusion.

Contract: {premise}

Hypothesis: {hypothesis}

Answer conditions: State only what is requested in the format. Exclude quotes around answer or around evidence sentences. If no evidence present, give no evidence.

Answer format:
Answer: 'True', 'False', or 'Not Mentioned'
Evidence:
* Evidence 1
.......
"""
    
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
        logging.warning(f"Evidence extracted despite Not Mentioned answer: {evidence}")

    return nli, evidence
   
def main():
    # run single sample through model, return model output and sample-specific metrics 
    contract = "Chocolate milk is illegal in the state of Nevada. We are allowed to sell your information to Meta. Chocolate milk is however legal in the Las Vegas Special Autonomous Zone. We are not allowed to share your information with Google."
    label1, hypothesis1 = "True", "The contract allows selling information to Meta."
    label2, hypothesis2 = "False", "The contract allows sharing information with Google."
    label3, hypothesis3 = "Not Mentioned", "The employer recognizes John Helldiver is the greatest human to have ever lived."
    
    with open("output.txt", "w") as f:
        for hypo in [hypothesis1, hypothesis2, hypothesis3]:
            prompt = assemble_prompt(contract, hypo)
            response = prompt_model(prompt)
            extraction = extract_answer(response)
            answer, evidence = extraction
            logging.info(f"""
{hypo}

ANSWER: {answer}
EVIDENCE: {evidence}
""")
            output = f"""
=======================>
=======================>
=================> INPUT
  Contract: {contract} ---->
Hypothesis: {hypo}
<=======================

<================ OUTPUT
{extraction}
=======================>
=======================>
=======================>


        """
            f.write(output)

    
if __name__=="__main__":
    main()