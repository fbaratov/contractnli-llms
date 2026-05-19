from backend.inference import Inference, OllamaInference
import click
import ollama
import logging

from backend.format_json import save_response
from backend.utils import load_config, output_exists, get_dirs

import time

"""
Plan
Gets responses from response dict, saves to "mirror" folder
Goal: Make labelling deterministic

Plan: Prefill, formatted & no-think. Exclude thinking no-structure models (e.g. the bigger gemmas -- do based on configs)

Definite: Set all models to NoThink, pass a structure which just asks it to assign a label and do nothing else

Script input: response dir, output dir, and singular config.
Shell script role: Mimics tempsweep, but instead determines script inputs and 


1. Load responses
2. Do prefill/new prompt
3. Save this to mirror dict to preserve original responses
"""


def classify_from_prefill(inference: Inference) -> str:
    prediction = ...
    return prediction

def run_relabel(config, response_dir, output_dir, dataset, server=None, skip_complete=True):

    if server is None:
        logging.info("API connecting to server at default port!")
    else:    
        logging.info(f"API connecting to server running on address {server}")

    client=ollama.Client(host=server)
    inference = OllamaInference(config, dataset.class_names, client=client)

    try:
        ollama.pull(model=config["model"])
    except ollama.ResponseError:
        logging.error("Model could not be pulled")
        
    if skip_complete:
        logging.info("Skipping complete inferences...")

    # for each sample ==>
    for ex in dataset:

        start_time = time.time()
        answer, evidence, output = inference.process_sample(ex.__dict__, think=config["think"])
        end_time = time.time()

        logging.info(f"D: {ex.document_id} H: {ex.hypothesis_id} || Inference time: {round(end_time-start_time, 3)} seconds")
        
        # save answer to output/sample_idx
        save_response(config, ex, answer, evidence, output, output_dir, inf_time=(end_time-start_time))

@click.command()
@click.option("--model_config", type=click.Path(exists=True))
@click.option("--response_dir", type=click.Path(exists=True))
@click.option("--output_dir", type=click.Path())
@click.option("--server", type=str, default=None)
def main(model_config, response_dir, output_dir):
    # set up config 
    config = load_config(model_config)


    # refuse if structure is empty and think is False

    if config["structure"] is None and not config["think"]:
        raise ValueError("Structure is empty and Think is False: No need to prefill!")

if __name__ == "__main__":
    main()