import os

from backend.dataset.utils import load_dataset
from backend.inference import Inference, OllamaInference
import click
import ollama
import logging

from backend.nli_labels import ExNLILabel
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from backend.format_json import save_response
from backend.utils import load_config, load_response_dict, output_exists, get_dirs

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

class OutputWrapper:
    def __init__(self, response, thinking, logprobs=None):
        self.response = response
        self.thinking = thinking
        self.logprobs = logprobs

    def __init__(self, response_dict):
        self.response = response_dict["response"]
        self.thinking = response_dict["thinking"]
        self.logprobs = response_dict["logprobs"]


def run_relabel(config, response_dict, output_dir, dataset, server=None, skip_complete=True):

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
        if skip_complete and output_exists(ex, output_dir):
            logging.debug(f"Answer for document {ex.document_id} hypothesis {ex.hypothesis_id} exists!")
            continue

        annotation = response_dict[ex.document_id]["annotation_sets"][0]["annotations"][ex.hypothesis_id]

        if annotation["prediction"] == ExNLILabel.INVALID_ANSWER.to_anno_name():
            logging.debug("Skipping InvalidAnswer")
            continue

        start_time = time.time()
        prediction, full_response = inference.classify_from_explanation(sample=ex.__dict__, original_response=annotation["response"])
        end_time = time.time()
        

        logging.info(f"D: {ex.document_id} H: {ex.hypothesis_id} || Inference time: {round(end_time-start_time, 3)} seconds")

        # save answer, keep the original response there but overwrite the prediction
        if full_response is None:
            save_response(config, ex, prediction, None, OutputWrapper(annotation), output_dir, inf_time=(end_time-start_time))
        else:
            save_response(config, ex, prediction, None, OutputWrapper(annotation), output_dir, inf_time=(end_time-start_time), prediction_response=full_response)
        
        # print(prediction)
        # break

@click.command()
@click.option("--model_config", type=click.Path(exists=True))
@click.option("--response_dir", type=click.Path(exists=True))
@click.option("--output_dir", type=click.Path())
@click.option("--server", type=str, default=None)
def main(model_config, response_dir, output_dir, server):
    # set up config 
    config = load_config(model_config)

    # load everything right up here
    loaded_dataset = load_dataset(dset_path=config["dset_path"], dset_type=config["dset_type"])
    response_dict = load_response_dict(response_dir)

    # refuse if structure is empty and think is False

    if config["structure"] is None and not config["think"]:
        raise ValueError("Structure is empty and Think is False: No need to prefill!")


    # determine directories
    parent_dir, run_dir = os.path.split(output_dir)
    print(parent_dir, run_dir)
    log_dir = f"{parent_dir}/log/"
    log_file = f"{log_dir}/{run_dir}.log"

    # setup log file
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(filename=log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    #setup statuses (easier to see if everything works that way)
    status_running = f"{log_dir}/running"
    status_complete = f"{log_dir}/complete"
    status_failed = f"{log_dir}/failed"
        
    
    # run inference
    try:
        # remove all status files first
        for status in (status_running, status_complete, status_failed):
            if os.path.exists(status):
                os.remove(status)
    
        open(status_running, "w")
        run_relabel(
            config=config,
            response_dict=response_dict,
            output_dir=output_dir,
            dataset=loaded_dataset,
            server=server,
            skip_complete=True)
        os.remove(status_running)
        open(status_complete, "w")
        fh.close()
    except:
        os.remove(status_running)
        open(status_failed, "w")

        logging.exception("Got exception when running inference")
        fh.close()
        raise

if __name__ == "__main__":
    main()