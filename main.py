import json
import pickle
import click
import logging

from backend.contract_nli_bert.contract_nli.dataset.loader import NLILabel
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
import os
from tqdm import tqdm # type: ignore
# stuff for prompting
import ollama
from backend.dataset.utils import load_dataset
from backend.format_json import save_response
from backend.utils import load_config, load_json
from backend.inference import OllamaInference
#print("Prompting utils loaded.")

def verify_config(config):
    required_config_keys = ["model", "prompt", "binary", "structure", "options", "backend"]
    for key in required_config_keys:
        if key not in config.keys():
            raise KeyError(f"Key '{key}' must be in the config!")
    
    # verify options is correct
    options = config["options"]
    required_options_keys = ["temperature", "seed"]
    for key in required_options_keys:
        if key not in options.keys():
            raise KeyError(f"Key '{key}' must be provided in options!")

def output_exists(ex, output_dir):
    out_path = f"{output_dir}/{ex.document_id}.json"
    hypothesis_id = ex.hypothesis_id
    if os.path.exists(out_path):
        out_path = load_json(out_path)


        try:
            if hypothesis_id in out_path["annotation_sets"][0]["annotations"].keys():
                return True
        except KeyError as e:
            logging.info(f"KeyError for {out_path} when trying to access out_path[\"annotation_sets\"][0][\"annotations\"]")

    return False

def zero_shot(config, output_dir, dataset, server=None, skip_complete=True):

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
        
    if config["binary"]:
        logging.warning("Skipping NotMentioned labels. Ignore this warning if this is what is meant to happen.")

    # for each sample ==>
    for ex in tqdm(dataset):
        if config["binary"] and ex.label == NLILabel.NOT_MENTIONED:
            continue

        if skip_complete and output_exists(ex, output_dir):
            logging.info(f"Answer for document {ex.document_id} hypothesis {ex.hypothesis_id} exists!")
            continue

        
        answer, evidence, output = inference.process_sample(ex.__dict__)

        # save answer to output/sample_idx
        save_response(config, ex, answer, evidence, output, output_dir)
        

@click.command()
@click.option("--seed", type=int)
@click.option("--model_config", type=click.Path(exists=False))
@click.option("--output_dir", type=click.Path())
@click.option("--run_label", type=str, default="")
@click.option("--n_logprobs", type=int, default=0)
@click.option("--temperature", type=float, default=None)
@click.option("--server", type=str, default=None)
@click.option("--skip_complete", type=bool, default=True)
def main(model_config, output_dir, run_label, seed, n_logprobs, temperature, server, skip_complete):
    """
    Runs inference based on model config and saves it to specified location. Dataset/model params determined in config, arguments determine run-specific stuff such as seed.

    :param model_config: Description
    :param output_dir: Description
    :param run_label: Description
    :param seed: Description
    :param n_logprobs: Description
    """


    # set up config 
    config = load_config(model_config)

    if "options" in config.keys():
        config["options"]["seed"] = seed
    else:
        config["options"] = {"seed": seed}

    config["n_logprobs"] = n_logprobs

    
    # verify config has all necessary fields
    verify_config(config)

    # setup output directory
    model = config["model"]
    prompt = config["prompt"]
    seed = config["options"]["seed"]
    run_output_dir = f"{output_dir}/{model.replace(':', '_')}/{prompt}/{run_label}/seed{seed}"
    os.makedirs(run_output_dir, exist_ok=True)
    
    # setup log file
    log_dir = f"{output_dir}/{model.replace(':', '_')}/{prompt}/{run_label}/logs/"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/seed{seed}.log"
    
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(filename=log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    


    # overwrite temperature
    if temperature is not None:
        logger.info(f"Setting temperature to {temperature} based on argument")
        config["options"]["temperature"] = temperature

    logger.info(str(config))
    
    #setup statuses (easier to see if everything works that way)
    status_running = f"{log_dir}/.seed{seed}_running"
    status_complete = f"{log_dir}/.seed{seed}_complete"
    status_failed = f"{log_dir}/.seed{seed}_failed"
    
    # run inference
    try:
        # remove all status files first
        for status in (status_running, status_complete, status_failed):
            if os.path.exists(status):
                os.remove(status)

        loaded_dataset = load_dataset(dset_path=config["dset_path"], dset_type=config["dset_type"])
        open(status_running, "w")
        zero_shot(config, run_output_dir, loaded_dataset, server=server, skip_complete=skip_complete)
        os.remove(status_running)
        open(status_complete, "w")
        fh.close()
    except:
        os.remove(status_running)
        open(status_failed, "w")

        logging.exception("Got exception when running inference")
        fh.close()
        raise

if __name__=="__main__":
    main()
