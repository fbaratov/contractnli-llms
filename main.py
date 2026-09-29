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
from backend.utils import load_config, output_exists, get_dirs
from backend.inference import OllamaInference
import time
#print("Prompting utils loaded.")


def verify_config(config):
    required_config_keys = ["model", "prompt", "binary", "structure", "options", "backend", "think"]
    for key in required_config_keys:
        if key not in config.keys():
            raise KeyError(f"Key '{key}' must be in the config!")
    
    # verify options is correct
    options = config["options"]
    required_options_keys = ["temperature", "seed"]
    for key in required_options_keys:
        if key not in options.keys():
            raise KeyError(f"Key '{key}' must be provided in options!")

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

    if skip_complete:
        logging.info("Skipping complete inferences...")

    # for each sample ==>
    for i, ex in enumerate(tqdm(dataset)):
        # if i % 10 > 0:
        #     continue

        if config["binary"] and ex.label == NLILabel.NOT_MENTIONED:
            continue


        if skip_complete and output_exists(ex, output_dir):
            logging.debug(f"Answer for document {ex.document_id} hypothesis {ex.hypothesis_id} exists!")
            continue

        start_time = time.time()
        answer, evidence, output = inference.process_sample(ex.__dict__, think=config["think"])
        end_time = time.time()

        logging.info(f"D: {ex.document_id} H: {ex.hypothesis_id} || Inference time: {round(end_time-start_time, 3)} seconds")
        # save answer to output/sample_idx
        save_response(config, ex, answer, evidence, output, output_dir, inf_time=(end_time-start_time))



@click.command()
@click.option("--seed", type=int, default=0)
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

    # this doesn't matter .ollama ignores value. planned.
    

    # set up config 
    config = load_config(model_config)

    if "options" in config.keys():
        config["options"]["seed"] = seed
    else:
        config["options"] = {"seed": seed}

    config["n_logprobs"] = n_logprobs
    # config["think"] = None

    

    # determine directories
    run_output_dir, log_dir, run_complete = get_dirs(config, output_dir, run_label, seed)

    
    # setup log file
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/seed{seed}.log"

    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(filename=log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    
    # overwrite temperature
    if temperature is not None and temperature < 0:
        logger.info(f"Negative temperature {temperature} passed, using temperature from config")
    elif temperature is not None:
        logger.info(f"Setting temperature to {temperature} based on argument")
        config["options"]["temperature"] = temperature
    if temperature is None:
        logger.info(f"Using temperature provided in config options")

    # verify config has all necessary fields
    verify_config(config)
    


    # config is complete, log it
    logger.info(str(config))
    
    #setup statuses (easier to see if everything works that way)
    status_running = f"{log_dir}/.seed{seed}_running"
    status_complete = f"{log_dir}/.seed{seed}_complete"
    status_failed = f"{log_dir}/.seed{seed}_failed"

    # cancel this experiment if the run is already marked as complete
    if run_complete:
        logging.error("Run is already complete (status file exists)! Cancelling....")
        raise FileExistsError(f"Run is already complete: {status_complete} exists.")
    
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
