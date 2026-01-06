import pickle
import click
import logging

from contract_nli.dataset.loader import NLILabel
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
import os
from tqdm import tqdm # type: ignore
# stuff for prompting
from backend.dataset_utils import get_evidence, load_dataset
from inference import process_sample
from backend.format_json import save_response
from backend.utils import test_dataset, load_config
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

def zero_shot(config, output_dir):
    # load test dataset
    examples = load_dataset(test_dataset)

    if config["binary"]:
        logging.warning("Skipping NotMentioned labels. Ignore this warning if this is what is meant to happen.")

    # for each sample ==>
    for ex in tqdm(examples):
        if config["binary"] and ex.label == NLILabel.NOT_MENTIONED:
            continue

        answer, evidence, output = process_sample(ex, config)

        # save answer to output/sample_idx
        save_response(config, ex, answer, evidence, output, output_dir)
        

@click.command()
@click.option("--seed", type=int)
@click.option("--model_config", type=click.Path(exists=False))
@click.option("--output_dir", type=click.Path())
@click.option("--run_label", type=str, default="")
def main(model_config, output_dir, run_label, seed):

    # set up config 
    config = load_config(model_config)

    if "options" in config.keys():
        config["options"]["seed"] = seed
    else:
        config["options"] = {"seed": seed}

    config["n_logprobs"] = 0 # hardcoded number of logprobs, would be easy to specify in configs instead. low number to make inference quicker
    
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

    # run inference
    try:
        zero_shot(config, run_output_dir)
        fh.close()
    except:
        logging.exception("Got exception when running inference")
        fh.close()
        raise

if __name__=="__main__":
    main()
