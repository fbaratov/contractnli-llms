import click
import logging

from contract_nli.dataset.loader import NLILabel
logging.basicConfig(level=logging.WARNING)
import os
from tqdm import tqdm # type: ignore
import yaml
#print("Base packages loaded.\nLoading prompting utils....", end=" ")
import json
# stuff for prompting
from prompts import prompts
from dataset_utils import get_evidence, load_dataset
from inference import process_sample


#print("Prompting utils loaded.")

# global stuff to define
test_dataset = "data/test.json"

def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_response(config, ex, response, answer, thinking, evidence, output_dir):
    
    output = {
        "document_id": ex.document_id,
        "hypothesis_id": ex.hypothesis_id,
        "response": response,
        "prediction": answer,
        "thinking": thinking,
        "nli_label": ex.label.to_anno_name(),
        "prediction_evidence": evidence,
        "annotated_spans": get_evidence(ex)
    }
    output.update(config)
    
    filename = f"response_{output['document_id']}_{output['hypothesis_id']}.json"
    with open(f"{output_dir}/{filename}", "w") as f:
        json.dump(output, f)

    logging.debug("Response ")


def zero_shot(config, output_dir):
    # load test dataset
    examples = load_dataset(test_dataset)
    #print("Dataset loaded.")

    if config["binary"]:
        logging.warning("Skipping NotMentioned labels. Ignore this warning if this is what is meant to happen.")

    # for each sample ==>
    for i, ex in tqdm(enumerate(examples)):
        if config["binary"] and ex.label == NLILabel.NOT_MENTIONED:
            continue
        # extract evidence
        # annotated_spans = get_evidence(ex)
        # run inference on it
        response, thinking, answer, evidence = process_sample(ex, config)
        # save answer to output/sample_idx
        save_response(config, ex, response, answer, thinking, evidence, output_dir)
        
    # <==

@click.command()
# @click.argument("model", type=str, default="gemma3")
# @click.argument("prompt", type=str, default="default")

@click.option("--model_config", type=click.Path(exists=False))
@click.option("--output_dir", type=click.Path(), default="responses")
@click.option("--run_label", type=click.Path(exists=False), default="normal_run")
# @click.option("--model", type=str, default=None) #Removed with the logic: model setup is a little more complicated so load that from config
@click.option("--prompt", type=str, default=None)
@click.option("--seed", type=int)
@click.option("--binary", type=bool, default=False)
def main(model_config, output_dir, run_label, seed, prompt, binary):
    
    # set up config 
    config = load_config(model_config)
    model = config["model"]
    config["prompt"] = prompt
    config["binary"] = binary
    if "options" in config.keys():
        config["options"]["seed"] = seed
    else:
        config["options"] = {"seed": seed}
    
    # output directory status
    run_output_dir = f"{output_dir}/{model.replace(':', '_')}/{prompt}/{run_label}"
    os.makedirs(run_output_dir, exist_ok=True)
    
    # run inference
    zero_shot(config, run_output_dir)

if __name__=="__main__":
    main()