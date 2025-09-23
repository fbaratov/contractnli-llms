import click
import logging
logging.basicConfig(level=logging.INFO)
import os
from tqdm import tqdm # type: ignore
import yaml
logging.info("Base packages loaded.")
import json
# stuff for prompting
from prompts import prompts
from dataset_utils import get_evidence, load_dataset
from inference import process_sample


logging.info("Prompting utils loaded")

# global stuff to define
test_dataset = "data/test.json"

def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_response(config, ex, response, answer, evidence, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output = {
        "document_id": ex.document_id,
        "hypothesis_id": ex.hypothesis_id,
        "response": response,
        "answer": answer,
        "nli_label": ex.label.to_anno_name(),
        "evidence": evidence,
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
    logging.info("Dataset loaded.")

    # for each sample ==>
    for i, ex in tqdm(enumerate(examples)):
        # extract evidence
        # annotated_spans = get_evidence(ex)
        # run inference on it
        response, answer, evidence = process_sample(ex, config)
        # save answer to output/sample_idx
        save_response(config, ex, response, answer, evidence, output_dir)
        break
    # <==

    logging.info("Inference complete.")

@click.command()
# @click.argument("model", type=str, default="gemma3")
# @click.argument("prompt", type=str, default="default")
@click.option("--config_path", type=click.Path(exists=False))
@click.option("--output_dir", type=click.Path(), default="responses")
@click.option("--run_label", type=click.Path(exists=False), default="normal_run")
def main(config_path, output_dir, run_label):
    config = load_config(config_path)
    model = config["model"]
    run_output_dir = f"{output_dir}/{model}/{run_label}"
    zero_shot(config, run_output_dir)

if __name__=="__main__":
    main()