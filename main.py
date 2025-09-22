import click
import logging
logging.basicConfig(level=logging.INFO)
import os
from tqdm import tqdm

logging.info("Base packages loaded.")

# stuff for prompting
from prompts import prompts
from dataset_utils import get_evidence, load_dataset
from inference import process_sample


logging.info("Prompting utils loaded")

# global stuff to define
test_dataset = "data/test.json"

def save_response(response, idx,  output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/response_{idx}.txt", "w") as f:
        f.write(response)

def zero_shot(model, prompt, output_dir):
    # load test dataset
    examples = load_dataset(test_dataset)
    logging.info("Dataset loaded.")

    # for each sample ==>
    for i, ex in tqdm(enumerate(examples)):
        # extract evidence
        annotated_spans = get_evidence(ex)
        # run inference on it
        response, _, _ = process_sample(ex, model=model, prompt_template=prompt)
        # save answer to output/sample_idx
        save_response(response, i, output_dir)
    # <==

    logging.info("Inference complete.")

@click.command()
@click.argument("model", type=str, default="gemma3")
@click.argument("prompt", type=str, default="default")
@click.argument("output_dir", type=click.Path(exists=False), default="output")
def main(model, prompt, output_dir):
    zero_shot(model, prompt, output_dir)

if __name__=="__main__":
    main()