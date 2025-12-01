import json
import os
from tqdm import tqdm
from .nli_labels import ExNLILabel
import numpy as np
import click
import logging


def encode_onehot_vector(prediction):
    encoding = np.zeros(len(ExNLILabel))
    encoding[ExNLILabel.from_str(prediction).value] = 1.
    return list(encoding)

def encode_dict(choice):
    # takes a choice vector and represents it as a dictionary i guess.
    pred_dict = {
        ExNLILabel(i).to_anno_name(): float(p)
        for i, p in enumerate(choice)
    }
    return pred_dict

def encode_spans(evidence):
    #! not yet implemented! TODO: implement :)
    return evidence

def format_annotation(response, prediction: str|dict[str, str], thinking, evidence = None, class_probs: str|None = None, choice: str|None = None, logprobs = None):
    choice = encode_onehot_vector(prediction if choice is None else choice)
    spans = encode_spans(evidence)
    annotation = {
        "response": response,
        "thinking": thinking,
        "logprobs": logprobs,
        "prediction": prediction,
        "choice": choice,
        "class_probs": encode_dict(choice) if class_probs is None else class_probs,
        "spans": None
    }
    return annotation

def reformat_response(response):
    reformatted_response = {
        "model": response["model"],
        "options": response["options"],
        "id": response["document_id"],
        "prompt": response["prompt"],
        "binary": response["binary"],
        "annotation_sets": [
            {
                "annotations": {
                    response["hypothesis_id"]: format_annotation(response)
                }
            }
        ]
    }
    return reformatted_response

def save_response(config, ex, answer, evidence, output, output_dir):
    out_path = f"{output_dir}/{ex.document_id}.json"
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            response_dict = json.load(f)
    else:
        response_dict = {
            "id": ex.document_id,
            "model": config["model"],
            "options": config["options"],
            "prompt": config["prompt"],
            "binary": config["binary"],
            "annotation_sets": [{
                    "annotations": {}
                    }
            ]
        }

    response_dict["annotation_sets"][0]["annotations"][ex.hypothesis_id] = format_annotation(
        prediction=answer,
        evidence=evidence,
        response=output.answer,
        thinking=output.thinking,
        logprobs=output.logprobs,
        
        )

    with open(out_path, "w") as f:
        json.dump(response_dict, f)
        

@click.command()
@click.option("--main_dir", type=click.Path(exists=True))
def main(main_dir):
    models = ["qwen3_30b"]
    prompt = "narendra_nli"
    seeds = [0, 1, 42]

    for model in models:
        for seed in seeds:
            response_dir = f"{main_dir}/{model}/{prompt}/seed{seed}"
            print(f"{response_dir}")
            
if __name__=="__main__":
    main()