import json
import os
from tqdm import tqdm
from nli_labels import ExNLILabel
import numpy as np
import click

def encode_vector(prediction):
    encoding = np.zeros(len(ExNLILabel))
    encoding[ExNLILabel.from_str(prediction).value] = 1.
    return list(encoding)

def encode_dict(choice):
    pred_dict = {
        ExNLILabel(i).to_anno_name(): float(p)
        for i, p in enumerate(choice)
    }
    return pred_dict

def encode_spans(evidence):
    #! not yet implemented! TODO: implement :)
    return evidence

def format_annotation(response, prediction, thinking, evidence):
    choice = encode_vector(prediction)
    spans = encode_spans(evidence)
    annotation = {
        "response": response,
        "thinking": thinking,
        "prediction": prediction,
        "choice": choice,
        "class_probs": encode_dict(choice),
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

def save_response(config, ex, response, answer, thinking, evidence, output_dir):
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
        response=response,
        prediction=answer,
        thinking=thinking,
        evidence=evidence
        )

    with open(out_path, "w") as f:
        json.dump(response_dict, f)


def format_for_evaluation(response_dir):
    raise DeprecationWarning("Running this script is directly no longer necessary, functionality implemented in main.py")

    formatted_dict = {}
    for _, _, files in os.walk(response_dir):
        
        for file in tqdm(files, desc="Loading responses"):
            if file == "format.json":
                continue
            fpath = f"{response_dir}/{file}"
            with open(fpath, "r") as f:
                response = json.load(f)
                try:
                    document_id = response["document_id"]
                    if document_id not in formatted_dict.keys():
                        formatted_dict[document_id] = reformat_response(response)
                    else:
                        hypothesis_id = response["hypothesis_id"]
                        formatted_dict[document_id]["annotation_sets"][0]["annotations"][hypothesis_id] = format_annotation(response)
                except KeyError:
                    print(fpath)
                    print(response)
                    exit()
    return formatted_dict


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
            

            formatted_dict = format_for_evaluation(response_dir)

            with open(f"{response_dir}/format.json", "w") as f:
                json.dump(formatted_dict, f)

if __name__=="__main__":
    main()