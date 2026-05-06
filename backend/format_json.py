import json
import os
from .nli_labels import ExNLILabel
import numpy as np


def encode_onehot_vector(prediction):
    if type(prediction) is str:
        prediction = ExNLILabel.from_str(prediction).value
    if type(prediction) is ExNLILabel:
        prediction = prediction.value

    encoding = np.zeros(len(ExNLILabel))
    encoding[prediction] = 1.
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

def format_annotation(response, prediction: str|dict[str, str], thinking, evidence = None, class_probs: str|None = None, choice: str|None = None, logprobs = None, inf_time=None):
    choice = encode_onehot_vector(prediction if choice is None else choice)
    spans = encode_spans(evidence)
    annotation = {
        "response": response,
        "thinking": thinking,
        "inference_time": inf_time,
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

def save_response(config, ex, answer, evidence, output, output_dir, inf_time=None):
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
        response=output.response,
        thinking=output.thinking,
        logprobs=logprob_to_dict(output.logprobs),
        inf_time=inf_time
        )

    with open(out_path, "w") as f:
        json.dump(response_dict, f)




def convert_logprob(logprob):
        return {
            "token": logprob.token,
            "logprob": logprob.logprob
        }

def logprob_to_dict(logprobs):
    if logprobs is None:
        return None
    
    out_list = []
    for token in logprobs:
        out_dict = convert_logprob(token)
        out_dict["top_logprobs"] = [
            convert_logprob(logprob) for logprob in token.top_logprobs
        ]
        out_list.append(out_dict)

    return out_list