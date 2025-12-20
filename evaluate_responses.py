import json
import os
from tqdm import tqdm
from backend.dataset_utils import organize_dataset
from backend.evaluation import ExNLILabel, evaluate_all
import numpy as np
import click
from backend.utils import load_response_dict

def evaluate_responses(hypo_dict, dataset):
    per_doc = {}

    organized_dset = organize_dataset(dataset)


    for id, doc_dict in tqdm(hypo_dict.items(), desc="Evaluating NLI"):
        
        confusion = {
            "e_as_e": 0,
            "c_as_e": 0,
            "n_as_e": 0,
            "e_as_c": 0,
            "c_as_c": 0,
            "n_as_c": 0,
            "e_as_n": 0,
            "c_as_n": 0,
            "n_as_n": 0,
            "invalid": 0
        }
        

        hypo_dict = doc_dict["annotation_sets"][0]["annotations"]
        for hypo_id, response_dict in hypo_dict.items():
            try:
                if response_dict["prediction"] is None:
                    none_prediction += 1
                    continue
            except KeyError:
                print("Skipping file with no prediction key")
                continue
        
            prediction = response_dict["prediction"].lower()

            label = organized_dset[id]["annotation_sets"][0]["annotations"][hypo_id]["choice"].lower()
            
            label_char = label[0]
            prediction_char = prediction[0]
            
            if prediction_char == "i":
                dict_key = "invalid"
            else:
                dict_key = f"{label_char}_as_{prediction_char}"
            
            confusion[dict_key] += 1

        per_doc[id] = confusion

    return per_doc

def format_eval_dict(per_doc):
    sum_dict = {k: 0 for k in list(per_doc.values())[0]}
    for _, v in per_doc.items():
        for k, count in v.items():
            sum_dict[k] += count
    
    
    return sum_dict

def confusion_eval(response_dict, eval_dir, dataset, eval_label):
    per_doc = evaluate_responses(response_dict, dataset)
    sum_dict = format_eval_dict(per_doc)

    per_doc["sum"] = sum_dict

    with open(f"{eval_dir}/confusion_eval_{eval_label}.json", "w") as f:
        json.dump(per_doc, f)

def reproducibility_eval(response_dict, eval_dir, dataset, eval_label):

    results = list(response_dict.values())

    eval_dict = evaluate_all(
        dataset,
        results,
        ks=[1, 3, 5, 8, 10, 15, 20, 30, 40, 50],
        task="classification"
    )

    with open(f"{eval_dir}/repro_eval_{eval_label}.json", "w") as f:
        json.dump(eval_dict, f)

@click.command()
@click.option("--response_dir", type=click.Path(exists=True))
@click.option("--eval_dir", type=click.Path())
@click.option("--data", type=click.Path(exists=True), default="data/test.json")
@click.option("--eval_label")
def main(response_dir, eval_dir, data, eval_label):
    with open(data, "r") as f:
        dataset = json.load(f)

    os.makedirs(eval_dir, exist_ok=True)
    
    print(f"{response_dir}")
    response_dict = load_response_dict(response_dir)


    print("Conducting reproducibility eval")
    reproducibility_eval(response_dict, eval_dir, dataset, eval_label)
    print("Conducting confusion eval")
    confusion_eval(response_dict, eval_dir, dataset, eval_label)
            


if __name__=="__main__":
    main()