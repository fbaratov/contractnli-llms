import json
import os
from tqdm import tqdm
from backend.evaluation import ExNLILabel, evaluate_all
import numpy as np
import click
from backend.utils import load_response_dict

def evaluate_responses(hypo_dict, dataset):
    per_doc = {}

    organized_dset = {}
    for d in dataset["documents"]:
        organized_dset[d["id"]] = d


    for id, doc_dict in tqdm(hypo_dict.items(), desc="Evaluating NLI"):
        
        true_e = 0
        true_c = 0
        false_e = 0
        false_c = 0
        none_prediction = 0

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
            match prediction:
                case "entailment":
                    if prediction == label:
                        true_e += 1
                    else:
                        false_e += 1
                
                case "contradiction":
                    if prediction == label:
                        true_c += 1
                    else:
                        false_c += 1
                
                case _:
                    none_prediction += 1


        per_doc[id] = {
            "true_e": true_e,
            "false_e": false_e,
            "false_c": false_c,
            "true_c": true_c,
            "none_prediction": none_prediction,
            "true_prediction": true_e + true_c,
            "false_prediction": false_c + false_e
        }

    return per_doc

def format_eval_dict(per_doc):
    sum_dict = {k: 0 for k in list(per_doc.values())[0]}
    for _, v in per_doc.items():
        for k, count in v.items():
            sum_dict[k] += count
    
    
    return sum_dict

def confusion_eval(response_dir, eval_dir, dataset, eval_label):

    response_dict = load_response_dict(response_dir)
    per_doc = evaluate_responses(response_dict, dataset)
    sum_dict = format_eval_dict(per_doc)

    per_doc["sum"] = sum_dict

    with open(f"{eval_dir}/confusion_eval_{eval_label}.json", "w") as f:
        json.dump(per_doc, f)

def reproducibility_eval(response_dir, eval_dir, dataset, eval_label):
  
    # fps = []
    # for filedir, _, files in (os.walk(response_dir)):
    #     if files is None:
    #         continue
    #     for file in files:
    #         fp = f"{filedir}/{file}"
    #         fps.append(fp)

    # results = []
    # for fp in tqdm(fps, desc="Loading_responses"):
    #     with open(fp, "r") as f:
    #         results.append(json.load(f))

    results = list(load_response_dict(response_dir).values())

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

    print("Conducting reproducibility eval")
    reproducibility_eval(response_dir, eval_dir, dataset, eval_label)
    print("Conducting confusion eval")
    confusion_eval(response_dir, eval_dir, dataset, eval_label)
            


if __name__=="__main__":
    main()