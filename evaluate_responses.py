import json
import os
from tqdm import tqdm
from backend.dataset.loader import NLI4WillsLoader
from backend.dataset.utils import load_dataset, organize_dataset
from backend.evaluation import ExNLILabel, evaluate_all
# import numpy as np
import click
from backend.utils import load_json, load_response_dict

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

def get_invalid_rate(results):
    
    invalid_count = 0
    total = 0

    for doc in results:
        annotation_dict = doc["annotation_sets"][0]["annotations"]
        total += len(annotation_dict.keys())
        for hypo_id in annotation_dict.keys():
            prediction = annotation_dict[hypo_id]["prediction"]
            if prediction == "InvalidAnswer":
                invalid_count += 1
    
    return invalid_count / total

def reproducibility_eval(response_dict, eval_dir, dataset, eval_label):

    results = list(dict(sorted(response_dict.items())).values())

    eval_dict = evaluate_all(
        dataset,
        results,
        ks=[1, 3, 5, 8, 10, 15, 20, 30, 40, 50],
        task="classification"
    )

    eval_dict["invalid_rate"] = get_invalid_rate(results)

    with open(f"{eval_dir}/repro_eval_{eval_label}.json", "w") as f:
        json.dump(eval_dict, f)


@click.command()
@click.option("--response_dir", type=click.Path(exists=True))
@click.option("--eval_dir", type=click.Path())
@click.option("--dataset", type=click.Path(exists=True), default="data/contract-nli/test.json")
@click.option("--eval_label")
@click.option("--dset_type", type=str, default="contract_nli")
def main(response_dir, eval_dir, dataset, eval_label, dset_type):

    os.makedirs(eval_dir, exist_ok=True)
    
    print(f"{response_dir}")
    response_dict = load_response_dict(response_dir)

    # invalid_rate = get_invalid_rate(response_dict)
    # line = (f"{round(invalid_rate, 3)} || {response_dir}")
    
    # with open("file.txt", "a", encoding="utf-8") as f:
    #     f.write(f"{line}\n")


    # line = check_alignment(response_dict)
    # with open("file_eval.txt", "a", encoding="utf-8") as f:
    #     f.write(f"{line} || {response_dir}\n")

    #!!! UNCOMMENT FOR EVALUATION
    if dset_type != "contract_nli":
        dataset = load_dataset(dset_path=dataset, dset_type=dset_type)
        
        # convert dataset to cnli format
        dataset = dataset.to_cnli()

        # convert output to cnli format (shorten certain lists/dicts)
        for k,v in response_dict.items():
            response_dict[k] = NLI4WillsLoader.output_to_cnli(v)
    else:
        dataset = load_json(dataset)




    # print("Conducting reproducibility eval")
    reproducibility_eval(response_dict, eval_dir, dataset, eval_label)


    # print("Conducting confusion eval")
    # confusion_eval(response_dict, eval_dir, dataset, eval_label)
            


if __name__=="__main__":
    main()