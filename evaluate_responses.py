import json
import os
from tqdm import tqdm
from evaluation import ExNLILabel, evaluate_all
import numpy as np
import click

def load_response_dict(response_dir):
    file_dict = {}
    for _, _, files in os.walk(response_dir):
        
        for file in tqdm(files, desc="Loading responses"):
            fpath = f"{response_dir}/{file}"
            with open(fpath, "r") as f:
                file_json = json.load(f)
                file_dict[fpath] = file_json

    return file_dict

def evaluate_responses(responses):
    false_e = {}
    false_c = {}
    true_e = {}
    true_c = {}
    none_prediction = {}


    for k,v in tqdm(responses.items(), desc="Evaluating NLI"):

        try:
            if v["prediction"] is None:
                none_prediction[k] = v
                continue
        except KeyError:
            print("Skipping file with no prediction key")
            continue
        
        prediction = v["prediction"].lower()
        label = v["nli_label"].lower()
        match prediction:
            case "entailment":
                if prediction == label:
                    true_e[k] = v
                else:
                    false_e[k] = v
            
            case "contradiction":
                if prediction == label:
                    true_c[k] = v
                else:
                    false_c[k] = v
            
            case _:
                raise ValueError("Not supposed to get here!")

    return true_e, true_c, false_e, false_c, none_prediction

def format_eval_dict(true_e, true_c, false_e, false_c, none_prediction):
    eval_dict = {
        "true_e": len(true_e),
        "true_c": len(true_c),
        "false_e": len(false_e),
        "false_c": len(false_c),
        "none_prediction": len(none_prediction)
    }
    total_predictions = len(true_e) + len(true_c) + len(false_e) + len(false_c)
    total = total_predictions + len(none_prediction)
    total = total if total > 0 else 1
    eval_dict["confusion"] = [
        eval_dict["true_e"] / total,
        eval_dict["true_c"] / total,
        eval_dict["false_e"] / total,
        eval_dict["false_c"] / total
    ]
    eval_dict["none_percentage"] = len(none_prediction) / total
    
    return eval_dict

def confusion_eval(main_dir, model, prompt, seed):
    raise NotImplementedError("Not yet updated to work with update response format")
    response_dir = f"{main_dir}/{model}/{prompt}/seed{seed}"

    response_dict = load_response_dict(response_dir)
    true_e, true_c, false_e, false_c, none_prediction = evaluate_responses(response_dict)
    eval_dict = format_eval_dict(true_e, true_c, false_e, false_c, none_prediction)


    with open(f"{main_dir}/{model}/{prompt}/eval/eval_seed{seed}.json", "w") as f:
        json.dump(eval_dict, f)

def reproducibility_eval(main_dir, model, prompt, seed, dataset):
    response_dir = f"{main_dir}/{model}/{prompt}/seed{seed}"

    results = []
    for _, _, files in os.walk(response_dir):
        for file in tqdm(files, desc="Loading responses"):
            with open(file, "r") as f:
                results.append(json.load(f))
    
    eval_dict = evaluate_all(
        dataset,
        results,
        ks=[1, 3, 5, 8, 10, 15, 20, 30, 40, 50],
        task="classification"
    )

    with open(f"{main_dir}/{model}/{prompt}/eval/repro_eval_seed{seed}.json", "w") as f:
        json.dump(eval_dict, f)

@click.command()
@click.option("--output_dir", type=click.Path(), default="responses")
@click.option("--prompt", type=str, default=None)
@click.option("--seed", type=int)
@click.option("--model", type=bool, default=False)
@click.option("--data", type=click.Path(exists=True), default="data/test.json")
def main(output_dir, prompt, model, seed, data):
    with open(data, "r") as f:
        dataset = json.load(f)

    os.makedirs(f"{output_dir}/{model}/{prompt}/eval", exist_ok=True)
    
    response_dir = f"{output_dir}/{model}/{prompt}/seed{seed}"
    print(f"{response_dir}")


    reproducibility_eval(output_dir, model, prompt, seed, dataset)
    confusion_eval(output_dir, model, prompt, seed)
            


if __name__=="__main__":
    main()