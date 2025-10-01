import json
import os
from tqdm import tqdm

def load_response_dict(response_dir):

    for _, _, files in os.walk(response_dir):
        file_dict = {}
        
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
        
        if v["prediction"] is None:
            none_prediction[k] = v
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

def save_eval(true_e, true_c, false_e, false_c, none_prediction, save_path):
    eval = {
        "true_e": len(true_e),
        "true_c": len(true_c),
        "false_e": len(false_e),
        "false_c": len(false_c),
        "none_prediction": len(none_prediction)
    }
    total_predictions = len(true_e) + len(true_c) + len(false_e) + len(false_c)
    total = total_predictions + len(none_prediction)
    total = total if total > 0 else 1
    eval["confusion"] = [
        eval["true_e"] / total,
        eval["true_c"] / total,
        eval["false_e"] / total,
        eval["false_c"] / total
    ]
    eval["none_percentage"] = len(none_prediction) / total

    with open(save_path, "w") as f:
        json.dump(eval, f)


def main():
    models = ["deepseek-r1_8b",  "gemma3",  "gemma3_27b",  "gpt-oss_20b",  "llama3.1_8b", "qwen3_30b"]
    prompt = "narendra_nli"
    seeds = [0,1,42]

    for model in models:
        for seed in seeds:
            response_dir = f"responses/temp0/{model}/{prompt}/seed{seed}"
            print(f"{response_dir}")
            responses = load_response_dict(response_dir)
            true_e, true_c, false_e, false_c, none_prediction = evaluate_responses(responses)
            eval_dir = f"responses/temp0/{model}/{prompt}/eval/"
            os.makedirs(eval_dir, exist_ok=True)
            save_path = f"{eval_dir}/seed{seed}_eval.json"
            save_eval(true_e, true_c, false_e, false_c, none_prediction, save_path)

if __name__=="__main__":
    main()