import os
import json
import re
import click
from tqdm import tqdm

def remove_between(text: str, start_phrase: str, end_phrase: str) -> str:
    """
    Removes everything between (and including) start_phrase and end_phrase
    from the given text. If the phrases are not found, returns the text unchanged.
    """
    pattern = re.escape(start_phrase) + ".*?" + re.escape(end_phrase)
    return re.sub(pattern, "", text, flags=re.DOTALL)



def check_word(word: str, s: str):
    w_lower = word.lower()
    s_lower = s.lower()
    return w_lower in s_lower

def get_prediction(response):
    # step necessary for qwen, doesn't affect the rest
    clean_response = remove_between(response, "<think>", "</think>")

    contradiction = check_word("contradiction", clean_response)
    entailment = check_word("entailment", clean_response)

    if contradiction and entailment:
        return None
    elif contradiction:
        return "Contradiction"
    elif entailment:
        return "Entailment"
    else:
        return None

def update_predictions(response_dir):
    for _, _, files in os.walk(response_dir):
        
        for file in tqdm(files):
            fpath = f"{response_dir}/{file}"
            try:
                with open(fpath, "r") as f:
                    file_json = json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading {file}")
                exit()
            try:

                prediction = get_prediction(file_json["response"])
                file_json["prediction"] = prediction
            

                with open(fpath, "w") as f:
                    json.dump(file_json, f)
            except KeyError:
                continue
            

@click.command()
@click.option("--main_dir", type=click.Path(exists=True))
def main(main_dir):
    models = ["deepseek-r1_8b",  "gemma3",  "gemma3_27b",  "gpt-oss_20b",  "llama3.1_8b", "qwen3_30b"]
    prompt = "narendra_nli"
    seeds = [0,1,42]

    for model in models:
        for seed in seeds:
            response_dir = f"{main_dir}/{model}/{prompt}/seed{seed}"
            print(f"{response_dir}")
            update_predictions(response_dir)


if __name__=="__main__":
    raise DeprecationWarning("The function of this script is now covered by inference.py")
    # main()