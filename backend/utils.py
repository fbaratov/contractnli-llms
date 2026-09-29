import yaml
import json
from tqdm import tqdm
import logging
import os

def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_json(json_path):
    with open(json_path, "r") as f:
        file_json = json.load(f)
    return file_json

def load_response_dict(response_dir):
    # loads files as a dict with keys being document id's
    file_dict = {}
    for _, _, files in os.walk(response_dir):
        
        for file in tqdm(files, desc="Loading responses"):
            if "json" not in file:
                continue
            fpath = f"{response_dir}/{file}"
            try:
                file_json = load_json(fpath)            
            except json.JSONDecodeError:
                print(f"Error decoding file at {fpath}!")
            file_dict[file_json["id"]] = file_json

    return file_dict

def output_exists(ex, output_dir):
    out_path = f"{output_dir}/{ex.document_id}.json"
    hypothesis_id = ex.hypothesis_id
    if os.path.exists(out_path):
        out_path = load_json(out_path)


        try:
            if hypothesis_id in out_path["annotation_sets"][0]["annotations"].keys():
                return True
        except KeyError as e:
            logging.info(f"KeyError for {out_path} when trying to access out_path[\"annotation_sets\"][0][\"annotations\"]")

    return False

def get_dirs(config, output_dir, run_label, seed):
    model = config["model"]
    prompt = config["prompt"]

    run_output_dir = f"{output_dir}/{model.replace(':', '_')}/{prompt}/{run_label}/seed{seed}"
    log_dir = f"{output_dir}/{model.replace(':', '_')}/{prompt}/{run_label}/logs/"


    status_complete_path = f"{log_dir}/.seed{seed}_complete"
    run_complete = os.path.exists(status_complete_path)

    return run_output_dir, log_dir, run_complete