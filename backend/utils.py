import yaml
import json
from tqdm import tqdm
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

