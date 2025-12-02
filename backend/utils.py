import yaml
import json
from tqdm import tqdm
import os

# global stuff to define
test_dataset = "data/test.json"

def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_json(json_path):
    with open(json_path, "r") as f:
        file_json = json.load(f)
    return file_json

def load_response_dict(response_dir):
    file_dict = {}
    for _, _, files in os.walk(response_dir):
        
        for file in tqdm(files, desc="Loading responses"):
            fpath = f"{response_dir}/{file}"
            file_json = load_json(fpath)            
            file_dict[file_json["id"]] = file_json

    return file_dict

