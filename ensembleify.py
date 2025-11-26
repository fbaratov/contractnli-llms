from collections import Counter
import json
import os
import click
from tqdm import tqdm

from backend.format_json import format_annotation
from backend.nli_labels import ExNLILabel
from backend.utils import load_json

def predictions_to_probs(predictions: dict[str, str]) -> dict[float]:
    ranked_preds = dict(Counter(predictions.values()))
    

    for k,v in ranked_preds.items():
        ranked_preds[k] = v / len(predictions)

    # add missing labels
    for label in ExNLILabel:
        label_name = label.to_anno_name()
        if label_name not in ranked_preds:
            ranked_preds[label_name] = 0.

    return ranked_preds

def aggregate_sample(samples: dict[str, dict]) -> dict:
    """
    Based on a dictionary of samples of the same document-hypothesis pair, unify into a single aggregate across models.
    """
    predictions = {} # contains model:prediction items

    for model,sample in samples.items():
        predictions[model] = sample["prediction"]

    # set choice and class_probs
    class_probs = predictions_to_probs(predictions)

    # based on probabilities, pick a choice.
    # TODO: consider the case where two have the same value (use many models to guarantee it doesn't happen?)
    choice = max(class_probs, key=class_probs.get)

    return format_annotation(
        response=None,
        prediction=predictions,
        thinking=None,
        evidence=None,
        class_probs=class_probs,
        choice=choice
        )

def aggregate_json(documents: dict[str, dict]) -> dict:
    """
    Based on a list of document dicts, aggregate the jsons by aggregating across samples. 
    """
    # get list of all hypotheses relevant to the document
    first_doc = list(documents.values())[0]
    hypotheses = list(first_doc["annotation_sets"][0]["annotations"].keys())
    
    
    annotations = {}

    for hypothesis in hypotheses:
        # get all annotations for the document-hypothesis pair
        samples = {}
        for model, d in documents.items():
            anno = d["annotation_sets"][0]["annotations"][hypothesis]
            samples[model] = anno

        final = aggregate_sample(samples)
        annotations[hypothesis] = final
    
    return annotations

def load_across_models(filepaths: list[str]) -> list[dict]:
    """
    Load document JSONs as dictionaries.
    """
    documents = []
    for fp in filepaths:
        doc = load_json(fp)
        documents.append(doc)
    
    return documents

def group_by_document(response_paths: list[str]) -> dict:
    """
    Take examine all relevant filepaths and group them by the relevant document
    """

    # get all doc jsons, assuming all models did inference on the exact same documents (if they didn't that's a problem)
    children = [entry.name for entry in os.scandir(response_paths[0]) if entry.is_file()]

    documents = {}
    for filename in tqdm(children, desc="Grouping documents...."):
        document = {}
        for rp in response_paths:
            fp = f"{rp}/{filename}"

            if not os.path.exists(fp):
                print(f"Path {fp} does not exist! What?")

            response = load_json(fp)
            document[response["model"]] = response
        documents[filename] = document

    return documents        


@click.command()
@click.option("--out_dir", type=click.Path(exists=False))
@click.option("--path_file", type=click.Path(exists=True))
def main(out_dir, path_file):
    """
    Loads JSON files from given paths and aggregates them sample-wise across multiple models, per seed.
    """
    # create out_dir
    out_dir = f"{out_dir}/ensemble"
    os.makedirs(out_dir, exist_ok=True)

    # load paths from file
    paths = load_json(path_file)["paths"]
    
    # given multiple filepaths (gpt/seed0, gemma3/seed0, etc etc), group files by document
    doc_groups = group_by_document(paths)

    # for each document group, run through the aggregator
    for filename, documents in tqdm(doc_groups.items(), desc="Processing samples..."):
        

        first_doc = list(documents.values())[0]
        annotations = aggregate_json(documents)
        
        out_dict = {
            "id": first_doc["id"],
            "binary": first_doc["binary"],
            "annotation_sets": [
                annotations
            ]
        }

        out_file = f"{out_dir}/{filename}"
        with open(out_file, "w") as f:
            json.dump(out_dict, f)
        
    
    # save aggregate under out_dir/ensemble/{doc_id}.json
    pass


if __name__=="__main__":
    main()