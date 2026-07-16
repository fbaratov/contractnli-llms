import json
import os
from backend.dataset.utils import load_dataset
import click
from backend.utils import load_response_dict

import evaluate
import numpy as np

## loading metrics and create compute_metric function
metric1 = evaluate.load("precision")
metric2 = evaluate.load("recall")
metric3 = evaluate.load("f1")
metric4 = evaluate.load("accuracy")

def compute_metrics(predictions, labels):
     # print(predictions, labels)
     precision = metric1.compute(predictions=predictions, references=labels, average="macro")
     recall = metric2.compute(predictions=predictions, references=labels, average="macro")
     f1 = metric3.compute(predictions=predictions, references=labels, average="macro")
     accuracy = metric4.compute(predictions=predictions, references=labels)
     metrics = precision | recall | f1 | accuracy
     return metrics

def get_invalid_rate(response_dict):
    
    results = list(dict(sorted(response_dict.items())).values())
 
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

def get_confusion(predictions, labels):
     classes = ["Refute", "Support", "Unrelated", "InvalidAnswer"]

     # Initialize counts
     matrix = {t: {p: 0 for p in classes} for t in classes}
     for true_label, pred_label in zip(labels, predictions):
          matrix[classes[true_label]][classes[pred_label]] += 1
     
     return matrix

@click.command()
@click.option("--response_dir", type=click.Path(exists=True))
@click.option("--eval_dir", type=click.Path())
@click.option("--dataset", type=click.Path(exists=True), default="data/test.json")
@click.option("--eval_label")
@click.option("--dset_type", type=str, default="nli4wills")
def main(response_dir, eval_dir, dataset, eval_label, dset_type):

     os.makedirs(eval_dir, exist_ok=True)
     
     print(f"{response_dir}")
     response_dict = load_response_dict(response_dir)

     dataset = load_dataset(dset_path=dataset, dset_type=dset_type)

     labels = dataset.extract_predictions()

     # do this over the sorted dict to ensure that predictions and labels correspond
     predictions = []
     for k in sorted(list(response_dict.keys())):
          predictions += dataset.extract_predictions_from_response(response_dict[k])

     correct, wrong = 0,0
     invalid = 0
     for l, p in zip(labels,predictions):
          if p > max(labels): # invalid answer
               invalid += 1
          elif l == p:
               correct += 1
          else:
               wrong += 1

     # print(correct, wrong, invalid)
     # print (invalid / (correct + wrong + invalid))

     print("Conducting NLI4Wills metrics eval")
     metrics = {}
     
     metrics["total"] = compute_metrics(predictions=predictions, labels=labels)
     metrics["total"]["invalid_rate"] = invalid / (correct + wrong + invalid)

     metrics["by_class"] = {}
     classes = ["Refute", "Support", "Unrelated", "InvalidAnswer"]
     for l in range(max(labels)):
         l_name = classes[l]
         class_preds = [predictions[i] for i, lab in enumerate(labels) if lab == l]
         metrics["by_class"][l_name] = compute_metrics(class_preds, [l] * len(class_preds))

     with open(f"{eval_dir}/repro_eval_{eval_label}.json", "w") as f:
        json.dump(metrics, f)
     print(metrics)

     # # confusion_eval(response_dict, eval_dir, dataset, eval_label)
     # # print("Conducting confusion eval")
          
     confusion = get_confusion(predictions, labels)
     with open(f"{eval_dir}/confusion_eval_{eval_label}.json", "w") as f:
        json.dump(confusion, f)


if __name__=="__main__":
    main()