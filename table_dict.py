import json
from tqdm import tqdm
import click
import os

@click.command()
@click.option("--main_dir", type=click.Path(exists=True))
def main(main_dir):
    models = ["deepseek-r1_8b",  "gemma3",  "gemma3_27b",  "gpt-oss_20b",  "llama3.1_8b", "qwen3_30b"]
    prompt = "narendra_nli"
    table = {}
    for model in tqdm(models):
        aggregate_file = f"responses/temp0/{model}/{prompt}/eval/aggregated_metrics.json"
        
        with open(aggregate_file, "r") as f:
            agg_metrics = json.load(f)
        
        relevant_metrics = agg_metrics["macro_label_micro_doc"]["class_binary"]

        table[model] = {
            "accuracy": relevant_metrics["accuracy"],
            "f1_entailment": relevant_metrics["f1_entailment"],
            "f1_contradiction": relevant_metrics["f1_contradiction"]
        }
    
    table_path = f"{main_dir}/eval"
    os.makedirs(table_path, exist_ok=True)
    
    with open(f"{table_path}/table.json", "w") as f:
        json.dump(table, f)
        

if __name__=="__main__":
    main()