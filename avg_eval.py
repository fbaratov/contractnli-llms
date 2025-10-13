import json
from typing import List
import click
import numpy as np

def aggregate_metrics(results: List[dict]) -> dict:
    ret = {}
    for key, val in results[0].items():
        if isinstance(val, dict):
            ret[key] = aggregate_metrics([result[key] for result in results])
        else:
            ret[key] = {
                'average': np.average([result[key] for result in results]),
                'std': np.std([result[key] for result in results])
            }
    return ret


def recursive_get(dic: dict, keys: List[str]):
    for key in keys:
        dic = dic[key]
    return dic


def run(metrics_set, save_fp): 
    results_agg = aggregate_metrics([m for m in metrics_set])

    fout = open(save_fp, "w")

    fout.write('%s' % json.dumps(results_agg, indent=2))


@click.command()
@click.option("--main_dir", type=click.Path(exists=True))
def main(main_dir):
    models = ["deepseek-r1_8b",  "gemma3",  "gemma3_27b",  "gpt-oss_20b",  "llama3.1_8b", "qwen3_30b"]
    prompt = "narendra_nli"
    seeds = [0,1,42]
    
    for model in models:
        metrics_set = []

        for seed in seeds:
            response_dir = f"responses/temp0/{model}/{prompt}/seed{seed}"
            print(f"{response_dir}")
            
            with open(f"responses/temp0/{model}/{prompt}/eval/format_eval_seed{seed}.json", "r") as f:
                e = json.load(f)
                metrics_set.append(e)

        aggregate_file = f"responses/temp0/{model}/{prompt}/eval/aggregated_metrics.json"
        run(metrics_set, aggregate_file)
        
        

if __name__=="__main__":
    main()