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
            try:
                ### this one is to prevent None breaking things (why is it even there?)
                for result in results:
                    if result[key] is None:
                        result[key] = np.nan
                ###
                ret[key] = {
                    'average': np.average([result[key] for result in results]),
                    'std': np.std([result[key] for result in results])
                }
            except TypeError as e:
                print(e)
                print([result[key] for result in results])
                print(results)
                exit()
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
@click.option("-m", type=list, default=None, multiple=True)
@click.option("-o", type=click.Path(exists=False), default=None)
def main(m, o):
    out_file = o
    metrics_set = []

    for fp in m:
        with open(fp, "r") as f:
            e = json.load(f)
            metrics_set.append(e)

            run(metrics_set, out_file)
        
        

if __name__=="__main__":
    main()