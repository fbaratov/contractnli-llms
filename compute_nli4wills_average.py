import json
import sys
import numpy as np
from pathlib import Path

import click


def load_json_files(file_paths):
    """Load multiple JSON files into a list of dictionaries."""
    data = []
    for path in file_paths:
        with open(path, "r") as f:
            data.append(json.load(f))
    return data


def compute_stats(metrics_set):
    """
    Compute mean and standard deviation metric-wise.
    Assumes all dictionaries contain numeric values.
    """
    if not metrics_set:
        raise ValueError("No data provided.")

    # # Collect values per metric
    # metrics = {}
    # for d in dicts:
    #     for key, value in d.items():
    #         metrics.setdefault(key, []).append(value)

    # # Compute mean and std
    # for key, values in metrics.items():

    results = {}
    for k in metrics_set[0].keys():
        values = [d[k] for d in metrics_set]
        avg = np.mean(values)
        std = np.std(values) if len(values) > 1 else 0.0
        results[k] = {
            "avg": avg,
            "std": std
        }

    return results


    


@click.command()
@click.option("-m", type=str, default=None, multiple=True)
@click.option("-o", type=click.Path(exists=False), default=None)
def main(m, o):
    out_file = o
    metrics_set = []

    for fp in m:
        with open(fp, "r") as f:
            e = json.load(f)
            metrics_set.append(e)

    stats = {}
    
    stats["total"] = compute_stats([m["total"] for m in metrics_set])

    stats["by_class"] = {}
    for k,_ in metrics_set[0]["by_class"].items():
        stats[k] = compute_stats([m["by_class"][k] for m in metrics_set])
    
    with open(out_file, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Saved averaged metrics to {out_file}")

if __name__ == "__main__":
    main()