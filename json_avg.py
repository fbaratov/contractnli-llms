"""
average_json.py

Takes several JSON files that all share the same structure and produces
a merged JSON where every numeric (int/float) leaf value is replaced by:

    {"avg": <mean across files>, "std": <standard deviation across files>}

Non-numeric leaves (str, bool, null) are left as-is, assuming they are
identical across files (the value from the first file is kept).

Structure rules:
- dict  -> recurse key by key
- list  -> recurse element by element (all files must have same-length lists
           at that position)
- number -> replaced with {"avg": ..., "std": ...}
- other -> taken from the first file unchanged
"""

import json
import statistics
from pathlib import Path
from typing import Any, List, Union


def _merge(values: List[Any]) -> Any:
    """Merge a list of same-position values (one per input file)."""
    first = values[0]

    # Numeric leaf (bool is a subclass of int, so exclude it explicitly)
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        nums = [float(v) for v in values]
        avg = statistics.mean(nums)
        std = statistics.stdev(nums) if len(nums) > 1 else 0.0
        return {"avg": avg, "std": std}

    # Dict: recurse per key, including keys that may be missing in some files.
    if isinstance(first, dict):
        keys = sorted({k for v in values if isinstance(v, dict) for k in v.keys()})
        merged = {}
        for k in keys:
            present = [v[k] for v in values if isinstance(v, dict) and k in v]
            if not present:
                continue
            merged[k] = _merge(present)
        return merged

    # List: recurse per index, tolerating small length differences.
    if isinstance(first, list):
        length = max(len(v) for v in values if isinstance(v, list))
        return [
            _merge([v[i] for v in values if isinstance(v, list) and i < len(v)])
            if any(isinstance(v, list) and i < len(v) for v in values)
            else None
            for i in range(length)
        ]

    # Anything else (str, bool, None): assume identical across files
    return first


def average_json_files(paths: List[Union[str, Path]]) -> Any:
    """
    Load JSON files from `paths`, all sharing the same structure, and
    return a new structure where every numeric leaf is replaced by
    {"avg": ..., "std": ...}.
    """
    if not paths:
        raise ValueError("Need at least one JSON file")

    data = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data.append(json.load(f))

    return _merge(data)


def average_json_files_to_file(paths: List[Union[str, Path]], out_path: Union[str, Path]) -> None:
    """Average a list of JSON files and write the merged result to disk."""
    result = average_json_files(paths)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python json_avg.py output.json input1.json input2.json ...")
        sys.exit(1)

    out_path = sys.argv[1]
    input_paths = sys.argv[2:]
    average_json_files_to_file(input_paths, out_path)
    print(f"Wrote averaged result to {out_path}")