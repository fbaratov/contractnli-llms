#!/usr/bin/env python3
"""
Scours a directory for `aggregated_confusion.json` files, and renders each
one as a markdown confusion matrix (rows = true label, cols = predicted
label), where each cell shows "avg +/- std".

Only files whose path contains "temp0.3" are processed.

Expected JSON shape (either form is supported per-cell):
    {
        "true_label_a": {
            "pred_label_a": [0.1, 0.2, 0.15, ...],   # raw values -> avg/std computed
            "pred_label_b": {"avg": 0.3, "std": 0.05} # already aggregated
        },
        ...
    }

Usage:
    python build_confusion_matrices.py /path/to/search/root [-o OUTPUT.md]
"""

import argparse
import json
import statistics
from pathlib import Path

TARGET_FILENAME = "aggregated_confusion.json"
REQUIRED_SUBSTRING = "temp0.3"


def format_cell(value) -> str:
    """Turn a cell's raw contents into an 'avg +/- std' string."""
    if value is None:
        return ""

    # Already-aggregated dict form: {"avg": ..., "std": ...}
    if isinstance(value, dict):
        avg = value.get("average")
        std = value.get("std", 0.0)
        if avg is None:
            return ""
        return f"{avg:.0f} ± {std:.0f}"

    # Raw list of numbers -> compute avg/std ourselves
    if isinstance(value, (list, tuple)):
        nums = [float(v) for v in value]
        if not nums:
            return ""
        avg = statistics.mean(nums)
        std = statistics.stdev(nums) if len(nums) > 1 else 0.0
        return f"{avg:.3f} ± {std:.3f}"

    # Single scalar, no spread info available
    if isinstance(value, (int, float)):
        return f"{value:.3f} ± 0.000"

    return str(value)


def collect_labels(matrix: dict) -> list:
    """Collect the union of true labels and predicted labels, preserving
    the order they first appear (true labels first, then any predicted
    labels not already seen)."""
    labels = list(matrix.keys())
    for true_label, row in matrix.items():
        if isinstance(row, dict):
            for pred_label in row.keys():
                if pred_label not in labels:
                    labels.append(pred_label)
    return labels


def matrix_to_markdown(matrix: dict, title: str) -> str:
    labels = collect_labels(matrix)

    header = "| True \\ Pred | " + " | ".join(labels) + " |"
    separator = "|---" * (len(labels) + 1) + "|"

    rows = []
    for true_label in labels:
        # if true_label == "InvalidAnswer":
        #     continue
        row_data = matrix.get(true_label, {})
        cells = []
        for pred_label in labels:
            cell_value = row_data.get(pred_label) if isinstance(row_data, dict) else None
            cells.append(format_cell(cell_value))
        rows.append(f"| {true_label} | " + " | ".join(cells) + " |")

    table = "\n".join([header, separator, *rows])
    return f"## {title}\n\n{table}\n"


def find_target_files(root: Path):
    for path in root.rglob(TARGET_FILENAME):
        if REQUIRED_SUBSTRING in str(path):
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory to search recursively")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("confusion_matrices.md"),
        help="Output markdown file (default: confusion_matrices.md)"
    )
    args = parser.parse_args()

    sections = []
    found_any = False

    for path in sorted(find_target_files(args.root)):
        found_any = True
        try:
            with open(path, "r") as f:
                matrix = json.load(f)
                del matrix["InvalidAnswer"]
                del matrix["sum"]
        except (json.JSONDecodeError, OSError) as e:
            print(f"Skipping {path}: failed to read/parse ({e})")
            continue

        section = matrix_to_markdown(matrix, title=str(path))
        sections.append(section)
        print(f"Processed: {path}")

    if not found_any:
        print(f"No '{TARGET_FILENAME}' files containing '{REQUIRED_SUBSTRING}' found under {args.root}")
        return

    output_content = "\n\n".join(sections)
    args.output.write_text(output_content)
    print(f"\nWrote {len(sections)} matrix section(s) to {args.output}")


if __name__ == "__main__":
    main()