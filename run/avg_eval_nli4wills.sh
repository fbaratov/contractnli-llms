#!/bin/bash


set -e

BASE_CMD="python compute_nli4wills_average.py"

ROOT_DIR="${1:-.}"

# Ensure the input is a directory
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Error: '$ROOT_DIR' is not a directory."
  exit 1
fi

# Find all directories named "eval"
eval_dirs=$(find "$ROOT_DIR" -type d -name "eval")

# Process each eval directory
for eval_dir in $eval_dirs; do
  echo "===================================================================================================================="
  echo "===================================================================================================================="
  # Find JSON files with "confusion" in the name (relative to ROOT_DIR)
  mapfile -t json_files < <(find "$eval_dir" -type f -name "*repro*.json" | sed "s|^$ROOT_DIR/||")

    args=""
        for file in "${json_files[@]}"; do
            args+=" -m $file"
        done
    
    OUT_FILE="$eval_dir/aggregated_metrics.json"
    args+=" -o $OUT_FILE"

    # Print array in a readable form
    if ((${#json_files[@]} > 0)); then
        echo "Eval directory: ${eval_dir#$ROOT_DIR/}"
        echo "confusion_jsons=("
        for file in "${json_files[@]}"; do
        echo "  \"$file\""
        done
        echo ")"
    fi
    echo "Output to : $OUT_FILE"
    echo

    PYTHON_CMD="$BASE_CMD $args"
    echo $PYTHON_CMD
    $PYTHON_CMD
done



for eval_dir in $eval_dirs; do
  # Find JSON files with "confusion" in the name (relative to ROOT_DIR)
  mapfile -t json_files < <(find "$eval_dir" -type f -name "*confusion_*.json" | sed "s|^$ROOT_DIR/||")

    OUT_FILE="$eval_dir/aggregated_confusion.json"
    args="$OUT_FILE"

    for file in "${json_files[@]}"; do
        args+=" $file"
    done
    

    # Print array in a readable form
    if ((${#json_files[@]} > 0)); then
        echo "Eval directory: ${eval_dir#$ROOT_DIR/}"
        echo "confusion_jsons=("
        for file in "${json_files[@]}"; do
        echo "  \"$file\""
        done
        echo ")"
    fi
    echo "Output to : $OUT_FILE"
    echo

    PYTHON_CMD="python json_avg.py $args"

    $PYTHON_CMD
done

