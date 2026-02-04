#!/bin/bash

MAIN_DIR=$1

BASE_CMD="python evaluate_responses.py ${@:2}"



find "$MAIN_DIR" -type d | while read -r dir; do
    # Check if the directory contains any subdirectories
    if ! find "$dir" -mindepth 1 -type d | grep -q .; then
        if [[ "$dir" == *"eval"* ]]; then
            echo "Skipping eval directory: $dir"
            continue  # skip this iteration
        elif [[ "$dir" == *"logs"* ]]; then
            echo "Skipping log directory: $dir"
            continue # skip this iteration
        fi
        
        RESPONSE_DIR=$dir
        echo "Response directory: $RESPONSE_DIR"
        EVAL_DIR="${dir%/*}/eval"
        mkdir $EVAL_DIR
        EVAL_LABEL=$(basename "$dir")
        echo "Label: $EVAL_LABEL"

        PYTHON_CMD="$BASE_CMD --response_dir $RESPONSE_DIR --eval_dir $EVAL_DIR --eval_label $EVAL_LABEL"
        echo $PYTHON_CMD
        $PYTHON_CMD
        
        echo "Evaluation saved to: $EVAL_DIR"
        # Do something with $dir here
    fi
done


