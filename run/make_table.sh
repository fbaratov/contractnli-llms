#!/bin/bash


set -e

BASE_CMD="python to_table.py"

ROOT_DIR=$1
OUT_FILE=$2
TEMPLATE=$3

# Ensure the input is a directory
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Error: '$ROOT_DIR' is not a directory."
  exit 1
fi

# set relevant file based on template
if [ "$TEMPLATE" = "confusion" ]; then
    KEYPHRASE="aggregated_confusion.json"
elif [ "$TEMPLATE" = "metrics_binary" ]; then
    KEYPHRASE="aggregated_metrics.json"
elif [ "$TEMPLATE" = "metrics" ]; then
    KEYPHRASE="aggregated_metrics.json"
fi


# Find all directories named "eval"

mapfile -t json_files < <(find $ROOT_DIR -type f -name $KEYPHRASE)
args=" --template $TEMPLATE"
for file in "${json_files[@]}"; do
    args+=" -f $file"
done
    
args+=" -o $OUT_FILE"

# Print array in a readable form
if ((${#json_files[@]} > 0)); then
    echo "Root directory: ${eval_dir#$ROOT_DIR/}"
    echo "confusion_jsons=("
    for file in "${json_files[@]}"; do
    echo "  \"$file\""
    done
    echo ")"
fi
echo "Output to : $OUT_FILE"
echo "Using template : $TEMPLATE"
echo

PYTHON_CMD="$BASE_CMD $args"
$PYTHON_CMD