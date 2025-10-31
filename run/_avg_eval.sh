#!/bin/bash

MAIN_DIR=$1

BASE_CMD="python evaluate_responses.py"

# Ensure there are at least two arguments
if [ $# -lt 2 ]; then
  echo "Usage: $0 arg1 arg2 [arg3 ...]"
  echo "Requires at least two arguments."
  exit 1
fi

# Remove the last argument from the list
args=("${@:1:$#-1}")          # all arguments except the last

# Join all except last with '-m'
ASSEMBLED_STRING=$(printf " -m %s" "${args[@]}")
eval OUT_FP="\${$#}"

# Make sure output path does not exist
if [ -e "$OUT_FP" ]; then
    echo "❌ Path exists: $OUT_FP. Pick a new filepath or delete the previous file. "
    exit 1
fi

# Create dirpath for the output file
dirpath=$(dirname "$OUT_FP")
if [ ! -d "$dirpath" ]; then
    echo "Creating directory: $dirpath"
    mkdir -p "$dirpath"
else
    echo "Directory already exists: $dirpath"
fi


# run python command
PYTHON_CMD="$BASE_CMD $ASSEMBLED_STRING -o $OUT_FP"
echo $PYTHON_CMD
# $PYTHON_CMD
