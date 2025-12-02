#!/bin/bash

# activate conda env before running script
# conda deactivate
# conda activate ollama

# python main.py --model config #1 --output_dir OPTIONAL --run_label #3 --prompt #4 --seed #5 --binary #6
PYTHON_CMD="python main.py ${@:2}"
echo $PYTHON_CMD

SEEDS=(0 1 42)
CONFIG_DIR=$1

# Loop through the list of seeds
for SEED in "${SEEDS[@]}"
    do
        for MODEL in "$CONFIG_DIR"/*; do
            if [ -f "$MODEL" ]; then
                echo "Running $MODEL with seed $SEED..."
                $PYTHON_CMD --seed $SEED --model_config $MODEL
            fi
        done
    done

