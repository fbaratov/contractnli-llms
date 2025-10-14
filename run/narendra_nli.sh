#!/bin/bash

# activate conda env before running script
# conda deactivate
# conda activate ollama

# python main.py --model config #1 --output_dir OPTIONAL --run_label #3 --prompt #4 --seed #5 --binary #6
PYTHON_CMD="python main.py --prompt narendra_nli --binary True $@"
echo $PYTHON_CMD

SEEDS=(0 1 42)
MODELS=("configs/gemma3_default.yml" "configs/gpt-oss20b.yml" "configs/llama8b.yml" "configs/gemma3_27b.yml" "configs/deepseek-r1_8b.yml" "configs/qwen3_30b.yml")
# Loop through the list of seeds
for MODEL in "${MODELS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do
            echo "Running $MODEL with seed $SEED..."
            $PYTHON_CMD --seed $SEED --model_config $MODEL
        done
    done
