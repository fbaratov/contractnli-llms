#!/bin/bash


# python main.py --model config #1 --output_dir OPTIONAL --run_label #3 --prompt #4 --seed #5 --binary #6
PYTHON_CMD="python main.py --prompt narendra_nli --binary True"

SEEDS=(0 1 42)
RUN_LABEL="seed${SEED}"
MODELS=("configs/gemma3_default.yml" "configs/gpt-oss20b.yml" "configs/llama8b.yml" "configs/gemma3_27b.yml" "configs/deepseek-r1_8b.yml" "configs/qwen3_30b.yml")
# Loop through the list of seeds
for MODEL in "${MODELS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do
            echo "Running with seed $SEED..."
            RUN_LABEL="seed${SEED}"
            $PYTHON_CMD --seed $SEED --run_label $RUN_LABEL --model_config $MODEL
        done
    done
