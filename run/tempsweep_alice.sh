#!/bin/bash

# activate conda env before running script
# conda deactivate
# conda activate ollama

# python main.py --model config #1 --output_dir OPTIONAL --run_label #3 --prompt #4 --seed #5 --binary #6

PYTHON_CMD="python main.py ${@:2}"
echo $PYTHON_CMD

SEEDS=(0 1 42)
TEMPERATURES=(0.1) # try these three as the 1's are already done
CONFIG_DIR=$1

# Loop through the list of seeds
for MODEL in "$CONFIG_DIR"/*; do
    for SEED in "${SEEDS[@]}"; do
        for TEMP in "${TEMPERATURES[@]}"; do
                if [ -f "$MODEL" ]; then
                    echo "Queuing $MODEL with seed $SEED and temperature $TEMP ..."
                    model_label=$(basename "${MODEL%.*}")
                    run_label="${model_label}_temp${TEMP}"
                    sbatch run/main_with_server.sh $PYTHON_CMD --seed $SEED --model_config $MODEL --temperature $TEMP --run_label $run_label
                fi
        done
    done
done
