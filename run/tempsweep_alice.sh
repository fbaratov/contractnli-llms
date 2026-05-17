#!/bin/bash

# activate conda env before running script
# conda deactivate
# conda activate ollama

# python main.py --model config #1 --output_dir OPTIONAL --run_label #3 --prompt #4 --seed #5 --binary #6

PYTHON_CMD="python main.py ${@:2}"
echo $PYTHON_CMD

CONFIG_DIR=$1
tempsweep_params="$1/tempsweep.txt"

if [[ ! -f "$tempsweep_params" ]]; then
    echo "Error: tempsweep.txt not found in '$1'" >&2
    exit 1
fi

source "$tempsweep_params"


# Loop through the list of seeds
for MODEL in "$CONFIG_DIR"/*.yml; do
    for TEMP in "${TEMPERATURES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
                if [ -f "$MODEL" ]; then
                    
                    model_label=$(basename "${MODEL%.*}")
                    run_label="${model_label}_temp${TEMP}"
                    
                    output_dir=$(echo "$@" | grep -oP '(?<=--output_dir )\S+')
                    # check if this run is already complete. if so, skip it.

                    run_complete=$(python -c "from backend.utils import get_dirs, load_config ; config=load_config('${MODEL}'); run_complete = get_dirs(config, '${output_dir}', '${run_label}', ${SEED})[2]; print(run_complete)")

                    RUN_CMD="${PYTHON_CMD} --seed ${SEED} --model_config ${MODEL} --temperature ${TEMP} --run_label ${run_label}"

                    if [ "$run_complete" == "True" ]; then
                        echo "SKIPPING: $MODEL with seed $SEED and temperature $TEMP"
                    elif [ "$run_complete" == "False" ]; then
                        echo "RUNNING: $MODEL with seed $SEED and temperature $TEMP"
                        sbatch run/main_with_server.sh $PYTHON_CMD --seed $SEED --model_config $MODEL --temperature $TEMP --run_label $run_label
                    else
                        echo "INVALID OUT: $RUN_CMD"
                    fi

                    
                fi
        done
    done
done
