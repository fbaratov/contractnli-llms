#!/bin/bash
# dataset_path, out_file, response_dir, structure



SEEDS=(0 1 42)
MODELS=(qwen3_30b gemma3_27b gpt-oss_20b llama3.1_8b)

OUT_DIR=$1

# Loop through the list of seeds
for MODEL in "${MODELS[@]}"; do
    args=""
    for SEED in "${SEEDS[@]}"; do
            RESPONSE_DIR="out/responses/config_1/$MODEL/narendra_nli_nonbinary/seed$SEED"
            args+=" --response_dir $RESPONSE_DIR"
    done

    if [[ "gpt-oss_20b" == $MODEL ]]; then
        args+="";
    else
        args+=" --structure classification";
    fi
    

    args+=" --out_dir $OUT_DIR"
    args+=" --label $MODEL"

    # run
    PYTHON_CMD="python uncertainty.py ${args}"
    echo $PYTHON_CMD
    $PYTHON_CMD

done
