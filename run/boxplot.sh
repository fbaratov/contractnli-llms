#!/bin/bash
# dataset_path, out_file, response_dir, structure



SEEDS=(0 1 42)
MODELS=(qwen3_30b gemma3_27b gpt-oss_20b llama3.1_8b)

OUT_DIR_PREFIX="out/figs/config_1/first_token/"

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
    
    OUT_FILE="${OUT_DIR_PREFIX}/${MODEL}.png"
    args+=" --out_file $OUT_FILE"

    # run
    PYTHON_CMD="python uncertainty.py ${args}"
    echo $PYTHON_CMD
    $PYTHON_CMD

done
