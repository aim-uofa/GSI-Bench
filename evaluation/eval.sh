#!/bin/bash
declare -A ORIGINAL_PATHS=(
    ["fine_dataset"]="GSI-Real"
    ["mesatask_dataset"]="GSI-Syn"
    ["bathroom_dataset"]="GSI-Syn"
    ["robothor_dataset"]="GSI-Syn"
)

declare -A GENERATED_PATHS=(
    ["fine_dataset"]="generated_images_fine"
    ["mesatask_dataset"]="generated_images_mesatask"
    ["bathroom_dataset"]="generated_images_bathroom"
    ["robothor_dataset"]="generated_images_robothor"
)

declare -A OUTPUT_PATHS=(
    ["fine_dataset"]="fine_eval"
    ["mesatask_dataset"]="mesatask_eval"
    ["bathroom_dataset"]="bathroom_eval"
    ["robothor_dataset"]="robothor_eval"
)

DATASETS=("fine_dataset" "mesatask_dataset" "bathroom_dataset" "robothor_dataset")
MODES=("instruction-compliance" "spatial-accuracy" "edit-locality")

EVAL_ROOT="./eval"

for MODEL_DIR in $EVAL_ROOT/*; do
    if [ -d "$MODEL_DIR" ]; then
        MODEL=$(basename "$MODEL_DIR")
        echo "=============================="
        echo " Processing Model: $MODEL"
        echo "=============================="

        for DATASET in "${DATASETS[@]}"; do
            ORIGINAL="./${DATASET}"
            EDIT="./${DATASET}"

            EDITED="$MODEL_DIR/${GENERATED_PATHS[$DATASET]}"
            OUTPUT="$MODEL_DIR/${OUTPUT_PATHS[$DATASET]}"
            DATASET_TYPE=${ORIGINAL_PATHS[$DATASET]}

            for MODE in "${MODES[@]}"; do
                CMD="python -m eval.eval\
                    --original \"$ORIGINAL\" \
                    --edited \"$EDITED\" \
                    --edit \"$EDIT\" \
                    --output \"$OUTPUT\" \
                    --mode \"$MODE\" \
                    --dataset \"$DATASET_TYPE\""

                echo "Executing command:"
                echo "$CMD"
                eval $CMD
            done
        done

        echo ">>> Finished Model: $MODEL"
    fi
done
