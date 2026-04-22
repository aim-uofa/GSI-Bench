#!/bin/bash
set -euo pipefail

# Usage:
#   ./eval_one.sh <model_dir> <dataset_key> [mode1,mode2,...]
#
# Example:
#   ./eval_one.sh ./eval/my_model fine_dataset
#   ./eval_one.sh ./eval/my_model fine_dataset instruction-compliance,spatial-accuracy,edit-locality

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <model_dir> <dataset_key> [mode1,mode2,...]"
  echo "Dataset keys: fine_dataset | mesatask_dataset | bathroom_dataset | robothor_dataset"
  echo "Modes: instruction-compliance | spatial-accuracy | edit-locality"
  echo "Default modes: instruction-compliance,spatial-accuracy,edit-locality"
  exit 1
fi

MODEL_DIR="$1"
DATASET_KEY="$2"
MODES_CSV="${3:-instruction-compliance,spatial-accuracy,edit-locality}"

declare -A DATASET_TYPES=(
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

if [ ! -d "$MODEL_DIR" ]; then
  echo "Model dir not found: $MODEL_DIR"
  exit 1
fi

if [ -z "${DATASET_TYPES[$DATASET_KEY]+x}" ]; then
  echo "Unknown dataset key: $DATASET_KEY"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 数据集路径
ORIGINAL="$ROOT_DIR/$DATASET_KEY"
EDIT="$ROOT_DIR/$DATASET_KEY"

# 模型生成结果路径
EDITED="$MODEL_DIR/${GENERATED_PATHS[$DATASET_KEY]}"

# 评测输出路径，和你的 Python 聚合脚本保持一致：xxx_eval
OUTPUT="$MODEL_DIR/${OUTPUT_PATHS[$DATASET_KEY]}"

# 数据集类型，传给 eval.eval
DATASET_TYPE="${DATASET_TYPES[$DATASET_KEY]}"

if [ ! -d "$ORIGINAL" ]; then
  echo "Original/Edit dataset dir not found: $ORIGINAL"
  exit 1
fi

if [ ! -d "$EDITED" ]; then
  echo "Edited image dir not found: $EDITED"
  exit 1
fi

mkdir -p "$OUTPUT"

IFS=',' read -r -a MODES <<< "$MODES_CSV"

for MODE in "${MODES[@]}"; do
  case "$MODE" in
    instruction-compliance|spatial-accuracy|edit-locality)
      ;;
    *)
      echo "Unknown mode: $MODE"
      echo "Supported modes: instruction-compliance | spatial-accuracy | edit-locality"
      exit 1
      ;;
  esac

  echo "Executing mode: $MODE"
  python -m eval.eval \
    --original "$ORIGINAL" \
    --edited "$EDITED" \
    --edit "$EDIT" \
    --output "$OUTPUT" \
    --mode "$MODE" \
    --dataset "$DATASET_TYPE"
done

echo "Done. Results saved to: $OUTPUT"