#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PREGENERATED_TRAIN_DEFAULT="$ROOT/data/outputs/train/with_physics"
if [[ ! -d "$PREGENERATED_TRAIN_DEFAULT" ]]; then
  PREGENERATED_TRAIN_DEFAULT="$ROOT/data/pregenerated_views/train"
fi
PREGENERATED_TRAIN="${PREGENERATED_TRAIN:-$PREGENERATED_TRAIN_DEFAULT}"

python test_robothor_simple_cluster_move.py \
  --scenes "train:all" \
  --output-dir "$ROOT/data/outputs/train/object" \
  --command-types object \
  --pregenerated-views "$PREGENERATED_TRAIN"
