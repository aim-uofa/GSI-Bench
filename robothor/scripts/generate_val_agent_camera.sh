#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PREGENERATED_VAL_DEFAULT="$ROOT/data/outputs/val/with_physics"
if [[ ! -d "$PREGENERATED_VAL_DEFAULT" ]]; then
  PREGENERATED_VAL_DEFAULT="$ROOT/data/pregenerated_views/val"
fi
PREGENERATED_VAL="${PREGENERATED_VAL:-$PREGENERATED_VAL_DEFAULT}"

python test_robothor_simple_cluster_move.py \
  --scenes "val:all" \
  --output-dir "$ROOT/data/outputs/val/agent_camera" \
  --command-types agent_camera \
  --pregenerated-views "$PREGENERATED_VAL"
