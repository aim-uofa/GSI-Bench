#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python test_robothor_simple_cluster_move.py \
  --scenes "train:all" \
  --output-dir "$ROOT/data/outputs/train/with_physics"
