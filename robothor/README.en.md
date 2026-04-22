# RoboTHOR Data Generation Pipeline (GSI-Bench-robothor-GEN)

This project keeps only the core scripts and utilities required for train/val data generation, so the common pipelines run from a clean repository layout and are easier to maintain in open source.

## Structure
- `action_utils/`: core utilities for command generation and execution (camera/object/rotate/receptacle/spatial_remove).
- `test_robothor_simple_cluster_move.py`: main entry script (scene iteration, view sampling, command generation/execution).
- `scripts/`: runnable generation scripts (train/val and per-command type).
- `data/pregenerated_views/`: pregenerated view directory (supports symlinks to existing data).
- `data/outputs/`: generated outputs (organized by train/val).

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate base views (train):
   ```bash
   bash scripts/generate_train.sh
   ```
3. Generate command data from existing views:
   ```bash
   bash scripts/generate_train_object.sh
   bash scripts/generate_train_rotate.sh
   ```

## Pregenerated Views
Some scripts require `--pregenerated-views` to point to `selected_views.json`. Please prepare your own pregenerated views or generate fresh ones by running the base view generation script (e.g., `scripts/generate_train.sh`). The resulting view directory (for example, `data/outputs/train/with_physics`) can be used directly for this parameter.

## Outputs & Repro
All results are written to `data/outputs/` by default. You can set `CUDA_VISIBLE_DEVICES` in scripts to select the GPU.

For advanced parameters (resolution, view sampling, physics on/off), see the CLI arguments and defaults in `test_robothor_simple_cluster_move.py`.
