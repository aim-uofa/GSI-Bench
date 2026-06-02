
# Generative-Spatial-Intelligence

Language: English | 中文见 [`README_zh.md`](README_zh.md)

This project provides a pipeline for generative spatial intelligence, focusing on 3D object detection, scene editing, and evaluation in indoor environments. It supports moving, rotating, and removing objects, and automatically generates editing tasks and evaluation results.

## Project Structure

- `dataloader.py`: Data loading and preprocessing
- `gen_modify.py`: Generation of editing tasks and modification instructions
- `get_img.py`: Image extraction and copying
- `model.py`: 3D detection and model wrapper
- `edit_visualize.py`: Visualization of editing results
- `eval.py`: Evaluation of editing effects
- `run.sh`: One-click script for running the pipeline

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- numpy
- Pillow
- shapely
- box
- yaml
- See each script for additional dependencies

## Quick Start

1. Install dependencies:
2. Run the main pipeline:
```
bash run.sh <source_dir> <dest_dir> <output_dir> <cuda_device>
```
>
Arguments:

- <source_dir>: Directory of raw data
- <dest_dir>: Directory for editing tasks and intermediate results
- <output_dir>: Directory for final outputs
- <cuda_device>: CUDA device index


## Main Features
- Automatic generation of object editing tasks (move, rotate, remove)
- 3D object detection and spatial reasoning
- Visualization of editing results (ground truth)
- Automatic evaluation of editing effects

## Environment Setup (Recommended)

This project requires a working CUDA GPU environment. The inference code uses `cuda:0` by default.

### CUDA 11.7 (conda, recommended)

```bash
# 0) Remove old env
conda deactivate
conda env remove -n gsi -y

# 1) Create a clean environment
conda create -n gsi python=3.10 pip -y
conda activate gsi

# 2) Install PyTorch + CUDA 11.7
conda install -c pytorch -c nvidia pytorch torchvision pytorch-cuda=11.7 -y

# 3) Install CUDA toolchain (headers + nvcc)
conda install -c nvidia cuda-toolkit=11.7 cuda-nvcc=11.7 -y

# 4) Set CUDA paths (recommend adding to activate.d)
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CPATH="$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH"

# 5) Verify (must pass)
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
nvcc --version
find "$CONDA_PREFIX" -name cuda_runtime_api.h | head
find "$CONDA_PREFIX" -path "*/thrust/complex.h" | head

# 6) Install other deps
pip install -U pip ninja
pip install -r requirements-cu117.txt

# 7) Build GroundingDINO (optional; required for text prompts)
pip install -e ./src/groundingdino --no-build-isolation
```

### PYTHONPATH setup (required)

The code uses `from utils.train_utils import ...` and `from utils.wrap_model import ...`,
so the evaluation root must be in `PYTHONPATH`. Run from the `evaluation/` directory:

```bash
cd evaluation
export PYTHONPATH=$PWD:$PYTHONPATH
```

> **Note:** Do NOT set `PYTHONPATH=$PWD/utils` — this shadows the `utils` namespace package and causes import errors.

### Optional: GroundingDINO (text prompt detection)

GroundingDINO is required only if you want text-prompt detection.

```bash
pip install -e git+https://github.com/IDEA-Research/GroundingDINO.git@856dde20aee659246248e20734ef9ba5214f5e44#egg=groundingdino
```

Make sure the config and weight paths match the code defaults:

```
<repo_root>/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py
<repo_root>/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth
```

## Required Model Weights (default config)

The default config is `utils/detect_anything/configs/demo.yaml` and requires:

```
./checkpoints/detany3d/detany3d_ckpts/other_exp_ckpt.pth
./checkpoints/detany3d/sam_ckpts/sam_vit_h_4b8939.pth
./checkpoints/detany3d/dino_ckpts/dinov2_vitl14_pretrain.pth
```

Download these weights and place them in the specified paths.

## Evaluation

This section outlines how to evaluate edited images across four metrics:

- IC (Instruction Compliance)
- SA (Spatial Accuracy)
- EL (Edit Locality)
- AC (MLLM-based Appearance Consistency)

Follow the steps below to structure outputs, run metric scripts, and aggregate results.

1) Prepare directory layout

Place generated images under `eval/<model_name>/` with one subfolder per dataset:

```
eval/
  <model_name>/
    generated_images_fine/
    generated_images_mesatask/
    generated_images_bathroom/
    generated_images_robothor/
```

Datasets should be available at the repository root (names expected by `eval.sh`):

```
fine_dataset/   mesatask_dataset/   bathroom_dataset/   robothor_dataset/
```

2) Generate edited images

Use your own model, or adapt `examples/mydataset.py`. `examples/inference.py` is a BAGEL **skeleton** only — full inference needs the [BAGEL](https://github.com/ByteDance-Seed/BAGEL) project (see [`REPRODUCE_BAGEL_RESULTS.en.md`](REPRODUCE_BAGEL_RESULTS.en.md)).

To reproduce published BAGEL × fine scores without regenerating images, download [`bagel_example/`](bagel_example/README.md) (~265MB, not in Git).

Save all generated images as `<img_id>_edit_<query_id>.(png|jpg)` so the evaluator can locate paired JSON and originals (see `eval/README.md`).

3) Run IC/SA/EL evaluation

Use the provided batch script (iterates all models under `eval/` and all datasets):

```
bash eval.sh
```

Outputs will be written to `eval/<model_name>/*_eval/` with per-metric JSON files, e.g. `instruction-compliance_eval_stats.json`.

4) Run MLLM-based AC scoring (optional but recommended)

```bash
pip install -r requirements-mllm.txt   # plus vllm for your CUDA build
cd mllm_eval
bash eval_infer.sh <qwen3_vl_model_path> default <port>
```

Outputs land in `mllm_eval/infer_results/` (e.g. `predictions_infer_2000_<model>_<dataset>_<model>.json`). Pass that directory to `--mllm-eval-dir`.

5) Aggregate all metrics (IC/SA/EL/AC)

Run the aggregator to collect per-model, per-dataset means and a simple average score:

```
python -m eval.aggregate \
  --root-dir ./eval \
  --output-dir <output_dir> \
  --mllm-eval-dir <dir_with_mllm_ac_jsons>
```

The final results are JSON files in the specific output directory.
