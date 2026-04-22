# Eval Usage Guide

Language: English | 中文见 [`README.md`](README.md)

This document covers the evaluation flow for `eval.sh` and `python -m eval.eval`, including input format, naming conventions, weights, and environment requirements.

## Entry points

- Batch evaluation (multiple models, multiple datasets): `bash eval.sh`
- One-off evaluation (custom directories): `python -m eval.eval --original <dir> --edited <dir> --output <dir> --mode <mode> --dataset <name>`

## Directory layout (`eval.sh` convention)

`eval.sh` iterates each subdirectory under `./eval/*` as a per-model output directory and evaluates four datasets:

```
<repo_root>/
  fine_dataset/
  mesatask_dataset/
  bathroom_dataset/
  robothor_dataset/
  eval/
    <model_name>/
      generated_images_fine/
      generated_images_mesatask/
      generated_images_bathroom/
      generated_images_robothor/
```

Outputs are written to:

```
eval/<model_name>/
  fine_eval/
  mesatask_eval/
  bathroom_eval/
  robothor_eval/
```

## Input and naming conventions (critical)

The evaluator derives JSON and source-image paths from the **edited image filename**, so names must match.

### 1) Edited images (`--edited` directory)

**Must contain `_edit_`**:

```
<img_id>_edit_<query_id>.jpg|png
```

The script first applies `img_file.replace("_rgb","")`, so you can have:

```
<img_id>_edit_<query_id>_rgb.jpg
```

but parsing uses the name after stripping `_rgb`.

### 2) Source images (`--original` directory)

The first file in `--original` whose **name starts with `<img_id>`** is treated as the source image:

```
<img_id>*.jpg|png
```

### 3) Edit JSON (`--original` directory)

**The JSON filename must match the edited image name** (only the extension changes to `.json`) and must live in `--original`:

```
<img_id>_edit_<query_id>.json
```

> Note: `--edit` in `eval.py` is **currently unused**; JSONs are looked up only via the above rule.

### 4) View-change optional GT image

If the prompt contains `camera` but does not contain `relative to the camera`, it is classified as a `view` operation, and the script attempts to load a GT edited image:

```
<img_id>_*gtedit_<query_id>*.*   (in --original)
```

If not found, the view evaluation falls back to failure.

## Edit JSON format (minimum fields)

Fields used by the evaluator (produced by `gen_data/generate.py`):

```json
{
  "prompt": "Move the chair 20 centimeters to the left, while keeping other objects unchanged.",
  "target": "chair",
  "original_bbox_3d": [x, y, z, w, h, l, yaw],
  "new_bbox_3d": [x, y, z, w, h, l, yaw],
  "camera_intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "rotation_matrix": [[...],[...],[...]],
  "new_rotation_matrix": [[...],[...],[...]]
}
```

Field requirements:

- **Required**: `prompt`, `target`, `original_bbox_3d`, `camera_intrinsics`, `rotation_matrix`
- **move/scale**: `new_bbox_3d`
- **rotate**: `new_rotation_matrix`
- **remove**: only uses `original_bbox_3d`
- **view**: uses the GT edited image; doesn't strictly need bbox, but `original_bbox_3d` is still used for locality

## Weights and config

Evaluation runs DetAny3D inference. Default config:

```
--config ./detect_anything/configs/demo.yaml
```

The actual config in this repo lives at:

```
utils/detect_anything/configs/demo.yaml
```

So pass it explicitly:

```
python -m eval.eval --config utils/detect_anything/configs/demo.yaml ...
```

Weight paths required by `demo.yaml` (prepare them yourself):

```
./checkpoints/detany3d/detany3d_ckpts/other_exp_ckpt.pth
./checkpoints/detany3d/sam_ckpts/sam_vit_h_4b8939.pth
./checkpoints/detany3d/dino_ckpts/dinov2_vitl14_pretrain.pth
```

Optional (for text-prompt detection):

```
GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py
GroundingDINO/weights/groundingdino_swinb_cogcoor.pth
```

## Environment (recommended)

Main dependencies on the evaluation side (minimum set):

- Python 3.8+
- `torch`, `torchvision`
- `mmcv` (must include `mmcv.ops.multi_scale_deform_attn`)
- `xformers`
- `timm`
- `einops`
- `numpy`
- `opencv-python`
- `Pillow`
- `PyYAML`
- `python-box`
- `scikit-image`
- `lpips`
- `tqdm`
- `shapely`
- `matplotlib`
- `six`
- `termcolor`

Optional:

- `scipy` (rotation utilities)
- GroundingDINO and its deps (for text-prompt detection)
- `pycocotools` (some helper functions)
- `open3d` (dataset test scripts)

> Also: the code uses imports like `from utils.train_utils ...`, so the evaluation root must be on `PYTHONPATH`:
> `export PYTHONPATH=$PWD:$PYTHONPATH` (run from `evaluation/`).
> **Do not** set `PYTHONPATH=$PWD/utils` — that causes namespace collisions.

### pip install example

The following is an **example** install (pick PyTorch / mmcv versions matching your CUDA / driver):

```bash
# 1) PyTorch (replace per your CUDA version)
# e.g., CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2) mmcv (must include ops; recommended via openmim)
pip install -U openmim
mim install mmcv

# 3) Other deps
pip install numpy opencv-python pillow pyyaml python-box scikit-image lpips tqdm \
            timm einops shapely matplotlib six termcolor

# 4) xformers (matched to torch/cuda; missing → import error)
pip install xformers

# 5) Optional
pip install scipy pycocotools
```

## Output

In the `--output` directory you get:

- `infer_cache.json` (DetAny3D inference cache)
- `instruction-compliance_eval_results.json` / `spatial-accuracy_eval_results.json`
- `*_eval_stats.json` (aggregated statistics)

## FAQ

- **`--edit` has no effect**: the current code ignores this argument; the JSON must be in `--original` and match the naming rule.
- **Cannot find the config**: use `--config utils/detect_anything/configs/demo.yaml`.
- **Cannot find `detect_anything`**: use `PYTHONPATH=./utils` or a symlink.
