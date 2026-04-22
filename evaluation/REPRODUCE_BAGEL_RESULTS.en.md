# Reproducing BAGEL Results

Language: English | 中文见 [`REPRODUCE_BAGEL_RESULTS.md`](REPRODUCE_BAGEL_RESULTS.md)

This document explains how to reproduce the full BAGEL evaluation results on the `fine` dataset using the `bagel_example/` artifacts shipped with this repo.

> **Bottom line:**
> - **"Reproducing the final evaluation scores from existing generated images" is fully supported.**
> - **"Regenerating BAGEL edited images from the original model" requires extra BAGEL inference code**, because `examples/inference.py` depends on `data/`, `modeling/`, `inferencer.py`, etc., which are not shipped in full in this repo.

---

## 1. `bagel_example/` layout

We provide the complete BAGEL × fine-dataset intermediate artifacts so you can resume reproduction from any point.

```
evaluation/bagel_example/
├── fine_dataset/                  # Eval dataset: 211 source images + 441 edit JSONs (~64MB)
├── generated_images_fine/         # 441 BAGEL-generated edited images (~195MB)
├── BAGEL/
│   └── fine_eval/                 # IC / SA / EL results (incl. DetAny3D inference cache, ~7MB)
├── predictions_infer_2000_Qwen3-VL-235B-A22B-Instruct_fine_BAGEL.json
│                                  # Qwen3-VL AC scores (192KB)
└── agg_results/                   # Final aggregated scores (for cross-check, 24KB)
```

**Three reproduction paths:**

| Path | Starting point | GPU required | Steps |
|------|---------------|--------------|-------|
| **A. Full reproduction** | `fine_dataset/` + `generated_images_fine/` | Yes | Run IC/SA/EL → build LMDB → MLLM scoring → aggregate |
| **B. Skip evaluation** | `BAGEL/fine_eval/` + `predictions_*.json` | No | Aggregate directly |
| **C. Inspect only** | `agg_results/` | No | Compare JSON values |

### Verified reference values

```
IC:       31.973
SA:       22.185
EL.ssim:  28.748
EL.lpips: 27.894
AC:       31.882
Average:  28.484
```

---

## 2. Reproduction scope

### 2.1 What works out of the box (this repo + `bagel_example/`)

1. Set up the environment (Section 3)
2. Prepare datasets and weights (Sections 4–5, or use `bagel_example/fine_dataset/` directly)
3. Run IC / SA / EL on `generated_images_*` (Section 6)
4. Build LMDB + MLLM scoring to produce AC JSON (Section 9)
5. Aggregate the final score (Section 7)

### 2.2 What needs extra code

1. **Regenerating BAGEL edits from scratch**: `examples/inference.py` depends on `data.transforms`, `data.data_utils`, `modeling.*`, `inferencer.py` — these require the original BAGEL project code.
2. **MLLM AC inference**: requires a vLLM service and enough GPU resources to serve something like Qwen3-VL.

---

## 3. Environment setup from scratch

### 3.1 Create the conda env

We provide a recommended script:

```bash
cd ${REPO_ROOT}
bash setup_gsi_env.sh
```

To rebuild:

```bash
cd ${REPO_ROOT}
bash setup_gsi_env.sh --recreate
```

It will:

- Create a `gsi` env
- Install `python=3.10`
- Install `torch + torchvision + pytorch-cuda=11.7`
- Install `cuda-toolkit=11.7` and `cuda-nvcc=11.7`
- Install `requirements-cu117.txt`
- Try to install the local `src/groundingdino`

### 3.2 Activate

```bash
conda activate gsi
```

### 3.3 Set PYTHONPATH

The evaluation code imports `detect_anything`, whose package lives under `utils/detect_anything`. Run before evaluation:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

### 3.4 Sanity check

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import mmcv; print(mmcv.__version__)"
python -c "import lpips; print('lpips ok')"
```

If any of these fail, downstream evaluation is unlikely to work.

---

## 4. Dataset setup from scratch

### 4.1 Expected datasets

The repo expects four datasets by default:

- `fine_dataset`
- `mesatask_dataset`
- `bathroom_dataset`
- `robothor_dataset`

Place raw zips or extracted directories under a common parent, e.g.:

- `${REPO_ROOT}/GSI-Bench`

### 4.2 Create symlinks

A helper script is provided:

```bash
cd ${REPO_ROOT}
bash prepare_datasets.sh ${REPO_ROOT}/GSI-Bench
```

To overwrite existing symlinks:

```bash
bash prepare_datasets.sh ${REPO_ROOT}/GSI-Bench --force
```

### 4.3 What the script does

- Tries to unzip `fine_dataset.zip` / `mesatask_dataset.zip` / `bathroom_dataset.zip` / `robothor_dataset.zip`
- Or uses already-extracted directories
- Creates at the repo root:
  - `fine_dataset`
  - `mesatask_dataset`
  - `bathroom_dataset`
  - `robothor_dataset`

### 4.4 Quick validation (with `bagel_example`)

If you only want BAGEL × fine reproduction, skip the above and use `bagel_example/fine_dataset/` directly.

---

## 5. Weight setup from scratch

### 5.1 Required weights

- `other_exp_ckpt.pth`
- `sam_vit_h_4b8939.pth`
- `dinov2_vitl14_pretrain.pth`
- `GroundingDINO_SwinB_cfg.py`
- `groundingdino_swinb_cogcoor.pth`

### 5.2 Symlink the weights

```bash
cd ${REPO_ROOT}
bash prepare_weights.sh ${REPO_ROOT}/GSI-weight
```

To overwrite existing links:

```bash
bash prepare_weights.sh ${REPO_ROOT}/GSI-weight --force
```

### 5.3 Verify

```bash
ls checkpoints/detany3d/detany3d_ckpts/other_exp_ckpt.pth
ls checkpoints/detany3d/sam_ckpts/sam_vit_h_4b8939.pth
ls checkpoints/detany3d/dino_ckpts/dinov2_vitl14_pretrain.pth
```

---

## 6. Run IC / SA / EL (Path A)

Re-run the evaluation on the data and images in `bagel_example/`. **Requires GPU and the eval weights.**

```bash
cd evaluation
export PYTHONPATH=$PWD:$PYTHONPATH
conda activate gsi

# Use bagel_example data
bash eval_one.sh \
  ./bagel_example/BAGEL \
  ./bagel_example/fine_dataset \
  instruction-compliance,spatial-accuracy,edit-locality
```

Results are written to `bagel_example/BAGEL/fine_eval/`:

- `instruction-compliance_eval_results.json`
- `spatial-accuracy_eval_results.json`
- `edit-locality_eval_results.json`
- and the corresponding `*_eval_stats.json`

> **Batch evaluation:** for multiple models, drop images under `eval/<model_name>/generated_images_fine/` and run `bash eval.sh`.

---

## 7. Aggregate the final scores (Path A / B)

### 7.1 Aggregate using bagel_example

```bash
cd evaluation
export PYTHONPATH=$PWD:$PYTHONPATH

python -m eval.aggregate \
  --root-dir ./bagel_example \
  --output-dir ./bagel_example/my_agg_results \
  --mllm-eval-dir ./bagel_example
```

### 7.2 Argument semantics

- `--root-dir`: the parent of the per-model directories. The script iterates subdirectories as model names and looks up `*_eval/`.
- `--mllm-eval-dir`: directory holding AC prediction JSONs. The script globs `*{dataset}_{model}.json`.

### 7.3 Output

Under `my_agg_results/`:

- `EVAL_output_summary.json` (all metrics)
- `EVAL_output_ac.json` / `EVAL_output_ic.json` / `EVAL_output_sa.json` / `EVAL_output_edit_locality.json`
- `EVAL_output_average.json`

### 7.4 Cross-check

Compare against `bagel_example/agg_results/` — values should match exactly:

```
IC:       31.973
SA:       22.185
EL.ssim:  28.748
EL.lpips: 27.894
AC:       31.882
Average:  28.484
```

---

## 8. If you want to "reproduce starting from image generation"

Two sub-cases to distinguish.

### 8.1 Reproduce evaluation from **existing** images

This path is complete:

1. `generated_images_fine/`
2. `bash eval_one.sh ...`
3. `python -m eval.aggregate ...`

### 8.2 Regenerate BAGEL images via `examples/inference.py`

**Not guaranteed to run out of the box.**

The repo ships:

- `examples/inference.py`
- `examples/mydataset.py`

But `examples/inference.py` still depends on modules that are missing or incomplete in this repo:

- `data.transforms`
- `data.data_utils`
- `modeling.bagel`
- `modeling.qwen2`
- `modeling.autoencoder`
- `inferencer.py`

So to truly "reproduce `generated_images_fine/` from model inference", you additionally need:

1. The original BAGEL project code
2. A usable BAGEL model directory
3. The full inference dependencies

In other words, **this repo is closer to "evaluation repo + an incomplete BAGEL inference skeleton"** than to a full BAGEL generation project.

---

## 9. AC scoring (optional)

AC (Acceptance Consistency) uses an MLLM (e.g., Qwen3-VL) to rate source vs. edited images. Full pipeline is three steps:

### 9.1 Build an LMDB dataset

Use `mllm_eval/build_lmdb.py` to pack source, edited, and prompt into LMDB:

```bash
cd evaluation/mllm_eval

python build_lmdb.py \
  --dataset-dir ../bagel_example/fine_dataset \
  --generated-dir ../bagel_example/generated_images_fine \
  --output-dir ./EVAL_lmdb_dataset/fine_BAGEL \
  --dataset-name fine \
  --model-name BAGEL
```

Deps: `pip install lmdb opencv-python`

### 9.2 Serve vLLM and run inference

```bash
bash eval_infer.sh <path_to_qwen3_vl_model> default 8000
```

This starts a vLLM service and runs `mllm_eval.py` to score each image pair in the LMDB, producing `predictions_*.json`.

Requires:
- vLLM (`pip install vllm`)
- Qwen-VL utils (`pip install qwen-vl-utils`)
- Enough GPU to serve Qwen3-VL or a similar MLLM

### 9.3 Skip AC inference

If you do not want to stand up vLLM, use the AC JSON already shipped in `bagel_example/`:

```
bagel_example/predictions_infer_2000_Qwen3-VL-235B-A22B-Instruct_fine_BAGEL.json
```

Pass it to `eval.aggregate` via `--mllm-eval-dir`.

---

## 10. Fastest reproduction (no GPU)

If you only want to verify the aggregated scores, three commands suffice:

```bash
cd evaluation
export PYTHONPATH=$PWD:$PYTHONPATH

# Aggregate existing evaluation results in bagel_example
python -m eval.aggregate \
  --root-dir ./bagel_example \
  --output-dir ./bagel_example/my_agg_results \
  --mllm-eval-dir ./bagel_example

# Inspect the result
cat ./bagel_example/my_agg_results/EVAL_output_summary.json

# Compare with the reference
cat ./bagel_example/agg_results/EVAL_output_summary.json
```

The two JSONs should match exactly.

---

## 11. Common pitfalls

### 11.1 `detect_anything` import fails

Fix:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

### 11.2 Aggregator cannot find models

`--root-dir` must point to the *parent* of model directories. If your results are in `./bagel_example/BAGEL/fine_eval/`, pass `--root-dir ./bagel_example` (not `./bagel_example/BAGEL`).

### 11.3 `examples/inference.py` reports missing modules

That is not your environment — this repo intentionally does not ship the full BAGEL inference stack.

### 11.4 Using `mllm_eval/eval_infer.sh` directly

The script can start vLLM and run inference, but you must prepare `./EVAL_lmdb_dataset` yourself first. If you only want the final scores, just use the shipped AC JSON together with `python -m eval.aggregate`.

---

## 12. Summary

`bagel_example/` provides the full chain of intermediate artifacts, so BAGEL × fine scores can be reproduced from any starting point:

- **No GPU**: aggregate existing `fine_eval/` + `predictions_*.json` (Path B)
- **With GPU**: re-run IC/SA/EL on `fine_dataset/` + `generated_images_fine/` (Path A)
- **With GPU + vLLM**: also re-run AC scoring (Section 9)

The only part that cannot be completed inside this repo is "generating BAGEL edited images from scratch" — this requires the original BAGEL inference code (`data.*`, `modeling.*`, `inferencer.py`); see [BAGEL](https://github.com/ByteDance-Seed/BAGEL).
