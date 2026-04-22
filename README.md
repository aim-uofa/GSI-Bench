# GSI-Bench: Generative Spatial Intelligence Benchmark

Official implementation of the paper:

> **Exploring Spatial Intelligence from a Generative Perspective**
> [[Paper]](paper/main.pdf)

GSI-Bench evaluates the ability of generative models to understand and manipulate 3D spatial relationships in indoor scenes.

| Metric | Full Name | What It Measures |
|--------|-----------|------------------|
| **IC** | Instruction Compliance | Whether the edit follows the instruction |
| **SA** | Spatial Accuracy | Spatial precision of the edit |
| **EL** | Edit Locality | Whether unedited regions remain intact |
| **AC** | Acceptance Consistency | MLLM-based holistic quality score |

---

## Quick Navigation

> **If you only want to evaluate your model on GSI-Bench, go directly to [Evaluation](#evaluation).**
>
> Steps 1 and 2 document how we constructed the benchmark data. They are open-sourced for transparency and reproducibility, but are **not required** for running evaluations.

```
GSI-Bench/
├── evaluation/     # Evaluation framework (IC / SA / EL / AC)  ← start here
├── robothor/       # [Optional] Data generation pipeline 1: RoboTHOR indoor scenes
├── mesatask/       # [Optional] Data generation pipeline 2: MesaTask tabletop scenes
├── paper/          # Paper PDF
└── tests/          # Unit & integration tests
```

---

## Evaluation

### 1. Environment Setup

```bash
conda create -n gsi-eval python=3.10 -y
conda activate gsi-eval

cd evaluation

# Install PyTorch matching your CUDA version (example: CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install mmcv with C++ ops
pip install -U openmim && mim install mmcv

# Install remaining dependencies
pip install -r requirements.txt

# Optional: build GroundingDINO for text-prompt detection
pip install -e ./src/groundingdino --no-build-isolation
```

### 2. Download Model Weights

| Weight | Size | Source |
|--------|------|--------|
| `other_exp_ckpt.pth` (DetAny3D) | ~500MB | Released with this project |
| `sam_vit_h_4b8939.pth` (SAM ViT-H) | ~2.4GB | [Meta AI](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| `dinov2_vitl14_pretrain.pth` (DINOv2) | ~1.1GB | [Meta AI](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) |
| `groundingdino_swinb_cogcoor.pth` (optional) | ~690MB | [IDEA-Research](https://github.com/IDEA-Research/GroundingDINO) |

Place all weights in one directory, then run:
```bash
bash prepare_weights.sh <path_to_weight_directory>
# Creates symlinks under checkpoints/ and GroundingDINO/weights/
```

### 3. Download Evaluation Datasets

```bash
# Download the four GSI-Bench evaluation datasets and place in one directory
bash prepare_datasets.sh <path_to_downloaded_datasets>
# Creates symlinks: fine_dataset/  mesatask_dataset/  bathroom_dataset/  robothor_dataset/
```

### 4. Generate Edited Images with Your Model

Your model should produce edited images following the naming convention:
```
eval/<model_name>/generated_images_fine/<img_id>_edit_<query_id>.png
eval/<model_name>/generated_images_mesatask/<img_id>_edit_<query_id>.png
eval/<model_name>/generated_images_bathroom/<img_id>_edit_<query_id>.png
eval/<model_name>/generated_images_robothor/<img_id>_edit_<query_id>.png
```

We provide a BAGEL-based example: `python examples/inference.py` (see [`evaluation/REPRODUCE_BAGEL_RESULTS.md`](evaluation/REPRODUCE_BAGEL_RESULTS.md)).

### 5. Run Evaluation

```bash
cd evaluation
export PYTHONPATH=$PWD:$PYTHONPATH

# IC / SA / EL evaluation (iterates all models × all datasets)
bash eval.sh

# (Optional) MLLM-based AC scoring — requires serving an LLM
cd mllm_eval
bash eval_infer.sh <model_path> default <port>
cd ..

# Aggregate all metrics into a final report
python -m eval.aggregate \
  --root-dir ./eval \
  --output-dir ./eval_results \
  --mllm-eval-dir <dir_with_mllm_ac_jsons>

cd ..   # back to repo root
```

**Output:** `eval_results/` with per-model, per-dataset JSON files containing IC/SA/EL/AC scores.

See [`evaluation/eval/README.md`](evaluation/eval/README.md) for detailed input format and troubleshooting.

---

## Data Generation Pipelines (Optional)

> The following two pipelines document how we constructed the GSI-Bench data. They are **not needed for evaluation** — the evaluation datasets are provided as downloads above.

### Pipeline 1: RoboTHOR Indoor Scenes

**Environment:**
```bash
conda create -n gsi-robothor python=3.10 -y
conda activate gsi-robothor
pip install -r robothor/requirements.txt
# Dependencies: ai2thor>=5.0.0, numpy, Pillow, matplotlib
# AI2-THOR downloads scene assets automatically on first run (~2GB)
# Requires: NVIDIA GPU + CloudRendering (headless) or X server (display)
```

**Generate data:**
```bash
cd robothor

# 1) Generate base views + camera-relative commands for ALL 60 training scenes
#    Output: data/outputs/train/with_physics/
bash scripts/generate_train.sh

# 2) Generate additional command types (requires pregenerated views from step 1)
bash scripts/generate_train_object.sh          # object-relative positioning
bash scripts/generate_train_rotate.sh           # rotation commands
bash scripts/generate_train_receptacle.sh       # receptacle placement
bash scripts/generate_train_spatial_remove.sh    # spatial removal
bash scripts/generate_train_agent_camera.sh      # agent camera movement

# 3) Generate validation data
bash scripts/generate_val_agent_camera.sh

cd ..   # back to repo root
```

**Output:** `data/outputs/{train,val}/` with JSONL records + RGB/depth/segmentation images per view per command.

**Timing:** ~2–5 min per scene depending on GPU. Full 60 scenes: several hours.

See [`robothor/README.md`](robothor/README.md) for details.

---

### Pipeline 2: MesaTask Tabletop Scenes

**Environment:**
```bash
conda create -n gsi-mesatask python=3.10 -y
conda activate gsi-mesatask
pip install -r mesatask/requirement.txt
# For inference (optional): pip install torch torchvision
# For rendering (optional): download Blender 4.3+ from https://www.blender.org/download/
# For physical optimization (optional): conda install -c conda-forge drake
```

**Download MesaTask-10K dataset:**
```bash
cd mesatask
git lfs install
git clone https://huggingface.co/datasets/InternRobotics/MesaTask-10K MesaTask-10K

# Prepare asset library (from dataset archives)
cd MesaTask-10K/Assets_library_archive
cat Assets_library_backup.tar.gz.* > Assets_library_merged.tar.gz
tar -xzvf Assets_library_merged.tar.gz -C ../Assets_library/
cd ../..
```

**Generate data:**
```bash
cd mesatask

# 1) Generate atomic transforms (move, rotate, scale)
python generate_atomic_transforms.py \
  --input-dir MesaTask-10K/Layout_info \
  --asset-annotation MesaTask-10K/Asset_annotation.json \
  --output-dir transformed_layouts \
  --num-variants 10 --seed 42

# 2) Render all layouts (requires Blender)
python dataset/vis_batch.py transformed_layouts \
  --output_dir dataset/vis_final --parallel 4

# 3) Assemble image-editing dataset
python organize_image_editing_dataset.py \
  --transformed-dir transformed_layouts \
  --vis-dir dataset/vis_final \
  --output-dir dataset/image_editing_dataset

cd ..   # back to repo root
```

**Timing:** Step 1 takes ~10 min for 10K scenes. Step 2 (rendering) depends on machine and parallelism.

See [`mesatask/README.md`](mesatask/README.md) for details.

---

## Verify the Repo

```bash
git clone <this-repo-url> GSI-Bench && cd GSI-Bench

# Run tests (no GPU or data needed)
pip install pytest
python -m pytest tests/ -v    # 43 tests should pass
```

## Environment Requirements Summary

| Component | Python | GPU | Conda Env |
|-----------|--------|-----|-----------|
| **tests/** | 3.8+ | No | any |
| **evaluation/** | 3.10 | NVIDIA (DetAny3D) | `gsi-eval` |
| **robothor/** | 3.10 | NVIDIA (CloudRendering) | `gsi-robothor` |
| **mesatask/** | 3.10 | Optional | `gsi-mesatask` |

---

## Citation

```bibtex
@article{gsibench2025,
  title={Exploring Spatial Intelligence from a Generative Perspective},
  year={2025}
}
```

## License

GSI-Bench is released under the MIT License — see [`LICENSE`](LICENSE).

Subdirectories containing code derived from third-party projects retain their
own licenses:

- [`robothor/LICENSE`](robothor/LICENSE)
- [`mesatask/LICENSE`](mesatask/LICENSE)
