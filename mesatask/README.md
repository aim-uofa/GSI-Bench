# MesaTask: Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning

Language: English | Chinese in `README_zh.md`

MesaTask is a research repository for task-driven tabletop scene generation. It provides:
- The MesaTask-10K dataset layout files and asset annotations
- An inference pipeline to generate 3D scenes from natural-language tasks
- Visualization and rendering utilities (Blender-based)
- Tools to generate atomic transforms and image-editing pairs for instruction-following data

For detailed atomic-transform generation, see `README_transforms.md`.

## Repository layout

- `MesaTask-10K/`: dataset root (layouts, annotations, assets, model)
- `generate_atomic_transforms.py`: atomic transforms generator
- `instruction_templates.py`: bilingual instruction templates
- `dataset/vis_single.py`, `dataset/vis_batch.py`: rendering and visualization
- `organize_image_editing_dataset.py`: assemble image-editing dataset (wrapper)
- `oraganize_image_editing_dataset.py`: legacy spelling, still supported
- `get_task_info.py`, `inference.py`: inference pipeline
- `config.yaml`: rendering and asset paths

## Environment setup

Recommended: Python 3.10

1) Create and activate environment
```bash
conda create -n MesaTask python=3.10
conda activate MesaTask
```

2) Install dependencies
```bash
# Core dependencies
pip install -r requirement.txt

# For inference (optional but recommended)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

3) Install Blender (for rendering)
```bash
wget https://download.blender.org/release/Blender4.3/blender-4.3.2-linux-x64.tar.xz
tar -xvJf blender-4.3.2-linux-x64.tar.xz
```

Update `config.yaml` to point to your Blender executable, e.g.:
```yaml
blender_executable: "./blender-4.3.2-linux-x64/blender"
```

If you only need data generation without rendering, Blender is not required.

## Data preparation

1) Download MesaTask-10K from Hugging Face.
2) Place the dataset at `./MesaTask-10K/` (repo root).
3) Prepare the asset library:
```bash
cd MesaTask-10K
mkdir -p Assets_library

cd Assets_library_archive
cat Assets_library_backup.tar.gz.* > Assets_library_merged.tar.gz

tar -xzvf Assets_library_merged.tar.gz -C ../Assets_library/

cd ..
rm -r Assets_library_archive
```

Expected dataset structure:
```text
MesaTask-10K/
|-- MesaTask_model
|-- Asset_annotation.json
|-- sbert_text_features.pkl
|-- Assets_library/
|-- Layout_info/
|-- ...
```

## Reproducible pipeline (atomic transforms + rendering + image-editing pairs)

All commands below are run from repo root:

1) Generate atomic transforms (set a fixed seed for reproducibility)
```bash
python generate_atomic_transforms.py \
  --input-dir MesaTask-10K/Layout_info \
  --asset-annotation MesaTask-10K/Asset_annotation.json \
  --output-dir transformed_layouts5 \
  --num-variants 10 \
  --seed 123
```

2) Render all generated layouts (requires Blender)
```bash
python dataset/vis_batch.py transformed_layouts5 \
  --output_dir dataset/vis_final \
  --parallel 4
```

3) Build image-editing dataset
```bash
python organize_image_editing_dataset.py \
  --transformed-dir transformed_layouts5 \
  --vis-dir dataset/vis_final \
  --output-dir dataset/image_editing_dataset
```

Reproducibility tips:
- Fix `--seed` for transform generation
- Keep the same dataset version, Python version, and Blender version
- Avoid modifying layout JSONs between steps

## Visualization (optional)

Single layout:
```bash
python dataset/vis_single.py MesaTask-10K/Layout_info/dining_table/dining_table_0000/layout.json \
  --output_dir dataset/vis_data
```

Batch rendering:
```bash
python dataset/vis_batch.py MesaTask-10K/Layout_info \
  --output_dir dataset/vis_data \
  --parallel 4
```

More details: `dataset/README_visualization.md`

## Inference pipeline

1) Generate task information
```bash
python get_task_info.py \
  --task_name "Organize books and magazines on the table" \
  --table_type "Nightstand" \
  --api_key "your_api_key" \
  --model "gpt-4o" \
  --output_dir "output"
```

2) Generate and render the scene
```bash
python inference.py \
  --input_file output/task_001/task_info.json \
  --mesatask_model_path ./MesaTask-10K/MesaTask_model \
  --rendering
```

Optional physical optimization:
```bash
python tools/layoutopt/glb2obj.py \
  --glb_dir ./MesaTask-10K/Assets_library \
  --obj_dir ./MesaTask-10K/Assets_library_obj \
  --max_workers 16

python inference.py \
  --input_file output/task_001/task_info.json \
  --mesatask_model_path ./MesaTask-10K/MesaTask_model \
  --physical_optimization \
  --rendering
```

## Citation

```text
@misc{hao2025mesatask,
  title={MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning},
  author={Hao, Jinkun and Liang, Naifu and Luo, Zhen and Xu, Xudong and Zhong, Weipeng and Yi, Ran and Jin, Yichen and Lyu, Zhaoyang and Zheng, Feng and Ma, Lizhuang and Pang, Jiangmiao},
  journal={arXiv preprint arXiv:2509.22281},
  year={2025}
}
```

## License

Apache License.
