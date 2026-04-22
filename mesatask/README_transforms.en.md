# Scene Object Transform Generator

Language: English | 中文见 [`README_transforms.md`](README_transforms.md)

Automatically batch-generate atomic transforms (move, rotate, scale) for objects in 3D scenes and produce bilingual Chinese/English instructions.

## Features

- ✅ **Batch processing** — iterates all scene types and scene IDs automatically
- ✅ **Three atomic ops** — move (5–20 cm), rotate (±45° / ±90° / ±180°), scale (0.5× / 0.75× / 1.5× / 2×)
- ✅ **Bilingual instructions** — each transform generates both Chinese and English instructions with 4 phrasings per language
- ✅ **Tabletop enlargement** — randomly scales the tabletop by 1.0–2.0× to add empty space
- ✅ **Collision detection** — 3D-IoU-based collision check, retries up to 50 times to avoid overlaps
- ✅ **Disambiguating selection** — only picks objects whose category is unique in the scene, avoiding ambiguous instructions
- ✅ **Metadata logging** — records op details, collision results, original values, etc.

## Directory layout

```
MesaTask/
├── generate_atomic_transforms.py   # main entry
├── instruction_templates.py        # Chinese/English instruction templates
├── MesaTask-10K/
│   ├── Layout_info/               # input: all scenes
│   │   ├── dining_table/
│   │   ├── coffee_table/
│   │   ├── bathroom_vanity/
│   │   └── ...
│   └── Asset_annotation.json      # object annotations
└── transformed_layouts5/          # output directory
    ├── dining_table/
    │   ├── dining_table_0000/
    │   │   ├── origin_layout/
    │   │   │   └── layout_enlarged_1.8x_1.6y.json
    │   │   ├── variants/
    │   │   │   ├── 001_teapot_0_move.json
    │   │   │   ├── 002_bowl_1_rotate.json
    │   │   │   └── ...
    │   │   └── transform_descriptions.jsonl
    │   └── ...
    └── ...
```

## Usage (recommended: run from the repo root)

### Quick reproduction (step by step)

```bash
# 1) Generate atomic transforms (fix the seed for reproducibility)
python generate_atomic_transforms.py \
  --input-dir MesaTask-10K/Layout_info \
  --asset-annotation MesaTask-10K/Asset_annotation.json \
  --output-dir transformed_layouts5 \
  --num-variants 10 \
  --seed 123

# 2) Batch render (requires Blender and config.yaml)
python dataset/vis_batch.py transformed_layouts5 \
  --output_dir dataset/vis_final \
  --parallel 4

# 3) Assemble the image-editing dataset
python organize_image_editing_dataset.py \
  --transformed-dir transformed_layouts5 \
  --vis-dir dataset/vis_final \
  --output-dir dataset/image_editing_dataset
```

### 1. Generate atomic transforms

```bash
python generate_atomic_transforms.py
```

Common optional arguments:

- `--input-dir`: scene input directory (default `MesaTask-10K/Layout_info`)
- `--asset-annotation`: annotation file (default `MesaTask-10K/Asset_annotation.json`)
- `--output-dir`: output directory (default `transformed_layouts5`)
- `--num-variants`: variants per scene
- `--seed`: random seed (specify for reproducibility)

The program will:
1. Iterate all scene types and scene IDs
2. Enlarge the tabletop and generate multiple variants per scene
3. Print progress and statistics

### 2. Render images (vis_batch)

```bash
python dataset/vis_batch.py transformed_layouts5 --output_dir dataset/vis_final --parallel 4
```

See `dataset/README_visualization.md` for Blender config and render parameters.

### 3. Assemble the image-editing dataset

```bash
python organize_image_editing_dataset.py \
  --transformed-dir transformed_layouts5 \
  --vis-dir dataset/vis_final \
  --output-dir dataset/image_editing_dataset
```

> Legacy script name (still supported): `oraganize_image_editing_dataset.py`.

### 4. Inspect the output

Each scene produces:
- `origin_layout/` — the enlarged original layout
- `variants/` — all transformed layouts
- `transform_descriptions.jsonl` — metadata log of transforms

## Output format

### Layout JSON

Each transformed layout contains:

```json
{
  "scene_settings": { ... },
  "item_placement_zone": [xmin, xmax, ymin, ymax],
  "objects": [ ... ],
  "instruction_zh": "请将茶壶（ceramic teapot）向右移动15厘米。",
  "instruction_en": "Move the teapot (ceramic teapot) 15 centimeters to the right.",
  "operation_meta": {
    "target_object": {
      "instance": "1_teapot_0",
      "name": "1_teapot_0_xxx",
      "category": "teapot",
      "caption": "ceramic teapot with floral design",
      "object_index": 3
    },
    "operation": "move",
    "direction": "right",
    "delta_cm": 15.0,
    "old_position": [64.1, 59.4, 9.6],
    "collision_check": {
      "passed": true,
      "attempts": 1,
      "max_iou": 0.0
    }
  }
}
```

### JSONL log

`transform_descriptions.jsonl` records per-transform metadata:

```jsonl
{"file": "001_teapot_0_move.json", "instruction_zh": "...", "instruction_en": "...", "operation_meta": {...}}
{"file": "002_bowl_1_rotate.json", "instruction_zh": "...", "instruction_en": "...", "operation_meta": {...}}
```

## Core features

### 1. Atomic operations

**Move**
- Directions: left, right, forward, backward
- Distance: random 5–20 cm
- Bounded by the tabletop placement zone

**Rotate**
- Axis: around Z
- Angles: ±45°, ±90°, ±180° (random)
- Represented as a quaternion

**Scale**
- Factors: 0.5×, 0.75× (shrink) or 1.5×, 2× (enlarge)
- Updates both `scale_factor` and `size` fields

### 2. Tabletop enlargement

```python
# Random enlargement by 1.0–2.0×
scale_x = random.uniform(1.0, 2.0)
scale_y = random.uniform(1.0, 2.0)

# Translate all objects to the new area center, preserving relative layout
offset_x = (new_width - orig_width) / 2
offset_y = (new_height - orig_height) / 2
```

### 3. Collision detection

3D bounding-box IoU check:

```python
# 3D bounding box
bbox = [xmin, xmax, ymin, ymax, zmin, zmax]

# 3D IoU
iou = intersection_volume / union_volume

# Threshold (default 0.0, no overlap allowed)
if iou > threshold:
    collision_detected = True
```

Retries up to 50 times until a collision-free transform is found.

### 4. Disambiguating selection

Only pick objects whose category is unique in the scene:

```python
# Count per-category
category_count = {"teapot": 1, "plate": 3, "bowl": 2}

# Pick only teapot (unique — unambiguous)
# Skip plate / bowl (multiple instances — ambiguous)
```

### 5. Bilingual templates

Each operation has 4 Chinese and 4 English phrasings:

```python
"move_right": {
    "zh": [
        "请将{obj}向右移动{value}厘米。",
        "请把{obj}往右边挪动{value}厘米。",
        "将{obj}向右平移{value}厘米。",
        "把{obj}朝右移{value}厘米。"
    ],
    "en": [
        "Move the {obj} {value} centimeters to the right.",
        "Shift the {obj} {value} cm to the right.",
        "Please move the {obj} rightward by {value} centimeters.",
        "Relocate the {obj} {value} cm to the right."
    ]
}
```

## Configuration

### Tunable CLI args

`generate_atomic_transforms.py` supports:

- `--max-attempts`: max collision-retry count (default 50)
- `--iou-threshold`: IoU threshold (default 0.0, no overlap)
- `--enlarge-min / --enlarge-max`: tabletop enlargement range (default 1.0–2.0)
- `--move-min-cm / --move-max-cm`: move distance range (default 5–20 cm)
- `--rotate-degrees`: candidate rotation angles
- `--scale-factors`: candidate scale factors

## Dependencies

```python
import json
import random
import math
from copy import deepcopy
from pathlib import Path
```

Standard library only — nothing extra to install.

## Statistics

After a run you'll see:

```
============================================================
Batch processing complete!
Total scenes processed: 13787
Total transforms generated: 137870
============================================================
```

## Notes

1. **Annotation dependency** — requires `Asset_annotation.json` with category and detailed_caption for each object.
2. **Category uniqueness** — if every category in a scene has duplicates, falls back to random selection.
3. **Collision retries** — after the max retry count, the last attempt is kept and a warning is recorded in metadata.
4. **Coordinate system** — centimeters; Z axis up.
5. **Rotation representation** — quaternion [x, y, z, w].
6. **Reproducibility** — fix `--seed` and use the same Python version + dataset to reproduce outputs.

## Extending

### Add a new atomic operation

1. Implement the op:

```python
def new_operation(obj, table_bounds, asset_info):
    new_obj = deepcopy(obj)
    # ... op logic ...

    # Generate instructions
    template_zh = random.choice(TEMPLATES["operation_key"]["zh"])
    template_en = random.choice(TEMPLATES["operation_key"]["en"])
    instr_zh = template_zh.format(obj=obj_desc, value=value)
    instr_en = template_en.format(obj=obj_desc, value=value)

    # Metadata
    meta = {
        "operation": "operation_name",
        # ... other fields ...
    }

    return new_obj, instr_zh, instr_en, meta
```

2. Add templates to `instruction_templates.py`.

3. Add the op to the list in `generate_variant()`:

```python
operation = random.choice([move_object, rotate_object, scale_object, new_operation])
```

### Change the object-selection policy

Edit `get_unique_category_objects()`:

```python
def get_unique_category_objects(objects):
    # custom selection logic
    # e.g., only certain categories, filter by size, etc.
    ...
```

## License

Add the appropriate license per project needs.
