# 场景物体变换生成器

自动批量生成3D场景中物体的原子变换（移动、旋转、缩放），并生成中英文双语指令。

## 功能特性

- ✅ **批量处理** - 自动处理所有场景类型和场景编号
- ✅ **三种原子操作** - 移动（5-20cm）、旋转（±45°/±90°/±180°）、缩放（0.5x/0.75x/1.5x/2x）
- ✅ **双语指令** - 每个变换生成中英文两种指令，每种语言4种表达方式
- ✅ **桌面扩展** - 随机扩大桌面1.0-2.0倍，增加空余空间
- ✅ **碰撞检测** - 基于3D IOU的碰撞检测，最多重试50次避免物体重叠
- ✅ **消歧义选择** - 只选择场景中类别唯一的物体，避免指令歧义
- ✅ **元数据记录** - 记录操作详情、碰撞检查结果、原始值等完整信息

## 目录结构

```
MesaTask/
├── generate_atomic_transforms.py   # 主程序
├── instruction_templates.py        # 中英文指令模板库
├── MesaTask-10K/
│   ├── Layout_info/               # 输入：所有场景
│   │   ├── dining_table/
│   │   ├── coffee_table/
│   │   ├── bathroom_vanity/
│   │   └── ...
│   └── Asset_annotation.json      # 物体标注信息
└── transformed_layouts5/          # 输出目录
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

## 使用方法（推荐从仓库根目录执行）

### 快速复现（一步一步）

```bash
# 1) 生成原子变换（固定 seed 可复现）
python generate_atomic_transforms.py \
  --input-dir MesaTask-10K/Layout_info \
  --asset-annotation MesaTask-10K/Asset_annotation.json \
  --output-dir transformed_layouts5 \
  --num-variants 10 \
  --seed 123

# 2) 批量渲染（需要 Blender 与 config.yaml）
python dataset/vis_batch.py transformed_layouts5 \
  --output_dir dataset/vis_final \
  --parallel 4

# 3) 组织图像编辑数据集
python organize_image_editing_dataset.py \
  --transformed-dir transformed_layouts5 \
  --vis-dir dataset/vis_final \
  --output-dir dataset/image_editing_dataset
```

### 1. 生成原子变换

```bash
python generate_atomic_transforms.py
```

可选参数（常用）：

- `--input-dir`：场景输入目录（默认 `MesaTask-10K/Layout_info`）
- `--asset-annotation`：标注文件（默认 `MesaTask-10K/Asset_annotation.json`）
- `--output-dir`：输出目录（默认 `transformed_layouts5`）
- `--num-variants`：每个场景生成多少组变换
- `--seed`：随机种子（指定后可复现）

程序会自动：
1. 遍历所有场景类型和场景编号
2. 为每个场景扩大桌面并生成多组变换
3. 输出进度信息和统计结果

### 2. 渲染图片（vis_batch）

```bash
python dataset/vis_batch.py transformed_layouts5 --output_dir dataset/vis_final --parallel 4
```

详见 `dataset/README_visualization.md`（包含 Blender 配置与渲染参数）。

### 3. 组织图像编辑数据集

```bash
python organize_image_editing_dataset.py \
  --transformed-dir transformed_layouts5 \
  --vis-dir dataset/vis_final \
  --output-dir dataset/image_editing_dataset
```

> 兼容脚本名：`oraganize_image_editing_dataset.py`（旧拼写）。

### 4. 查看结果

每个场景生成：
- `origin_layout/` - 扩大后的原始layout
- `variants/` - 所有变换后的layout
- `transform_descriptions.jsonl` - 变换元数据日志

## 输出格式

### Layout JSON

每个变换后的layout包含：

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

### JSONL 日志

`transform_descriptions.jsonl` 记录每个变换的元数据：

```jsonl
{"file": "001_teapot_0_move.json", "instruction_zh": "...", "instruction_en": "...", "operation_meta": {...}}
{"file": "002_bowl_1_rotate.json", "instruction_zh": "...", "instruction_en": "...", "operation_meta": {...}}
```

## 核心功能说明

### 1. 原子操作

**移动（move）**
- 方向：left, right, forward, backward
- 距离：5-20厘米随机
- 边界限制：不超出桌面placement zone

**旋转（rotate）**
- 轴向：绕Z轴旋转
- 角度：±45°, ±90°, ±180° 随机选择
- 使用四元数表示旋转

**缩放（scale）**
- 倍数：0.5x, 0.75x（缩小）或 1.5x, 2x（放大）
- 同时更新 scale_factor 和 size 字段

### 2. 桌面扩展

```python
# 随机扩大1.0-2.0倍
scale_x = random.uniform(1.0, 2.0)
scale_y = random.uniform(1.0, 2.0)

# 物体整体平移到新区域中心，保持相对位置不变
offset_x = (new_width - orig_width) / 2
offset_y = (new_height - orig_height) / 2
```

### 3. 碰撞检测

基于3D边界框的IOU检测：

```python
# 计算物体的3D边界框
bbox = [xmin, xmax, ymin, ymax, zmin, zmax]

# 计算3D IOU
iou = intersection_volume / union_volume

# 阈值检查（默认0.0，不允许任何重叠）
if iou > threshold:
    collision_detected = True
```

最多重试50次，直到找到无碰撞的变换。

### 4. 消歧义选择

只选择场景中类别唯一的物体：

```python
# 统计类别数量
category_count = {"teapot": 1, "plate": 3, "bowl": 2}

# 只选择 teapot（类别唯一，无歧义）
# 不选择 plate 或 bowl（有多个，会产生歧义）
```

### 5. 双语指令模板

每种操作有4种中文和4种英文表达方式：

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

## 配置选项

### 可调参数（命令行）

`generate_atomic_transforms.py` 支持以下常用参数：

- `--max-attempts`：碰撞检测最大重试次数（默认 50）
- `--iou-threshold`：IOU 阈值（默认 0.0，不允许重叠）
- `--enlarge-min / --enlarge-max`：桌面扩展倍数范围（默认 1.0~2.0）
- `--move-min-cm / --move-max-cm`：移动距离范围（默认 5~20 cm）
- `--rotate-degrees`：可选旋转角度列表
- `--scale-factors`：可选缩放倍数列表

## 依赖项

```python
import json
import random
import math
from copy import deepcopy
from pathlib import Path
```

标准库，无需额外安装。

## 统计信息

运行完成后会显示：

```
============================================================
批量处理完成!
总共处理: 13787 个场景
总共生成: 137870 个变换
============================================================
```

## 注意事项

1. **物体标注依赖** - 需要 `Asset_annotation.json` 提供物体的 category 和 detailed_caption
2. **类别唯一性** - 如果场景中所有物体类别都有重复，会fallback到随机选择
3. **碰撞检测** - 达到最大重试次数后会使用最后一次结果，但在metadata中标记warning
4. **坐标系统** - 使用厘米作为单位，Z轴向上
5. **旋转表示** - 使用四元数 [x, y, z, w] 表示旋转
6. **可复现性** - 固定 `--seed`，并使用相同的 Python 版本与数据集，可复现同样输出

## 扩展开发

### 添加新的原子操作

1. 在操作函数中实现：

```python
def new_operation(obj, table_bounds, asset_info):
    new_obj = deepcopy(obj)
    # ... 实现操作逻辑 ...

    # 生成指令
    template_zh = random.choice(TEMPLATES["operation_key"]["zh"])
    template_en = random.choice(TEMPLATES["operation_key"]["en"])
    instr_zh = template_zh.format(obj=obj_desc, value=value)
    instr_en = template_en.format(obj=obj_desc, value=value)

    # 元数据
    meta = {
        "operation": "operation_name",
        # ... 其他信息 ...
    }

    return new_obj, instr_zh, instr_en, meta
```

2. 在 `instruction_templates.py` 中添加模板

3. 在 `generate_variant()` 中添加到操作列表：

```python
operation = random.choice([move_object, rotate_object, scale_object, new_operation])
```

### 修改物体选择策略

编辑 `get_unique_category_objects()` 函数：

```python
def get_unique_category_objects(objects):
    # 自定义选择逻辑
    # 例如：只选择特定类别、按大小过滤等
    ...
```

## License

根据项目需要添加相应的许可证信息。
