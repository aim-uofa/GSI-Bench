# Eval 使用说明

Language: 中文 | English in [`README.en.md`](README.en.md)

本说明针对 `eval.sh` 和 `python -m eval.eval` 的评估流程，包含输入格式、命名规范、权重与环境依赖等要点。

## 入口

- 批量评估（多模型、多数据集）：`bash eval.sh`
- 单次评估（自定义目录）：`python -m eval.eval --original <dir> --edited <dir> --output <dir> --mode <mode> --dataset <name>`

## 目录结构（eval.sh 约定）

`eval.sh` 会遍历 `./eval/*` 下的每个子目录作为一个模型输出目录，并对四个数据集评估：

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

评估输出写到：

```
eval/<model_name>/
  fine_eval/
  mesatask_eval/
  bathroom_eval/
  robothor_eval/
```

## 输入与命名规范（非常关键）

评估脚本通过**编辑图像文件名**反推 JSON 与原图路径，命名必须匹配：

### 1) 编辑图像（`--edited` 目录）

**必须包含 `_edit_`**：

```
<img_id>_edit_<query_id>.jpg|png
```

脚本会先做 `img_file.replace("_rgb","")`，所以你可以有：

```
<img_id>_edit_<query_id>_rgb.jpg
```

但最终会按去掉 `_rgb` 后的名字解析。

### 2) 原始图像（`--original` 目录）

在 `--original` 目录中，**文件名前缀等于 `<img_id>`** 的第一张图会被当作原图：

```
<img_id>*.jpg|png
```

### 3) 编辑 JSON（`--original` 目录）

**JSON 文件名必须与编辑图像同名（仅扩展名改为 `.json`）**，且放在 `--original` 目录：

```
<img_id>_edit_<query_id>.json
```

> 注意：`eval.py` 里 `--edit` 参数目前**未被使用**，JSON 只按以上规则寻找。

### 4) 视角变换（view）可选 GT 图

如果 prompt 里包含 `camera` 且不包含 `relative to the camera`，会判定为 `view` 操作，并尝试加载 GT 编辑图：

```
<img_id>_*gtedit_<query_id>*.*   （位于 --original 目录）
```

若找不到，view 评估会降级为失败。

## 编辑 JSON 格式（最小字段）

评估用到的字段如下（来自 `gen_data/generate.py` 生成逻辑）：

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

字段要求：

- **必须**：`prompt`, `target`, `original_bbox_3d`, `camera_intrinsics`, `rotation_matrix`
- **move/scale**：需要 `new_bbox_3d`
- **rotate**：需要 `new_rotation_matrix`
- **remove**：只用 `original_bbox_3d`
- **view**：使用 GT edited 图，不强依赖 bbox，但 locality 仍会用到 `original_bbox_3d` 等

## 权重与配置

评估时会跑 DetAny3D 推理，默认配置为：

```
--config ./detect_anything/configs/demo.yaml
```

但本仓库配置实际在：

```
utils/detect_anything/configs/demo.yaml
```

因此建议显式指定：

```
python -m eval.eval --config utils/detect_anything/configs/demo.yaml ...
```

`demo.yaml` 中要求的权重路径（需自行准备）：

```
./checkpoints/detany3d/detany3d_ckpts/other_exp_ckpt.pth
./checkpoints/detany3d/sam_ckpts/sam_vit_h_4b8939.pth
./checkpoints/detany3d/dino_ckpts/dinov2_vitl14_pretrain.pth
```

可选（若使用文本检测）：

```
GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py
GroundingDINO/weights/groundingdino_swinb_cogcoor.pth
```

## 环境依赖（建议）

评估端用到的主要依赖（最小集合）：

- Python 3.8+
- `torch`, `torchvision`
- `mmcv`（必须包含 `mmcv.ops.multi_scale_deform_attn`）
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

可选：

- `scipy`（旋转相关）
- GroundingDINO 及其依赖（若要启用文本检测）
- `pycocotools`（部分工具函数会用到）
- `open3d`（数据集测试脚本会用到）

> 另外：代码里使用 `from utils.train_utils ...` 等导入，需要将 evaluation 根目录加入 PYTHONPATH：
> `export PYTHONPATH=$PWD:$PYTHONPATH`（在 evaluation/ 目录下运行）。
> **注意**：不要设 `PYTHONPATH=$PWD/utils`，这会导致命名空间冲突。

### pip 安装示例

以下为**示例**安装流程（需根据你的 CUDA/驱动版本选择合适的 PyTorch 与 mmcv 版本）：

```bash
# 1) PyTorch（根据你的 CUDA 版本替换下行命令）
# 例：CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2) mmcv（需包含 ops；推荐用 openmim 自动匹配）
pip install -U openmim
mim install mmcv

# 3) 其它依赖
pip install numpy opencv-python pillow pyyaml python-box scikit-image lpips tqdm \
            timm einops shapely matplotlib six termcolor

# 4) xformers（与 torch/cuda 匹配；缺失会在导入时报错）
pip install xformers

# 5) 可选
pip install scipy pycocotools
```

## 输出

在 `--output` 目录生成：

- `infer_cache.json`（DetAny3D 推理缓存）
- `instruction-compliance_eval_results.json` / `spatial-accuracy_eval_results.json`
- `*_eval_stats.json`（汇总统计）

## 常见问题

- **`--edit` 参数无效**：当前代码未使用该参数，JSON 必须放在 `--original` 目录且命名匹配。
- **找不到 config**：请用 `--config utils/detect_anything/configs/demo.yaml`。
- **找不到 detect_anything 包**：使用 `PYTHONPATH=./utils` 或软链接。
