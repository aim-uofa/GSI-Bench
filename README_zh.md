# GSI-Bench：从生成式视角探索空间智能

Language: 中文 | English in [`README.md`](README.md)

**🎉 CVPR 2026 录用**

论文官方实现：

> **Exploring Spatial Intelligence from a Generative Perspective**
> _CVPR 2026_
> [[论文]](paper/main.pdf) [[arXiv]](https://arxiv.org/abs/2604.20570) [[项目主页]](https://aim-uofa.github.io/GSI-Bench/)

GSI-Bench 用于评估生成式模型在室内场景中理解与操纵 3D 空间关系的能力。

| 指标 | 全称 | 衡量内容 |
|------|------|---------|
| **IC** | Instruction Compliance（指令遵循） | 输出是否真正执行了指令要求的空间操作？ |
| **SA** | Spatial Accuracy（空间精度） | 3D 位移、旋转或缩放是否贴近 ground-truth 几何？ |
| **AC** | Appearance Consistency（外观一致性） | 编辑后物体身份、类别与外观是否保持？ |
| **EL** | Edit Locality（编辑局部性） | 目标区域之外的场景是否保持不变？ |

---

## 快速导航

> **如果只想在 GSI-Bench 上评测你的模型，直接跳到 [评测](#评测)。**
>
> 第 1、2 部分是数据构造流程，为了透明和可复现而开源，**不是运行评测所必需**。

```
GSI-Bench/
├── evaluation/     # 评测框架（IC / SA / EL / AC）← 从这里开始
├── robothor/       # [可选] 数据生成流水线 1：RoboTHOR 室内场景
├── mesatask/       # [可选] 数据生成流水线 2：MesaTask 桌面场景
├── paper/          # 论文 PDF
└── tests/          # 单元/集成测试
```

---

## 评测

**可复现范围：**

| 目标 | 需要 | 文档 |
|------|------|------|
| 对**你自己的**编辑图跑 IC/SA/EL/AC | 评测数据集 + 权重 + `eval/<模型>/` 目录结构 | 本节 |
| 复现论文 **BAGEL × fine** 分数 | 单独下载 [`bagel_example/`](evaluation/bagel_example/README.md)（约 265MB，不在 Git 中） | [`REPRODUCE_BAGEL_RESULTS.md`](evaluation/REPRODUCE_BAGEL_RESULTS.md) |
| 重新生成 BAGEL 编辑图 | 外部 [BAGEL](https://github.com/ByteDance-Seed/BAGEL) 工程（本仓未包含完整推理栈） | 同上文档第 8 节 |

### 1. 环境准备

```bash
conda create -n gsi-eval python=3.10 -y
conda activate gsi-eval

cd evaluation

# 根据你的 CUDA 版本安装 PyTorch（示例：CUDA 11.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装带 C++ ops 的 mmcv
pip install -U openmim && mim install mmcv

# 安装剩余依赖
pip install -r requirements.txt

# 可选：构建 GroundingDINO（用于文本 prompt 检测）
pip install -e ./src/groundingdino --no-build-isolation
```

### 2. 下载模型权重

| 权重 | 大小 | 来源 |
|------|------|------|
| `other_exp_ckpt.pth`（DetAny3D） | ~500MB | [OpenDriveLab/DetAny3D](https://github.com/OpenDriveLab/DetAny3D) |
| `sam_vit_h_4b8939.pth`（SAM ViT-H） | ~2.4GB | [Meta AI](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| `dinov2_vitl14_pretrain.pth`（DINOv2） | ~1.1GB | [Meta AI](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) |
| `groundingdino_swinb_cogcoor.pth`（可选） | ~690MB | [IDEA-Research](https://github.com/IDEA-Research/GroundingDINO) |

把全部权重放到同一目录，再执行：
```bash
bash prepare_weights.sh <权重目录路径>
# 会在 checkpoints/ 与 GroundingDINO/weights/ 下创建软链接
```

### 3. 下载评测数据集

从 [项目主页](https://aim-uofa.github.io/GSI-Bench/) 获取四个评测数据集的压缩包，然后：

```bash
cd evaluation
bash prepare_datasets.sh <下载好的数据集目录>
# 会创建软链接：fine_dataset/  mesatask_dataset/  bathroom_dataset/  robothor_dataset/
```

### 4. 用你的模型生成编辑图像

你的模型需要按以下命名约定产出编辑图像：
```
eval/<model_name>/generated_images_fine/<img_id>_edit_<query_id>.png
eval/<model_name>/generated_images_mesatask/<img_id>_edit_<query_id>.png
eval/<model_name>/generated_images_bathroom/<img_id>_edit_<query_id>.png
eval/<model_name>/generated_images_robothor/<img_id>_edit_<query_id>.png
```

`examples/inference.py` 仅为 BAGEL 输出格式的**示例骨架**；完整出图需 [BAGEL](https://github.com/ByteDance-Seed/BAGEL) 原仓库。若只需复现论文 BAGEL 分数、不重新生成图，请下载 [`bagel_example/`](evaluation/bagel_example/README.md) 并阅读 [`evaluation/REPRODUCE_BAGEL_RESULTS.md`](evaluation/REPRODUCE_BAGEL_RESULTS.md)。

### 5. 运行评测

```bash
cd evaluation
export PYTHONPATH=$PWD:$PYTHONPATH

# IC / SA / EL 评测（遍历所有模型 × 所有数据集）
bash eval.sh

# （可选）基于 MLLM 的 AC 打分——需 vLLM + Qwen3-VL（见 evaluation/requirements-mllm.txt）
cd mllm_eval
pip install -r ../requirements-mllm.txt   # 另按 CUDA 环境安装 vllm
bash eval_infer.sh <qwen3_vl_模型路径> default <port>
# 结果写入 mllm_eval/infer_results/
cd ..

# 聚合所有指标生成最终报告
python -m eval.aggregate \
  --root-dir ./eval \
  --output-dir ./eval_results \
  --mllm-eval-dir ./mllm_eval/infer_results

cd ..   # 回到仓库根目录
```

**输出：** `eval_results/` 内按模型/数据集组织的 JSON 文件，包含 IC/SA/EL/AC 分数。

输入格式细节与疑难排查见 [`evaluation/eval/README.md`](evaluation/eval/README.md)。

---

## 数据生成流水线（可选）

> 以下两条流水线说明了我们是如何构造 GSI-Bench 数据的。它们**不是运行评测所必需**——评测数据集可以直接下载。

### 流水线 1：RoboTHOR 室内场景

**环境：**
```bash
conda create -n gsi-robothor python=3.10 -y
conda activate gsi-robothor
pip install -r robothor/requirements.txt
# 依赖：ai2thor>=5.0.0, numpy, Pillow, matplotlib
# AI2-THOR 会在首次运行时自动下载场景资源（~2GB）
# 需要：NVIDIA GPU + CloudRendering（headless）或 X server（图形界面）
```

**生成数据：**
```bash
cd robothor

# 1) 为全部 60 个训练场景生成基础视角 + 相对相机的命令
#    输出：data/outputs/train/with_physics/
bash scripts/generate_train.sh

# 2) 生成其他命令类型（需要第 1 步生成的视角）
bash scripts/generate_train_object.sh          # 相对物体位置
bash scripts/generate_train_rotate.sh           # 旋转命令
bash scripts/generate_train_receptacle.sh       # 容器放置
bash scripts/generate_train_spatial_remove.sh    # 空间删除
bash scripts/generate_train_agent_camera.sh      # agent 相机移动

# 3) 生成验证集
bash scripts/generate_val_agent_camera.sh

cd ..   # 回到仓库根目录
```

**输出：** `data/outputs/{train,val}/`，每个视角每个命令对应一条 JSONL 记录 + RGB/深度/分割图像。

**耗时：** 每个场景 ~2–5 分钟（视 GPU 而定）。60 个场景全跑：数小时。

详见 [`robothor/README.md`](robothor/README.md)。

---

### 流水线 2：MesaTask 桌面场景

**环境：**
```bash
conda create -n gsi-mesatask python=3.10 -y
conda activate gsi-mesatask
pip install -r mesatask/requirement.txt
# 推理（可选）：pip install torch torchvision
# 渲染（可选）：从 https://www.blender.org/download/ 下载 Blender 4.3+
# 物理优化（可选）：conda install -c conda-forge drake
```

**下载 MesaTask-10K 数据集：**
```bash
cd mesatask
git lfs install
git clone https://huggingface.co/datasets/InternRobotics/MesaTask-10K MesaTask-10K

# 准备资源库（从数据集压缩包解压）
cd MesaTask-10K/Assets_library_archive
cat Assets_library_backup.tar.gz.* > Assets_library_merged.tar.gz
tar -xzvf Assets_library_merged.tar.gz -C ../Assets_library/
cd ../..
```

**生成数据：**
```bash
cd mesatask

# 1) 生成原子变换（移动 / 旋转 / 缩放）
python generate_atomic_transforms.py \
  --input-dir MesaTask-10K/Layout_info \
  --asset-annotation MesaTask-10K/Asset_annotation.json \
  --output-dir transformed_layouts \
  --num-variants 10 --seed 42

# 2) 渲染所有 layout（需要 Blender）
python dataset/vis_batch.py transformed_layouts \
  --output_dir dataset/vis_final --parallel 4

# 3) 组装图像编辑数据集
python organize_image_editing_dataset.py \
  --transformed-dir transformed_layouts \
  --vis-dir dataset/vis_final \
  --output-dir dataset/image_editing_dataset

cd ..   # 回到仓库根目录
```

**耗时：** 第 1 步约 10 分钟（1 万个场景）。第 2 步（渲染）取决于机器与并行度。

详见 [`mesatask/README.md`](mesatask/README.md)。

---

## 验证仓库

```bash
git clone <本仓库地址> GSI-Bench && cd GSI-Bench

# 跑测试（无需 GPU 与数据）
pip install pytest
python -m pytest tests/ -v    # 应当 43 条全部通过
```

## 环境需求总览

| 组件 | Python | GPU | Conda Env |
|------|--------|-----|-----------|
| **tests/** | 3.8+ | 不需要 | 任意 |
| **evaluation/** | 3.10 | NVIDIA（DetAny3D） | `gsi-eval` |
| **robothor/** | 3.10 | NVIDIA（CloudRendering） | `gsi-robothor` |
| **mesatask/** | 3.10 | 可选 | `gsi-mesatask` |

---

## 引用

```bibtex
@article{zhu2026exploring,
  title={Exploring Spatial Intelligence from a Generative Perspective},
  author={Zhu, Muzhi and Jiang, Shunyao and Zheng, Huanyi and Luo, Zekai and Zhong, Hao and Li, Anzhou and Wang, Kaijun and Rong, Jintao and Liu, Yang and Chen, Hao and Lin, Tao and Shen, Chunhua},
  journal={arXiv preprint arXiv:2604.20570},
  year={2026}
}
```

## 许可证

GSI-Bench 采用 MIT 许可证——详见 [`LICENSE`](LICENSE)。

包含第三方项目源码的子目录遵循各自的许可证：

- [`robothor/LICENSE`](robothor/LICENSE)
- [`mesatask/LICENSE`](mesatask/LICENSE)
