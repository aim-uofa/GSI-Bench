# Generative-Spatial-Intelligence

Language: 中文 | English in [`README.md`](README.md)

本项目提供一条用于生成式空间智能研究的评测流水线，聚焦于室内场景中的 3D 物体检测、场景编辑与结果评估。支持对物体进行移动、旋转、删除等操作，并自动生成编辑任务与评估结果。

## 项目结构

- `dataloader.py`：数据加载与预处理
- `gen_modify.py`：编辑任务与修改指令生成
- `get_img.py`：图像抽取与拷贝
- `model.py`：3D 检测与模型封装
- `edit_visualize.py`：编辑结果可视化
- `eval.py`：编辑效果评估
- `run.sh`：一键运行流水线的脚本

## 环境依赖

- Python 3.8+
- PyTorch
- OpenCV
- numpy
- Pillow
- shapely
- box
- yaml
- 其他依赖请查看每个脚本

## 快速开始

1. 安装依赖
2. 运行主流水线：
```
bash run.sh <source_dir> <dest_dir> <output_dir> <cuda_device>
```
>
参数说明：

- <source_dir>：原始数据目录
- <dest_dir>：编辑任务与中间结果目录
- <output_dir>：最终输出目录
- <cuda_device>：CUDA 设备编号


## 主要功能
- 自动生成物体编辑任务（移动、旋转、删除）
- 3D 物体检测与空间推理
- 编辑结果（含 ground truth）可视化
- 编辑效果自动评估

## 环境配置（推荐）

本项目需要可用的 CUDA GPU 环境。推理代码默认使用 `cuda:0`。

### CUDA 11.7（conda 方案，推荐）

```bash
# 0) 移除旧环境
conda deactivate
conda env remove -n gsi -y

# 1) 创建干净环境
conda create -n gsi python=3.10 pip -y
conda activate gsi

# 2) 安装 PyTorch + CUDA 11.7
conda install -c pytorch -c nvidia pytorch torchvision pytorch-cuda=11.7 -y

# 3) 安装 CUDA 工具链（头文件 + nvcc）
conda install -c nvidia cuda-toolkit=11.7 cuda-nvcc=11.7 -y

# 4) 设置 CUDA 路径（建议加入 activate.d）
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CPATH="$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH"

# 5) 自检（必须通过）
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
nvcc --version
find "$CONDA_PREFIX" -name cuda_runtime_api.h | head
find "$CONDA_PREFIX" -path "*/thrust/complex.h" | head

# 6) 安装其他依赖
pip install -U pip ninja
pip install -r requirements-cu117.txt

# 7) 构建 GroundingDINO（可选；用于文本 prompt 检测）
pip install -e ./src/groundingdino --no-build-isolation
```

### PYTHONPATH 设置（必做）

代码中使用了 `from utils.train_utils import ...` 和 `from utils.wrap_model import ...`，因此评测根目录必须加入 `PYTHONPATH`。请在 `evaluation/` 目录下执行：

```bash
cd evaluation
export PYTHONPATH=$PWD:$PYTHONPATH
```

> **注意：** 不要设 `PYTHONPATH=$PWD/utils`——这会覆盖 `utils` 命名空间包并导致导入错误。

### 可选：GroundingDINO（文本 prompt 检测）

只有需要文本 prompt 检测时才需要 GroundingDINO：

```bash
pip install -e git+https://github.com/IDEA-Research/GroundingDINO.git@856dde20aee659246248e20734ef9ba5214f5e44#egg=groundingdino
```

配置和权重路径需与代码默认值一致：

```
<repo_root>/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py
<repo_root>/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth
```

## 必需的模型权重（默认配置）

默认配置是 `utils/detect_anything/configs/demo.yaml`，需要：

```
./checkpoints/detany3d/detany3d_ckpts/other_exp_ckpt.pth
./checkpoints/detany3d/sam_ckpts/sam_vit_h_4b8939.pth
./checkpoints/detany3d/dino_ckpts/dinov2_vitl14_pretrain.pth
```

请下载这些权重并放到上述路径。

## 评测

本节介绍如何跨四项指标评测编辑图像：

- IC（指令遵循）
- SA（空间精度）
- EL（编辑局部性）
- AC（基于 MLLM 的接受一致性）

按下列步骤组织输出、运行指标脚本、聚合结果。

1) 准备目录结构

把生成图像放到 `eval/<model_name>/` 下，每个数据集一个子目录：

```
eval/
  <model_name>/
    generated_images_fine/
    generated_images_mesatask/
    generated_images_bathroom/
    generated_images_robothor/
```

数据集要放在仓库根目录（`eval.sh` 所期望的命名）：

```
fine_dataset/   mesatask_dataset/   bathroom_dataset/   robothor_dataset/
```

2) 生成编辑图像

我们在 `examples/` 下提供了一个 BAGEL 示例：

```
python examples/inference.py
```

想接入其他模型或数据格式，可修改 `examples/mydataset.py`。生成图像命名必须遵循 `<img_id>_edit_<query_id>.(png|jpg)`，评估器据此定位对应的 JSON 与原图（详见 `eval/README.md`）。

3) 运行 IC/SA/EL 评测

使用仓库提供的批处理脚本（遍历 `eval/` 下所有模型与所有数据集）：

```
bash eval.sh
```

输出写入 `eval/<model_name>/*_eval/`，每个指标一个 JSON（如 `instruction-compliance_eval_stats.json`）。

4) 运行基于 MLLM 的 AC 打分（可选但推荐）

启动 `mllm_eval/` 下的评测辅助脚本，用一个在线 LLM 服务生成 AC 结果：

```
cd mllm_eval
bash eval_infer.sh <model_path> default <port>
```

这会生成预测 JSON（如 `predictions_infer_1000_<MODEL_NAME>.json`）。可以把这些文件放在同一目录，下一步通过 `--mllm-eval-dir` 传入。

5) 聚合所有指标（IC/SA/EL/AC）

运行聚合器，收集各模型、各数据集的平均值与简单综合得分：

```
python -m eval.aggregate \
  --root-dir ./eval \
  --output-dir <output_dir> \
  --mllm-eval-dir <含 MLLM AC 预测 json 的目录>
```

最终结果为输出目录中的 JSON 文件。
