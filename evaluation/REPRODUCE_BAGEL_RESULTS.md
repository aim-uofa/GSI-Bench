# BAGEL 结果复现说明

本文档说明如何使用仓库中提供的 `bagel_example/` 数据复现 BAGEL 模型在 fine 数据集上的完整评测结果。

> **结论：**
> - **”从已有生成图开始复现最终评测结果”已经可以完整跑通。**
> - **”从原始模型开始重新生成 BAGEL 编辑图”需要额外的 BAGEL 推理代码**，因为 `examples/inference.py` 依赖的 `data/`、`modeling/`、`inferencer.py` 等组件当前仓库内并不完整。

---

## 1. `bagel_example/` 目录说明

我们在仓库中提供了完整的 BAGEL × fine 数据集评测中间结果，用户可以从任意节点开始复现。

```
evaluation/bagel_example/
├── fine_dataset/                  # 评测数据集：211 张原图 + 441 个编辑 JSON（~64MB）
├── generated_images_fine/         # BAGEL 模型生成的 441 张编辑图（~195MB）
├── BAGEL/
│   └── fine_eval/                 # IC / SA / EL 评测结果（含 DetAny3D 推理缓存，~7MB）
├── predictions_infer_2000_Qwen3-VL-235B-A22B-Instruct_fine_BAGEL.json
│                                  # Qwen3-VL AC 打分结果（192KB）
└── agg_results/                   # 最终聚合分数（用于对照验证，24KB）
```

**三种复现路径：**

| 路径 | 起点 | 需要 GPU | 操作 |
|------|------|----------|------|
| **A. 完整复现** | `fine_dataset/` + `generated_images_fine/` | 是 | 跑 IC/SA/EL → 构建 LMDB → MLLM 打分 → 聚合 |
| **B. 跳过评测** | `BAGEL/fine_eval/` + `predictions_*.json` | 否 | 直接聚合 |
| **C. 仅对照** | `agg_results/` | 否 | 查看 JSON 比对数值 |

### 已验证的参考数值

```
IC:       31.973
SA:       22.185
EL.ssim:  28.748
EL.lpips: 27.894
AC:       31.882
Average:  28.484
```

---

## 2. 复现边界

### 2.1 可直接复现的部分（仅需本仓库 + `bagel_example/`）

1. 配环境（第 3 节）
2. 准备数据集与权重（第 4-5 节，或直接用 `bagel_example/fine_dataset/`）
3. 对 `generated_images_*` 跑 IC / SA / EL（第 6 节）
4. 构建 LMDB + MLLM 打分产出 AC JSON（第 9 节）
5. 聚合最终分数（第 7 节）

### 2.2 需要额外代码的部分

1. **从头生成 BAGEL 编辑图**：`examples/inference.py` 依赖 `data.transforms`、`data.data_utils`、`modeling.*`、`inferencer.py`，需要 BAGEL 原始工程代码。
2. **MLLM AC 推理**：需要 vLLM 服务 + 足够的 GPU 资源来部署 Qwen3-VL 等模型。

---

## 3. 从零准备环境

### 3.1 创建 conda 环境

仓库里已经提供了推荐脚本：

```bash
cd ${REPO_ROOT}
bash setup_gsi_env.sh
```

如果你要重建环境：

```bash
cd ${REPO_ROOT}
bash setup_gsi_env.sh --recreate
```

这个脚本会做下面几件事：

- 创建 `gsi` 环境
- 安装 `python=3.10`
- 安装 `torch + torchvision + pytorch-cuda=11.7`
- 安装 `cuda-toolkit=11.7` 与 `cuda-nvcc=11.7`
- 安装 `requirements-cu117.txt`
- 尝试安装本地 `src/groundingdino`

### 3.2 激活环境

```bash
conda activate gsi
```

### 3.3 设置 Python 路径

评测代码会导入 `detect_anything`，而实际包在 `utils/detect_anything` 下，所以建议每次运行前执行：

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

### 3.4 基本自检

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import mmcv; print(mmcv.__version__)"
python -c "import lpips; print('lpips ok')"
```

如果这一步都过不了，后面的评测大概率也跑不通。

---

## 4. 从零准备数据

### 4.1 期望的数据源

仓库默认需要 4 个数据集：

- `fine_dataset`
- `mesatask_dataset`
- `bathroom_dataset`
- `robothor_dataset`

建议把原始 zip 或已解压目录放在同一个目录下，例如：

- `${REPO_ROOT}/GSI-Bench`

### 4.2 建立软链接

仓库已提供脚本：

```bash
cd ${REPO_ROOT}
bash prepare_datasets.sh ${REPO_ROOT}/GSI-Bench
```

如果你要覆盖已有软链接：

```bash
bash prepare_datasets.sh ${REPO_ROOT}/GSI-Bench --force
```

### 4.3 这个脚本实际做了什么

它会：

- 尝试解压 `fine_dataset.zip` / `mesatask_dataset.zip` / `bathroom_dataset.zip` / `robothor_dataset.zip`
- 或直接使用已解压目录
- 在仓库根目录建立：
  - `fine_dataset`
  - `mesatask_dataset`
  - `bathroom_dataset`
  - `robothor_dataset`

### 4.4 快速验证（使用 bagel_example）

如果只想复现 BAGEL × fine 的结果，可以跳过上述步骤，直接使用 `bagel_example/fine_dataset/`。

---

## 5. 从零准备权重

### 5.1 权重来源

评测需要下载以下权重：

- `other_exp_ckpt.pth`
- `sam_vit_h_4b8939.pth`
- `dinov2_vitl14_pretrain.pth`
- `GroundingDINO_SwinB_cfg.py`
- `groundingdino_swinb_cogcoor.pth`

### 5.2 建立权重软链接

仓库已提供脚本：

```bash
cd ${REPO_ROOT}
bash prepare_weights.sh ${REPO_ROOT}/GSI-weight
```

如需覆盖已有链接：

```bash
bash prepare_weights.sh ${REPO_ROOT}/GSI-weight --force
```

### 5.3 验证权重

```bash
ls checkpoints/detany3d/detany3d_ckpts/other_exp_ckpt.pth
ls checkpoints/detany3d/sam_ckpts/sam_vit_h_4b8939.pth
ls checkpoints/detany3d/dino_ckpts/dinov2_vitl14_pretrain.pth
```

---

## 6. 跑 IC / SA / EL 评测（路径 A）

使用 `bagel_example/` 中的数据和生成图重新跑评测。**需要 GPU 和评测权重。**

```bash
cd evaluation
export PYTHONPATH=$PWD:$PYTHONPATH
conda activate gsi

# 用 bagel_example 中的数据跑评测
bash eval_one.sh \
  ./bagel_example/BAGEL \
  ./bagel_example/fine_dataset \
  instruction-compliance,spatial-accuracy,edit-locality
```

结果写入 `bagel_example/BAGEL/fine_eval/`：

- `instruction-compliance_eval_results.json`
- `spatial-accuracy_eval_results.json`
- `edit-locality_eval_results.json`
- 对应的 `*_eval_stats.json`

> **批量评测**：如果你有多个模型，将生成图放到 `eval/<model_name>/generated_images_fine/`，然后运行 `bash eval.sh`。

---

## 7. 聚合最终分数（路径 A / B）

### 7.1 使用 bagel_example 聚合

```bash
cd evaluation
export PYTHONPATH=$PWD:$PYTHONPATH

python -m eval.aggregate \
  --root-dir ./bagel_example \
  --output-dir ./bagel_example/my_agg_results \
  --mllm-eval-dir ./bagel_example
```

### 7.2 参数说明

- `--root-dir`：模型目录的上一层。脚本会遍历其下每个子目录作为模型名，并在其中查找 `*_eval/` 目录。
- `--mllm-eval-dir`：AC 预测 JSON 所在目录。脚本会用 glob `*{dataset}_{model}.json` 匹配。

### 7.3 输出

`my_agg_results/` 下会生成：

- `EVAL_output_summary.json`（所有指标汇总）
- `EVAL_output_ac.json` / `EVAL_output_ic.json` / `EVAL_output_sa.json` / `EVAL_output_edit_locality.json`
- `EVAL_output_average.json`

### 7.4 对照验证

将输出与 `bagel_example/agg_results/` 中的参考值比对，应完全一致：

```
IC:       31.973
SA:       22.185
EL.ssim:  28.748
EL.lpips: 27.894
AC:       31.882
Average:  28.484
```

---

## 8. 如果你想“从生成图片开始”完整复现

这里要特别区分两种情况。

### 8.1 用**已有**生成图复现评测

这条链路已经自洽：

1. `generated_images_fine/`
2. `bash eval_one.sh ...`
3. `python -m eval.aggregate ...`

### 8.2 用仓库里的 `examples/inference.py` 重新生成 BAGEL 图片

**当前不能保证直接跑通。**

虽然仓库提供了：

- `examples/inference.py`
- `examples/mydataset.py`

但 `examples/inference.py` 还依赖这些当前仓库中缺失或不完整的模块：

- `data.transforms`
- `data.data_utils`
- `modeling.bagel`
- `modeling.qwen2`
- `modeling.autoencoder`
- `inferencer.py`

因此，若你要真正“从模型推理开始”复现 `generated_images_fine/`，还需要：

1. BAGEL 原始工程代码
2. 可用的 BAGEL 模型目录
3. 这些模块对应的完整依赖与推理脚本

换句话说，**本仓库现在更像是“评测仓库 + 一个不完整的 BAGEL 推理示例”**，不是完整的 BAGEL 生成工程。

---

## 9. AC 打分复现（可选）

AC（Acceptance Consistency）通过 MLLM（如 Qwen3-VL）对原图与编辑图进行打分。完整流程分三步：

### 9.1 构建 LMDB 数据集

使用 `mllm_eval/build_lmdb.py` 将原图、编辑图、prompt 打包为 LMDB：

```bash
cd evaluation/mllm_eval

python build_lmdb.py \
  --dataset-dir ../bagel_example/fine_dataset \
  --generated-dir ../bagel_example/generated_images_fine \
  --output-dir ./EVAL_lmdb_dataset/fine_BAGEL \
  --dataset-name fine \
  --model-name BAGEL
```

依赖：`pip install lmdb opencv-python`

### 9.2 启动 vLLM 并推理

```bash
bash eval_infer.sh <path_to_qwen3_vl_model> default 8000
```

这会启动 vLLM 服务，运行 `mllm_eval.py` 对 LMDB 中的图片对逐一打分，输出 `predictions_*.json`。

需要：
- vLLM（`pip install vllm`）
- Qwen-VL 依赖（`pip install qwen-vl-utils`）
- 足够的 GPU 资源部署 Qwen3-VL 或类似 MLLM

### 9.3 跳过 AC 推理

如果不想自行搭建 vLLM，可以直接使用 `bagel_example/` 中已有的 AC JSON：

```
bagel_example/predictions_infer_2000_Qwen3-VL-235B-A22B-Instruct_fine_BAGEL.json
```

将其传给 `eval.aggregate` 的 `--mllm-eval-dir` 即可。

---

## 10. 快速复现（最短路径，无需 GPU）

如果只想验证聚合分数是否一致，无需 GPU，3 条命令即可：

```bash
cd evaluation
export PYTHONPATH=$PWD:$PYTHONPATH

# 聚合 bagel_example 中的已有评测结果
python -m eval.aggregate \
  --root-dir ./bagel_example \
  --output-dir ./bagel_example/my_agg_results \
  --mllm-eval-dir ./bagel_example

# 查看结果
cat ./bagel_example/my_agg_results/EVAL_output_summary.json

# 对比参考值
cat ./bagel_example/agg_results/EVAL_output_summary.json
```

两份 JSON 的数值应完全一致。

---

## 11. 常见坑

### 11.1 `detect_anything` 无法导入

解决：

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

### 11.2 聚合命令找不到模型

`--root-dir` 要指到"模型目录的上一层"。例如你的评测结果在 `./bagel_example/BAGEL/fine_eval/`，则 `--root-dir` 应该传 `./bagel_example`（而不是 `./bagel_example/BAGEL`）。

### 11.3 想从 `examples/inference.py` 直接生成图但报模块缺失

这不是你的环境问题，而是当前仓库里确实没有提供完整 BAGEL 推理依赖。

### 11.4 想直接用 `mllm_eval/eval_infer.sh` 生成 AC

该脚本可以正常启动 vLLM 并运行推理，但需要你提前准备好 `./EVAL_lmdb_dataset` 数据目录。如果你只需要最终分数，直接用已有的 AC JSON 配合 `python -m eval.aggregate` 即可。

---

## 12. 总结

`bagel_example/` 提供了完整的中间结果，可以从任意节点复现 BAGEL × fine 的评测分数：

- **无需 GPU**：直接用已有的 `fine_eval/` + `predictions_*.json` 跑聚合（路径 B）
- **有 GPU**：用 `fine_dataset/` + `generated_images_fine/` 重新跑 IC/SA/EL（路径 A）
- **有 GPU + vLLM**：还可以重新跑 AC 打分（第 9 节）

唯一不能在本仓库内完成的是”从零生成 BAGEL 编辑图”——这需要 BAGEL 原始推理代码（`data.*`、`modeling.*`、`inferencer.py`），可参考 [BAGEL 项目](https://github.com/ByteDance-Seed/BAGEL)。

