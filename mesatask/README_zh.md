# MesaTask：面向任务的桌面场景生成（3D 空间推理）

语言：中文 | 英文见 `README.md`

MesaTask 是一个面向任务的桌面场景生成研究仓库，提供：
- MesaTask-10K 数据集布局文件与资产标注
- 从自然语言任务生成 3D 场景的推理流程
- Blender 渲染与可视化工具
- 原子变换生成与图像编辑数据集整理工具

原子变换生成的细节请参考 `README_transforms.md`。

## 仓库内容概览

- `MesaTask-10K/`：数据集根目录（布局、标注、资产、模型）
- `generate_atomic_transforms.py`：原子变换生成脚本
- `instruction_templates.py`：中英文指令模板
- `dataset/vis_single.py`、`dataset/vis_batch.py`：渲染与可视化
- `organize_image_editing_dataset.py`：图像编辑数据集整理（包装脚本）
- `oraganize_image_editing_dataset.py`：旧拼写脚本，仍可用
- `get_task_info.py`、`inference.py`：推理流程
- `config.yaml`：渲染与资产路径配置

## 环境配置

建议使用 Python 3.10。

1) 创建并激活环境
```bash
conda create -n MesaTask python=3.10
conda activate MesaTask
```

2) 安装依赖
```bash
# 核心依赖
pip install -r requirement.txt

# 推理相关（可选但推荐）
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

3) 安装 Blender（用于渲染）
```bash
wget https://download.blender.org/release/Blender4.3/blender-4.3.2-linux-x64.tar.xz
tar -xvJf blender-4.3.2-linux-x64.tar.xz
```

修改 `config.yaml` 中 Blender 路径，例如：
```yaml
blender_executable: "./blender-4.3.2-linux-x64/blender"
```

如果只需要数据生成而不渲染，可不安装 Blender。

## 数据准备

1) 从 Hugging Face 下载 MesaTask-10K。
2) 将数据集放在仓库根目录 `./MesaTask-10K/`。
3) 解压资产库：
```bash
cd MesaTask-10K
mkdir -p Assets_library

cd Assets_library_archive
cat Assets_library_backup.tar.gz.* > Assets_library_merged.tar.gz

tar -xzvf Assets_library_merged.tar.gz -C ../Assets_library/

cd ..
rm -r Assets_library_archive
```

期望目录结构：
```text
MesaTask-10K/
|-- MesaTask_model
|-- Asset_annotation.json
|-- sbert_text_features.pkl
|-- Assets_library/
|-- Layout_info/
|-- ...
```

## 可复现流程（原子变换 + 渲染 + 图像编辑数据）

以下命令均在仓库根目录执行：

1) 生成原子变换（固定 seed 可复现）
```bash
python generate_atomic_transforms.py \
  --input-dir MesaTask-10K/Layout_info \
  --asset-annotation MesaTask-10K/Asset_annotation.json \
  --output-dir transformed_layouts5 \
  --num-variants 10 \
  --seed 123
```

2) 渲染生成的布局（需要 Blender）
```bash
python dataset/vis_batch.py transformed_layouts5 \
  --output_dir dataset/vis_final \
  --parallel 4
```

3) 整理图像编辑数据集
```bash
python organize_image_editing_dataset.py \
  --transformed-dir transformed_layouts5 \
  --vis-dir dataset/vis_final \
  --output-dir dataset/image_editing_dataset
```

可复现性建议：
- 固定 `--seed`
- 保持相同的数据集版本、Python 版本与 Blender 版本
- 不要修改中间生成的 layout JSON

## 可视化（可选）

单个布局：
```bash
python dataset/vis_single.py MesaTask-10K/Layout_info/dining_table/dining_table_0000/layout.json \
  --output_dir dataset/vis_data
```

批量渲染：
```bash
python dataset/vis_batch.py MesaTask-10K/Layout_info \
  --output_dir dataset/vis_data \
  --parallel 4
```

更多说明见 `dataset/README_visualization.md`。

## 推理流程

1) 生成任务信息
```bash
python get_task_info.py \
  --task_name "Organize books and magazines on the table" \
  --table_type "Nightstand" \
  --api_key "your_api_key" \
  --model "gpt-4o" \
  --output_dir "output"
```

2) 生成并渲染场景
```bash
python inference.py \
  --input_file output/task_001/task_info.json \
  --mesatask_model_path ./MesaTask-10K/MesaTask_model \
  --rendering
```

可选：物理优化
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

## 引用

```text
@misc{hao2025mesatask,
  title={MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning},
  author={Hao, Jinkun and Liang, Naifu and Luo, Zhen and Xu, Xudong and Zhong, Weipeng and Yi, Ran and Jin, Yichen and Lyu, Zhaoyang and Zheng, Feng and Ma, Lizhuang and Pang, Jiangmiao},
  journal={arXiv preprint arXiv:2509.22281},
  year={2025}
}
```

## 许可证

Apache License。
