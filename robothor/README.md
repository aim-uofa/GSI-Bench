# RoboTHOR 数据生成流水线（GSI-Bench-robothor-GEN）

本项目保留了训练/验证集生成所需的核心脚本与工具模块。目标是让常用的数据生成脚本在一个干净的仓库结构中即可直接运行，并便于后续开源维护。

## 目录结构
- `action_utils/`: 生成命令与执行动作的核心工具集（camera/object/rotate/receptacle/spatial_remove 等）。
- `test_robothor_simple_cluster_move.py`: 主入口脚本（场景遍历、视角采样、命令生成与执行）。
- `scripts/`: 数据生成的可执行脚本（训练/验证、不同命令类型）。
- `data/pregenerated_views/`: 预生成视角数据目录（支持软链接到已有数据）。
- `data/outputs/`: 生成结果输出目录（按 train/val 分类）。

## 快速开始
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 运行基础视角生成（训练集）：
   ```bash
   bash scripts/generate_train.sh
   ```
3. 在已有视角基础上生成不同指令数据：
   ```bash
   bash scripts/generate_train_object.sh
   bash scripts/generate_train_rotate.sh
   ```

## 预生成视角说明
部分脚本需要 `--pregenerated-views` 指定已有的 `selected_views.json`。请自行准备预生成视角数据，或通过运行基础视角生成脚本（如 `scripts/generate_train.sh`）重新生成。生成后的视角目录（例如 `data/outputs/train/with_physics`）可直接用于该参数。

## 输出与复现
所有结果默认输出到 `data/outputs/`。脚本中可以通过设置 `CUDA_VISIBLE_DEVICES` 控制使用的 GPU。

如需扩展参数（分辨率、视角采样、是否禁用物理等），请查看 `test_robothor_simple_cluster_move.py` 中的命令行参数与默认配置。
