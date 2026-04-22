# Drake 场景优化器

Language: 中文 | English in [`Readme.md`](Readme.md)

这是一个基于 [Drake](https://drake.mit.edu/) 物理引擎与 [Steerable Scene Generation](https://github.com/nepfaff/steerable-scene-generation) 项目构建的 Python 脚本，用于优化 3D 场景。它的主要功能是读取一个描述多个物体位置与尺寸的场景文件，在遵循预定义运动约束的前提下，自动调整物体位置以消除物理穿模（碰撞）。


## 配置文件

运行脚本前，请先修改 `config.yaml`，配置好需要的路径与优化参数。


## 使用方法

### 1. 准备输入文件

脚本需要一个包含以下两个文件的输入目录：

1.  **`scene_processed_scene.json`**：
    -   描述场景中每个物体的初始状态。

2.  **`scene_layout.txt`**：
    -   其中 "Scene Graph" 段落定义了物体之间的空间关系。

### 2. 运行优化脚本

在命令行运行 `drake_process.py`，并指定输入目录的路径：

```bash
python drake_process.py --floder_path /path/to/your/scene_folder
```

也可选择性地指定自定义配置文件路径：

```bash
python drake_process.py --floder_path /path/to/your/scene_folder --config /path/to/your/custom_config.yaml
```

### 3. 查看输出

脚本执行成功后，会在 `--floder_path` 指定的目录下生成一个新的 JSON 文件。文件名基于原始文件名加上配置里的 `output_suffix`，默认是 **`scene_processed_scene_optimized.json`**。

该文件包含所有物体新的、经物理优化后的位置，可直接用于后续的渲染或仿真流程。
