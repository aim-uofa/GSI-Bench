# Drake Scene Optimizer

This is a Python script built on the [Drake](https://drake.mit.edu/) physics engine and project [Steerable Scene Generation](https://github.com/nepfaff/steerable-scene-generation) for optimizing 3D scenes. The primary function of this script is to read a scene file describing the positions and sizes of multiple objects, and automatically adjust their positions to eliminate physical penetrations (collisions) while respecting predefined motion constraints.


## Configuration File

Before running the script, you need to modify a `config.yaml` file to configure the necessary paths and optimization parameters.


## How to Use

### 1. Prepare Input Files

The script requires an input folder containing the following two files:

1.  **`scene_processed_scene.json`**:
    -   Describes the initial state of each object in the scene.

2.  **`scene_layout.txt`**:
    -   Contains a "Scene Graph" section that defines spatial relationships between objects.

### 2. Run the Optimization Script

Run the `drake_process.py` script from the command line, specifying the path to your input folder.

```bash
python drake_process.py --floder_path /path/to/your/scene_folder
```

You can also optionally specify a different path for your configuration file:

```bash
python drake_process.py --floder_path /path/to/your/scene_folder --config /path/to/your/custom_config.yaml
```

### 3. Review the Output

Upon successful execution, the script will generate a new JSON file in the folder specified by `--floder_path`. The filename will be based on the original filename and the `output_suffix` from the config file, defaulting to **`scene_processed_scene_optimized.json`**.

This file contains the new, physically optimized positions for all objects and can be used directly in subsequent rendering or simulation pipelines.