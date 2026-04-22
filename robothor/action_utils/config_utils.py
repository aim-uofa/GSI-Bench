"""
Configuration management utility functions for RoboTHOR environment
"""
import os
import json
import argparse
from copy import deepcopy
from .scene_utils import parse_scene_specification


def create_default_config():
    """创建默认配置"""
    return {
        # Scene settings
        "scenes": ["FloorPlan_Train1_3"],  # 场景列表，支持多个场景批量处理
        "output_dir": "./output_all_debug",
        "random_seed": 1234,
        "disable_physics": False,  # 是否禁用物理模拟（提高性能）
        "separate_scene_folders": True,  # 每个场景单独一个文件夹
        "resume": True,  # 是否从之前的进度继续（跳过已完成的场景/视角）
        "debug": False,  # 是否启用调试模式（输出详细信息）

        # Controller settings
        "controller": {
            "agentMode": "default",
            "gridSize": 0.25,
            "rotateStepDegrees": 90,
            "snapToGrid": True,
            "visibilityDistance": 1.5,   # 1.5米可见距离
            "renderDepthImage": True,
            "renderInstanceSegmentation": True,
            "fieldOfView": 90,
            "width": 1024,
            "height": 1024,
            "quality": "High WebGL"
        },

        # Room view generation settings
        "room_views": {
            "k_per_room": 5,
            "eps": 1.6,
            "min_members": 15,
            "sample_positions_per_room": 50,
            "require_pickupable": False,
            "pregenerated_views_path": None  # 预先生成的视角文件路径（可选）
        },

        # Command processing settings
        "command_processing": {
            "max_objects": 3,
            "pickup_only": False,
            "command_types": ["camera"],
            "n_commands_range": [3, 6],
            "min_success": 2,
            "max_fail": 8
        }
    }


def save_config(config, output_dir):
    """保存配置到JSON文件"""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"[Info] Configuration saved to {config_path}")
    return config_path


def load_config(config_path):
    """从JSON文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(f"[Info] Configuration loaded from {config_path}")
    return config


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RoboTHOR environment test with configurable parameters')

    # Basic settings
    parser.add_argument('--config', type=str, default=None, help='Path to configuration JSON file')
    parser.add_argument('--scenes', type=str, default=None,
                        help='Scene specification. Examples: '
                             '"FloorPlan_Train1_3" (single scene), '
                             '"train:1:1-5" (Train1_1 to Train1_5), '
                             '"train:1-3:1-5" (Train1_1 to Train3_5), '
                             '"train:all" (all training scenes), '
                             '"val:all" (all validation scenes)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--disable-physics', action='store_true', help='Disable physics simulation for better performance')
    parser.add_argument('--no-separate-folders', action='store_true',
                        help='Do not create separate folders for each scene (all in one folder)')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress (skip completed scenes/views)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (verbose output)')

    # Controller settings
    parser.add_argument('--grid-size', type=float, default=None, help='Grid size for agent movement')
    parser.add_argument('--visibility-distance', type=float, default=None, help='Visibility distance')
    parser.add_argument('--fov', type=int, default=None, help='Field of view')
    parser.add_argument('--width', type=int, default=None, help='Image width')
    parser.add_argument('--height', type=int, default=None, help='Image height')
    parser.add_argument('--quality', type=str, default=None, help='Rendering quality')

    # Room view settings
    parser.add_argument('--k-per-room', type=int, default=None, help='Number of views per room')
    parser.add_argument('--eps', type=float, default=None, help='Room clustering radius')
    parser.add_argument('--min-members', type=int, default=None, help='Minimum cluster members')
    parser.add_argument('--pregenerated-views', type=str, default=None,
                        help='Path to pregenerated views directory (will look for {scene_name}/selected_views.json)')

    # Command processing settings
    parser.add_argument('--max-objects', type=int, default=None, help='Maximum objects to process per view')
    parser.add_argument('--pickup-only', action='store_true', help='Only process pickupable objects')
    parser.add_argument('--command-types', type=str, nargs='+', default=None,
                        help='Command types to generate (camera, object)')

    return parser.parse_args()


def merge_config(default_config, args):
    """将命令行参数合并到默认配置中"""
    config = deepcopy(default_config)

    # 如果指定了配置文件，先加载它
    if args.config:
        loaded_config = load_config(args.config)
        # 深度合并配置
        for key in loaded_config:
            if key in config and isinstance(config[key], dict) and isinstance(loaded_config[key], dict):
                config[key].update(loaded_config[key])
            else:
                config[key] = loaded_config[key]

    # 命令行参数优先级最高
    if args.scenes is not None:
        # 解析场景规格字符串
        config['scenes'] = parse_scene_specification(args.scenes)
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    if args.seed is not None:
        config['random_seed'] = args.seed
    if args.disable_physics:
        config['disable_physics'] = True
    if args.no_separate_folders:
        config['separate_scene_folders'] = False
    if args.resume:
        config['resume'] = True
    if args.debug:
        config['debug'] = True

    # Controller settings
    if args.grid_size is not None:
        config['controller']['gridSize'] = args.grid_size
    if args.visibility_distance is not None:
        config['controller']['visibilityDistance'] = args.visibility_distance
    if args.fov is not None:
        config['controller']['fieldOfView'] = args.fov
    if args.width is not None:
        config['controller']['width'] = args.width
    if args.height is not None:
        config['controller']['height'] = args.height
    if args.quality is not None:
        config['controller']['quality'] = args.quality

    # Room view settings
    if args.k_per_room is not None:
        config['room_views']['k_per_room'] = args.k_per_room
    if args.eps is not None:
        config['room_views']['eps'] = args.eps
    if args.min_members is not None:
        config['room_views']['min_members'] = args.min_members
    if args.pregenerated_views is not None:
        config['room_views']['pregenerated_views_path'] = args.pregenerated_views

    # Command processing settings
    if args.max_objects is not None:
        config['command_processing']['max_objects'] = args.max_objects
    if args.pickup_only:
        config['command_processing']['pickup_only'] = True
    if args.command_types is not None:
        config['command_processing']['command_types'] = args.command_types

    return config
