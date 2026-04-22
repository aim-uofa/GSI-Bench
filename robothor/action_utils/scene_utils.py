"""
Scene management utility functions for RoboTHOR environment
"""


def generate_scene_list(scene_type="train", start_major=1, end_major=1, start_minor=1, end_minor=5):
    """
    生成场景列表

    Args:
        scene_type: "train" 或 "val"
        start_major: 主编号起始 (train: 1-12, val: 1-3)
        end_major: 主编号结束
        start_minor: 次编号起始 (1-5)
        end_minor: 次编号结束

    Returns:
        场景名称列表
    """
    scenes = []
    prefix = "FloorPlan_Train" if scene_type.lower() == "train" else "FloorPlan_Val"

    for major in range(start_major, end_major + 1):
        for minor in range(start_minor, end_minor + 1):
            scenes.append(f"{prefix}{major}_{minor}")

    return scenes


def parse_scene_specification(scene_spec):
    """
    解析场景规格字符串

    支持的格式:
    - 单个场景: "FloorPlan_Train1_3"
    - 多个场景（逗号分隔）: "FloorPlan_Train1_1,FloorPlan_Train1_2"
    - 范围: "train:1-3:1-5" (train场景，主编号1-3，次编号1-5)
    - 范围简写: "train:1:1-5" (train场景，主编号1，次编号1-5)
    - 全部训练场景: "train:all" (FloorPlan_Train1_1 到 FloorPlan_Train12_5)
    - 全部验证场景: "val:all" (FloorPlan_Val1_1 到 FloorPlan_Val3_5)

    Returns:
        场景名称列表
    """
    scene_spec = scene_spec.strip()

    # 单个场景或逗号分隔的多个场景
    if scene_spec.startswith("FloorPlan_"):
        return [s.strip() for s in scene_spec.split(",")]

    # 范围格式
    parts = scene_spec.split(":")
    scene_type = parts[0].lower()  # train 或 val

    if len(parts) == 2 and parts[1] == "all":
        # 全部场景
        if scene_type == "train":
            return generate_scene_list("train", 1, 12, 1, 5)
        else:  # val
            return generate_scene_list("val", 1, 3, 1, 5)

    # 解析范围
    if len(parts) >= 2:
        major_part = parts[1]
        minor_part = parts[2] if len(parts) > 2 else "1-5"

        # 解析主编号
        if "-" in major_part:
            start_major, end_major = map(int, major_part.split("-"))
        else:
            start_major = end_major = int(major_part)

        # 解析次编号
        if "-" in minor_part:
            start_minor, end_minor = map(int, minor_part.split("-"))
        else:
            start_minor = end_minor = int(minor_part)

        return generate_scene_list(scene_type, start_major, end_major, start_minor, end_minor)

    # 默认返回单个场景
    return [scene_spec]
