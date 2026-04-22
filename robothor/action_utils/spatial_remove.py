# spatial_remove.py
"""
空间理解物体移除模块
根据空间关系（最近、最远、最左、最右等）选择并移除某个类别的物体
"""
import random
import math
from .spawn_utils import (
    _pos_dict, _dist3, get_object_by_id, get_camera_axes, _dist2, get_horizontal_axes
)


def get_objects_by_type(controller, object_type, visible_only=True):
    """
    获取场景中某个类别的所有物体

    Args:
        controller: AI2-THOR controller
        object_type: 物体类型（如 "Apple", "Mug" 等）
        visible_only: 是否只返回可见的物体

    Returns:
        list: 符合条件的物体列表
    """
    all_objs = controller.last_event.metadata.get("objects", [])
    result = []

    for obj in all_objs:
        if obj["objectType"] == object_type:
            if visible_only and not obj.get("visible", False):
                continue
            result.append(obj)

    return result


def compute_spatial_relations(controller, objects, min_difference_threshold=0.15):
    """
    计算物体相对于相机的空间关系

    Args:
        controller: AI2-THOR controller
        objects: 物体列表
        min_difference_threshold: 最小区分度阈值（米）
            - 对于距离（closest/farthest）：与次极值的距离差
            - 对于水平位置（leftmost/rightmost）：水平偏移量的差值
            - 对于垂直位置（highest/lowest）：高度差
            如果差值小于此阈值，则认为区分度不够，不返回该关系

    Returns:
        dict: 包含各种空间关系的字典
            - closest_to_camera: 最近的物体
            - farthest_from_camera: 最远的物体
            - leftmost: 最左边的物体
            - rightmost: 最右边的物体
            - highest: 最高的物体
            - lowest: 最低的物体
    """
    if not objects:
        return {}

    # 如果只有一个物体，无法判断相对关系
    if len(objects) == 1:
        return {}

    # 获取相机位置
    camera_pos = controller.last_event.metadata["cameraPosition"]
    camera_dict = {"x": camera_pos["x"], "y": camera_pos["y"], "z": camera_pos["z"]}

    # 获取相机的水平方向轴（只需要right用于计算左右关系）
    right, _ = get_horizontal_axes(controller)

    # 计算每个物体的空间属性
    obj_with_spatial = []
    for obj in objects:
        obj_pos = _pos_dict(obj["position"])

        # 计算距离
        distance = _dist3(obj_pos, camera_dict)

        # 计算相对位置向量（从相机到物体）
        rel_vec = {
            "x": obj_pos["x"] - camera_dict["x"],
            "y": obj_pos["y"] - camera_dict["y"],
            "z": obj_pos["z"] - camera_dict["z"]
        }

        # 计算在相机右方向轴上的投影（正值表示右边，负值表示左边）
        horizontal_offset = (rel_vec["x"] * right["x"] +
                           rel_vec["z"] * right["z"])

        # 获取物体的垂直位置（y坐标）
        vertical_position = obj_pos["y"]

        obj_with_spatial.append({
            "object": obj,
            "distance": distance,
            "horizontal_offset": horizontal_offset,
            "vertical_position": vertical_position
        })

    # 按不同标准排序并选择（添加区分度检查）
    result = {}

    # 最近的（需要与第二近的有足够距离差）
    sorted_by_distance = sorted(obj_with_spatial, key=lambda x: x["distance"])
    if len(sorted_by_distance) >= 2:
        closest = sorted_by_distance[0]
        second_closest = sorted_by_distance[1]
        distance_diff = second_closest["distance"] - closest["distance"]
        if distance_diff >= min_difference_threshold:
            result["closest_to_camera"] = closest["object"]

    # 最远的（需要与第二远的有足够距离差）
    if len(sorted_by_distance) >= 2:
        farthest = sorted_by_distance[-1]
        second_farthest = sorted_by_distance[-2]
        distance_diff = farthest["distance"] - second_farthest["distance"]
        if distance_diff >= min_difference_threshold:
            result["farthest_from_camera"] = farthest["object"]

    # 最左边的（horizontal_offset最小，需要与第二左的有足够差距）
    sorted_by_horizontal = sorted(obj_with_spatial, key=lambda x: x["horizontal_offset"])
    if len(sorted_by_horizontal) >= 2:
        leftmost = sorted_by_horizontal[0]
        second_leftmost = sorted_by_horizontal[1]
        horizontal_diff = abs(second_leftmost["horizontal_offset"] - leftmost["horizontal_offset"])
        if horizontal_diff >= min_difference_threshold:
            result["leftmost"] = leftmost["object"]

    # 最右边的（horizontal_offset最大，需要与第二右的有足够差距）
    if len(sorted_by_horizontal) >= 2:
        rightmost = sorted_by_horizontal[-1]
        second_rightmost = sorted_by_horizontal[-2]
        horizontal_diff = abs(rightmost["horizontal_offset"] - second_rightmost["horizontal_offset"])
        if horizontal_diff >= min_difference_threshold:
            result["rightmost"] = rightmost["object"]

    # 最高的（vertical_position最大，需要与第二高的有足够高度差）
    sorted_by_vertical = sorted(obj_with_spatial, key=lambda x: x["vertical_position"])
    if len(sorted_by_vertical) >= 2:
        highest = sorted_by_vertical[-1]
        second_highest = sorted_by_vertical[-2]
        vertical_diff = highest["vertical_position"] - second_highest["vertical_position"]
        if vertical_diff >= min_difference_threshold:
            result["highest"] = highest["object"]

    # 最低的（vertical_position最小，需要与第二低的有足够高度差）
    if len(sorted_by_vertical) >= 2:
        lowest = sorted_by_vertical[0]
        second_lowest = sorted_by_vertical[1]
        vertical_diff = abs(second_lowest["vertical_position"] - lowest["vertical_position"])
        if vertical_diff >= min_difference_threshold:
            result["lowest"] = lowest["object"]

    return result


def generate_spatial_remove_commands(
    controller,
    min_count=2,
    spatial_relations=("closest_to_camera", "farthest_from_camera", "leftmost", "rightmost", "highest", "lowest"),
    moveable_only=False,
    exclude_types=None,
    min_difference_threshold=0.05
):
    """
    生成空间理解的物体移除命令

    Args:
        controller: AI2-THOR controller
        min_count: 至少需要多少个同类物体才生成命令（默认2个）
        spatial_relations: 要生成的空间关系列表
        moveable_only: 是否只选择 pickupable 或 moveable 的物体（默认False，即所有可见物体）
        exclude_types: 要排除的物体类型列表（默认排除 Drawer, Floor, Wall 等）
        min_difference_threshold: 最小区分度阈值（米），确保选中的物体在该维度上与其他物体有足够区分度

    Returns:
        list: 命令列表，每个命令包含：
            - instruction: 自然语言指令
            - object_type: 物体类型
            - object_id: 目标物体ID
            - spatial_relation: 空间关系类型
    """
    commands = []

    # 默认排除的物体类型
    if exclude_types is None:
        exclude_types = {
            "Drawer",      # 抽屉 - 通常是家具的一部分
            "Floor",       # 地板
            "Wall",        # 墙壁
            "Window",      # 窗户
            "Door",        # 门
            "Ceiling",     # 天花板
            "Room",        # 房间
            "Shelf",        # 架子 - 通常是家具的一部分
        }
    else:
        exclude_types = set(exclude_types)

    # 获取所有可见的物体
    all_objs = controller.last_event.metadata.get("objects", [])

    if moveable_only:
        # 只选择可移动的物体
        visible_objs = [o for o in all_objs if o.get("visible", False) and
                        (o.get("pickupable", False) or o.get("moveable", False)) and
                        o["objectType"] not in exclude_types]
    else:
        # 选择所有可见物体（排除黑名单）
        visible_objs = [o for o in all_objs if o.get("visible", False) and
                        o["objectType"] not in exclude_types]

    # 按类型分组
    type_groups = {}
    for obj in visible_objs:
        obj_type = obj["objectType"]
        if obj_type not in type_groups:
            type_groups[obj_type] = []
        type_groups[obj_type].append(obj)

    # 调试信息：打印每种物体类型的数量
    print(f"[Debug] Found {len(visible_objs)} visible objects after filtering")
    for obj_type, objs in sorted(type_groups.items(), key=lambda x: len(x[1]), reverse=True):
        if len(objs) >= min_count:
            print(f"[Debug]   - {obj_type}: {len(objs)} instances (will generate commands)")

    # 对于每个有多个实例的类型，生成移除命令
    for obj_type, objs in type_groups.items():
        if len(objs) < min_count:
            continue

        # 计算空间关系（传入区分度阈值）
        spatial_dict = compute_spatial_relations(controller, objs, min_difference_threshold)

        # 调试信息：显示哪些关系可用
        available_relations = list(spatial_dict.keys())
        if available_relations:
            print(f"[Debug]   - {obj_type}: Available relations = {available_relations}")
        else:
            print(f"[Debug]   - {obj_type}: No relations with sufficient distinction (threshold={min_difference_threshold}m)")
            continue

        # 为每个空间关系生成命令
        for relation in spatial_relations:
            if relation not in spatial_dict:
                continue

            target_obj = spatial_dict[relation]

            # 生成自然语言指令
            relation_text_map = {
                "closest_to_camera": "at the closest position to the camera",
                "farthest_from_camera": "at the farthest position from the camera",
                "leftmost": "at the leftmost position",
                "rightmost": "at the rightmost position",
                "highest": "at the highest position",
                "lowest": "at the lowest position"
            }

            relation_text = relation_text_map.get(relation, relation)
            instruction = f"Remove the {obj_type} {relation_text}."

            commands.append({
                "instruction": instruction,
                "object_type": obj_type,
                "object_id": target_obj["objectId"],
                "spatial_relation": relation
            })

    return commands


def execute_spatial_remove_command(controller, object_id):
    """
    执行物体移除命令

    Args:
        controller: AI2-THOR controller
        object_id: 要移除的物体ID

    Returns:
        success (bool): 是否成功
        result_dict (dict): 包含以下字段的字典
            - reason (str): 操作结果原因
            - original_pos (dict): 原始位置 {x, y, z}
            - original_rot (dict): 原始旋转 {x, y, z}
            - object_was_visible (bool): 物体原本是否可见
            - object_removed (bool): 物体是否成功移除
    """
    # 获取物体元数据
    obj_meta = get_object_by_id(controller, object_id)
    if obj_meta is None:
        return False, {
            "reason": "Object not found",
            "original_pos": None,
            "original_rot": None,
            "object_was_visible": False,
            "object_removed": False
        }

    # 记录原始状态
    original_pos = _pos_dict(obj_meta["position"])
    original_rot = obj_meta["rotation"]
    was_visible = obj_meta.get("visible", False)

    # 执行DisableObject操作
    evt = controller.step(
        action="DisableObject",
        objectId=object_id
    )

    if not evt.metadata.get("lastActionSuccess", False):
        error_msg = evt.metadata.get("errorMessage", "Unknown error")
        return False, {
            "reason": f"DisableObject failed: {error_msg}",
            "original_pos": original_pos,
            "original_rot": original_rot,
            "object_was_visible": was_visible,
            "object_removed": False
        }

    # 验证物体是否真的被移除
    after_obj = get_object_by_id(controller, object_id)

    # 如果物体不存在或不再可见，视为成功移除
    is_removed = after_obj is None or not after_obj.get("visible", False)

    if is_removed:
        return True, {
            "reason": "Object successfully removed",
            "original_pos": original_pos,
            "original_rot": original_rot,
            "object_was_visible": was_visible,
            "object_removed": True
        }
    else:
        return False, {
            "reason": "Object still visible after DisableObject",
            "original_pos": original_pos,
            "original_rot": original_rot,
            "object_was_visible": was_visible,
            "object_removed": False
        }
