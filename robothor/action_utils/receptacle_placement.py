# receptacle_placement.py
"""
容器表面放置命令模块
支持将物体移动到指定容器的表面特定位置（中心、边缘等）
"""

import random
import math
from .spawn_utils import (
    _pos_dict, _dist3, _dist2, get_object_by_id,
    get_spawn_coords_on_receptacle, get_horizontal_axes
)


def compute_placement_position(coords, placement_type, camera_pos, right_vec, forward_vec):
    """
    根据放置类型从可用坐标中选择目标位置

    Args:
        coords: list[dict], 容器表面的spawn坐标列表
        placement_type: str, 放置类型
        camera_pos: dict, 相机位置 {x, y, z}
        right_vec: dict, 相机右向量 {x, y, z}
        forward_vec: dict, 相机前向量 {x, y, z}

    Returns:
        dict or None: 目标位置坐标 {x, y, z}
    """
    if not coords:
        return None

    if placement_type == "center":
        # 计算所有坐标的中心点
        center_x = sum(c["x"] for c in coords) / len(coords)
        center_y = sum(c["y"] for c in coords) / len(coords)
        center_z = sum(c["z"] for c in coords) / len(coords)
        target = {"x": center_x, "y": center_y, "z": center_z}
        # 找到最接近中心的实际spawn点
        return min(coords, key=lambda c: _dist3(c, target))

    elif placement_type == "nearest_to_camera":
        # 离相机最近的点
        return min(coords, key=lambda c: _dist3(c, camera_pos))

    elif placement_type == "farthest_from_camera":
        # 离相机最远的点
        return max(coords, key=lambda c: _dist3(c, camera_pos))

    elif placement_type == "leftmost":
        # 最左边的点（相机视角）
        # 左边对应于 -right_vec 方向
        def left_projection(c):
            # 投影到相机右向量上，越小越靠左
            return c["x"] * right_vec["x"] + c["z"] * right_vec["z"]
        return min(coords, key=left_projection)

    elif placement_type == "rightmost":
        # 最右边的点（相机视角）
        def right_projection(c):
            return c["x"] * right_vec["x"] + c["z"] * right_vec["z"]
        return max(coords, key=right_projection)

    elif placement_type == "frontmost":
        # 最靠近相机前方的点
        def front_projection(c):
            return c["x"] * forward_vec["x"] + c["z"] * forward_vec["z"]
        return max(coords, key=front_projection)

    elif placement_type == "backmost":
        # 最远离相机前方的点
        def back_projection(c):
            return c["x"] * forward_vec["x"] + c["z"] * forward_vec["z"]
        return min(coords, key=back_projection)

    return None


def estimate_available_placement_types(
        controller,
        obj_meta,
        receptacle_meta,
        placement_types=("center", "nearest_to_camera", "farthest_from_camera",
                        "leftmost", "rightmost", "frontmost", "backmost"),
        min_coords=3
):
    """
    估算容器上哪些放置类型是可行的

    Args:
        controller: AI2-THOR controller
        obj_meta: dict, 要移动的物体元数据
        receptacle_meta: dict, 目标容器元数据
        placement_types: tuple, 要测试的放置类型
        min_coords: int, 最少需要的spawn坐标数量

    Returns:
        dict: {placement_type: target_position}
    """
    rec_id = receptacle_meta["objectId"]

    # 获取容器表面的spawn坐标
    coords = get_spawn_coords_on_receptacle(controller, rec_id, anywhere=False)
    # if not coords:
    #     coords = get_spawn_coords_on_receptacle(controller, rec_id, anywhere=True)

    if not coords or len(coords) < min_coords:
        return {}

    # 获取相机信息
    agent = controller.last_event.metadata["agent"]
    camera_pos = _pos_dict(agent["position"])
    right_vec, forward_vec = get_horizontal_axes(agent)

    results = {}
    for ptype in placement_types:
        target_pos = compute_placement_position(coords, ptype, camera_pos, right_vec, forward_vec)
        if target_pos:
            results[ptype] = target_pos

    return results


def generate_receptacle_placement_commands(
        obj_meta,
        receptacle_meta,
        controller,
        placement_types=("center", "nearest_to_camera", "farthest_from_camera",
                        "leftmost", "rightmost"),
        sample_types=None
):
    """
    生成将物体放置到容器表面特定位置的命令

    Args:
        obj_meta: dict, 要移动的物体元数据
        receptacle_meta: dict, 目标容器元数据
        controller: AI2-THOR controller
        placement_types: tuple, 可选的放置类型
        sample_types: list or None, 如果指定，只生成这些类型的命令

    Returns:
        list[dict]: 命令列表，每个元素包含：
            - instruction: 自然语言指令
            - receptacle_id: 目标容器ID
            - placement_type: 放置类型
            - target_pos: 目标位置（用于验证）
    """
    obj_type = obj_meta["objectType"]
    rec_type = receptacle_meta["objectType"]
    rec_id = receptacle_meta["objectId"]

    # 估算可用的放置位置
    available_placements = estimate_available_placement_types(
        controller, obj_meta, receptacle_meta, placement_types=placement_types
    )

    if not available_placements:
        return []

    # 如果指定了sample_types，只使用这些类型
    if sample_types:
        available_placements = {k: v for k, v in available_placements.items() if k in sample_types}

    # 生成命令
    commands = []

    # 位置类型的自然语言描述
    placement_descriptions = {
        "center": "center",
        "nearest_to_camera": "edge nearest to the camera",
        "farthest_from_camera": "edge farthest from the camera",
        "leftmost": "leftmost edge",
        "rightmost": "rightmost edge",
        "frontmost": "front edge nearest to the camera",
        "backmost": "back edge farthest from the camera"
    }

    for ptype, target_pos in available_placements.items():
        desc = placement_descriptions.get(ptype, ptype)
        text = f"Move the {obj_type} to the {desc} of the {rec_type}."

        commands.append({
            "instruction": text,
            "receptacle_id": rec_id,
            "placement_type": ptype,
            "target_pos": target_pos  # 保存目标位置用于执行时验证
        })

    return commands


def execute_receptacle_placement_command(
        controller,
        obj_id,
        receptacle_id,
        placement_type="center"
):
    """
    执行将物体放置到容器表面特定位置的命令

    Args:
        controller: AI2-THOR controller
        obj_id: str, 物体ID
        receptacle_id: str, 目标容器ID
        placement_type: str, 放置类型

    Returns:
        success (bool): 是否成功
        result_dict (dict): 包含以下字段的字典
            - reason (str): 操作结果原因
            - original_pos (dict): 原始位置
            - original_rot (dict): 原始旋转
            - target_pos (dict): 目标位置
            - target_rot (dict): 目标旋转
            - final_pos (dict): 最终位置
            - final_rot (dict): 最终旋转
            - used_force (bool): 是否使用了强制移动
            - receptacle_id (str): 容器ID
            - placement_type (str): 放置类型
    """
    # 获取物体和容器元数据
    obj_meta = get_object_by_id(controller, obj_id)
    rec_meta = get_object_by_id(controller, receptacle_id)

    def _fail(reason, **extra):
        return False, {
            "reason": reason,
            "receptacle_id": receptacle_id,
            "placement_type": placement_type,
            **extra,
            "used_force": extra.get("used_force", False)
        }

    if obj_meta is None:
        return _fail("Object not found")

    if rec_meta is None:
        return _fail("Receptacle not found",
                    original_pos=_pos_dict(obj_meta["position"]),
                    original_rot=obj_meta["rotation"])

    # 记录原始位置和旋转
    original_pos = _pos_dict(obj_meta["position"])
    original_rot = obj_meta["rotation"]

    # 获取容器表面的spawn坐标
    coords = get_spawn_coords_on_receptacle(controller, receptacle_id, anywhere=False)
    if not coords:
        coords = get_spawn_coords_on_receptacle(controller, receptacle_id, anywhere=True)

    if not coords:
        return _fail("No spawn coordinates available on receptacle",
                    original_pos=original_pos,
                    original_rot=original_rot)

    # 获取相机信息
    agent = controller.last_event.metadata["agent"]
    camera_pos = _pos_dict(agent["position"])
    right_vec, forward_vec = get_horizontal_axes(agent)

    # 计算目标位置
    target_pos = compute_placement_position(
        coords, placement_type, camera_pos, right_vec, forward_vec
    )

    if target_pos is None:
        return _fail(f"Cannot compute target position for placement_type: {placement_type}",
                    original_pos=original_pos,
                    original_rot=original_rot)

    target_rot = original_rot  # 保持原始旋转

    # 尝试传送物体到目标位置
    evt = controller.step(
        action="TeleportObject",
        objectId=obj_id,
        position=target_pos,
        rotation=original_rot,
        forceAction=False
    )

    used_force = False
    used_pos = target_pos

    if not evt.metadata.get("lastActionSuccess", False):
        # 尝试稍微抬高 + forceAction
        bumped = dict(target_pos)
        bumped["y"] += 0.02
        evt = controller.step(
            action="TeleportObject",
            objectId=obj_id,
            position=bumped,
            rotation=original_rot,
            forceAction=True
        )
        used_force = True
        used_pos = bumped

        if not evt.metadata.get("lastActionSuccess", False):
            return _fail(evt.metadata.get("errorMessage", "teleport_failed"),
                        original_pos=original_pos,
                        original_rot=original_rot,
                        target_pos=target_pos,
                        target_rot=target_rot)

    # 验证最终位置
    after_meta = get_object_by_id(controller, obj_id)
    if after_meta is None:
        return _fail("teleport_lost_object",
                    original_pos=original_pos,
                    original_rot=original_rot,
                    target_pos=target_pos,
                    target_rot=target_rot,
                    final_pos=used_pos)

    # 检查物体是否可见
    visible = after_meta.get("visible", False)
    if not visible:
        return _fail("object_not_visible_after_placement",
                    original_pos=original_pos,
                    original_rot=original_rot,
                    target_pos=target_pos,
                    target_rot=target_rot,
                    final_pos=used_pos,
                    final_rot=original_rot)

    final_pos = _pos_dict(after_meta["position"])
    final_rot = after_meta["rotation"]

    # 检查位置偏差
    dist_err = ((final_pos["x"] - used_pos["x"]) ** 2 +
                # (final_pos["y"] - used_pos["y"]) ** 2 + #高度忽略，因为一点会有误差
                (final_pos["z"] - used_pos["z"]) ** 2) ** 0.5

    # 检查旋转偏差
    def compute_rotation_error(final_rot, original_rot):
        def angle_diff(a, b):
            diff = abs(a - b) % 360
            return min(diff, 360 - diff)

        rot_x = angle_diff(final_rot["x"], original_rot["x"])
        rot_y = angle_diff(final_rot["y"], original_rot["y"])
        rot_z = angle_diff(final_rot["z"], original_rot["z"])
        return rot_x + rot_y + rot_z

    rot_err = compute_rotation_error(final_rot, original_rot)

    if dist_err > 0.05:
        return _fail(f"teleport_mismatch_pos ({dist_err:.3f}m)",
                    original_pos=original_pos,
                    original_rot=original_rot,
                    target_pos=target_pos,
                    target_rot=target_rot,
                    final_pos=final_pos,
                    final_rot=final_rot)

    if rot_err > 5.0:
        return _fail(f"teleport_mismatch_rot ({rot_err:.1f} deg)",
                    original_pos=original_pos,
                    original_rot=original_rot,
                    target_pos=target_pos,
                    target_rot=target_rot,
                    final_pos=final_pos,
                    final_rot=final_rot)

    # 成功
    reason = f"placement_verified{'_force' if used_force else ''} (err={dist_err:.3f}m, type={placement_type})"
    return True, {
        "reason": reason,
        "original_pos": original_pos,
        "original_rot": original_rot,
        "target_pos": target_pos,
        "target_rot": target_rot,
        "final_pos": final_pos,
        "final_rot": final_rot,
        "used_force": used_force,
        "visible": visible,
        "receptacle_id": receptacle_id,
        "placement_type": placement_type
    }
