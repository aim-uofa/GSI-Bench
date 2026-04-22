# rotate.py
"""
物体旋转命令模块
支持将物体按照指定角度（45°, 90°, 180°）顺时针或逆时针旋转
"""

import random
from .spawn_utils import _pos_dict, get_object_by_id


def generate_rotate_commands(
        obj_meta,
        controller,
        angles=(45, 90, 180),
        directions=("clockwise", "counter-clockwise")
):
    """
    生成物体旋转命令

    Args:
        obj_meta: dict, 物体元数据
        controller: AI2-THOR controller
        angles: tuple, 旋转角度选项（度）
        directions: tuple, 旋转方向 ("clockwise" 顺时针, "counter-clockwise" 逆时针)

    Returns:
        list[dict]: 旋转命令列表，每个元素包含：
            - instruction: 自然语言指令
            - angle: 旋转角度（度）
            - direction: 旋转方向
    """
    obj_type = obj_meta["objectType"]
    obj_id = obj_meta["objectId"]

    # 检查物体是否可旋转（可拾取的物体通常可以旋转）
    if not obj_meta.get("pickupable", False) and not obj_meta.get("moveable", False):
        return []

    commands = []

    for angle in angles:
        for direction in directions:
            # 构建自然语言指令
            dir_text = "clockwise" if direction == "clockwise" else "counter-clockwise"
            text = f"Rotate the {obj_type} {angle} degrees {dir_text}."

            commands.append({
                "instruction": text,
                "angle": angle,
                "direction": direction
            })

    return commands


def execute_rotate_command(controller, obj_id, angle, direction):
    """
    执行物体旋转命令

    Args:
        controller: AI2-THOR controller
        obj_id: str, 物体ID
        angle: float, 旋转角度（度）
        direction: str, 旋转方向 ("clockwise" 或 "counter-clockwise")

    Returns:
        success (bool): 是否成功
        result_dict (dict): 包含以下字段的字典
            - reason (str): 操作结果原因
            - original_pos (dict): 原始位置 {x, y, z}
            - original_rot (dict): 原始旋转 {x, y, z}
            - target_pos (dict): 目标位置 {x, y, z}（与原始位置相同）
            - target_rot (dict): 目标旋转 {x, y, z}
            - final_pos (dict): 最终位置 {x, y, z}
            - final_rot (dict): 最终旋转 {x, y, z}
            - used_force (bool): 是否使用了强制旋转
    """
    # 获取物体元数据
    obj_meta = get_object_by_id(controller, obj_id)
    if obj_meta is None:
        return False, {
            "reason": "Object not found",
            "original_pos": None,
            "original_rot": None,
            "target_pos": None,
            "target_rot": None,
            "final_pos": None,
            "final_rot": None,
            "used_force": False
        }

    # 记录原始位置和旋转
    original_pos = _pos_dict(obj_meta["position"])
    original_rot = obj_meta["rotation"]

    # 计算目标旋转角度
    # 在 AI2-THOR 中，旋转主要是绕 Y 轴（垂直轴）
    # 顺时针旋转是正方向，逆时针是负方向
    delta_y = angle if direction == "clockwise" else -angle
    target_rot = {
        "x": original_rot["x"],
        "y": (original_rot["y"] + delta_y) % 360,
        "z": original_rot["z"]
    }

    target_pos = original_pos  # 位置保持不变

    # 尝试旋转物体（不使用 forceAction）
    evt = controller.step(
        action="TeleportObject",
        objectId=obj_id,
        position=original_pos,
        rotation=target_rot,
        forceAction=False
    )

    used_force = False

    # 如果失败，尝试使用 forceAction
    if not evt.metadata.get("lastActionSuccess", False):
        evt = controller.step(
            action="TeleportObject",
            objectId=obj_id,
            position=original_pos,
            rotation=target_rot,
            forceAction=True
        )
        used_force = True

        if not evt.metadata.get("lastActionSuccess", False):
            return False, {
                "reason": evt.metadata.get("errorMessage", "rotation_failed"),
                "original_pos": original_pos,
                "original_rot": original_rot,
                "target_pos": target_pos,
                "target_rot": target_rot,
                "final_pos": None,
                "final_rot": None,
                "used_force": used_force
            }

    # 验证旋转结果
    after_meta = get_object_by_id(controller, obj_id)
    if after_meta is None:
        return False, {
            "reason": "rotation_lost_object",
            "original_pos": original_pos,
            "original_rot": original_rot,
            "target_pos": target_pos,
            "target_rot": target_rot,
            "final_pos": original_pos,
            "final_rot": None,
            "used_force": used_force
        }

    final_pos = _pos_dict(after_meta["position"])
    final_rot = after_meta["rotation"]

    # 检查位置是否发生了意外变化
    dist_err = ((final_pos["x"] - original_pos["x"])**2 +
                (final_pos["y"] - original_pos["y"])**2 +
                (final_pos["z"] - original_pos["z"])**2) ** 0.5

    if dist_err > 0.05:
        return False, {
            "reason": f"rotation_pos_changed ({dist_err:.3f}m)",
            "original_pos": original_pos,
            "original_rot": original_rot,
            "target_pos": target_pos,
            "target_rot": target_rot,
            "final_pos": final_pos,
            "final_rot": final_rot,
            "used_force": used_force
        }

    # 检查旋转角度是否正确
    def angle_diff(a, b):
        """计算两个角度之间的最小差值（考虑360度环绕）"""
        diff = abs(a - b) % 360
        return min(diff, 360 - diff)

    rot_y_err = angle_diff(final_rot["y"], target_rot["y"])
    rot_x_err = angle_diff(final_rot["x"], target_rot["x"])
    rot_z_err = angle_diff(final_rot["z"], target_rot["z"])

    # Y轴旋转是主要的旋转轴，误差应该很小
    # X和Z轴（倾斜）应该保持不变
    if rot_y_err > 5.0:
        return False, {
            "reason": f"rotation_mismatch_y ({rot_y_err:.1f} deg)",
            "original_pos": original_pos,
            "original_rot": original_rot,
            "target_pos": target_pos,
            "target_rot": target_rot,
            "final_pos": final_pos,
            "final_rot": final_rot,
            "used_force": used_force
        }

    if rot_x_err > 5.0 or rot_z_err > 5.0:
        return False, {
            "reason": f"rotation_tilt_changed (x={rot_x_err:.1f}, z={rot_z_err:.1f} deg)",
            "original_pos": original_pos,
            "original_rot": original_rot,
            "target_pos": target_pos,
            "target_rot": target_rot,
            "final_pos": final_pos,
            "final_rot": final_rot,
            "used_force": used_force
        }

    # 旋转成功
    if used_force:
        reason = f"rotation_verified_force (y_err={rot_y_err:.1f}deg)"
    else:
        reason = f"rotation_verified (y_err={rot_y_err:.1f}deg)"

    return True, {
        "reason": reason,
        "original_pos": original_pos,
        "original_rot": original_rot,
        "target_pos": target_pos,
        "target_rot": target_rot,
        "final_pos": final_pos,
        "final_rot": final_rot,
        "used_force": used_force
    }
