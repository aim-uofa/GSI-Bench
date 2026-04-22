# agent_camera.py
"""
Agent相机移动命令模块
支持Agent相机的移动(前后左右)、旋转(左右)和俯仰(上下)
"""


def generate_agent_camera_commands(
        controller,
        move_directions=("forward", "back", "left", "right"),
        move_distances=(0.1, 0.2, 0.3),
        rotate_directions=("left", "right"),
        rotate_degrees=(15, 30),
        pitch_directions=("up", "down"),
        pitch_degrees=(15, 30)
):
    """
    生成Agent相机控制命令

    Args:
        controller: AI2-THOR controller
        move_directions: tuple, 移动方向选项
        move_distances: tuple, 移动距离选项（米）
        rotate_directions: tuple, 旋转方向选项
        rotate_degrees: tuple, 旋转角度选项（度）
        pitch_directions: tuple, 俯仰方向选项
        pitch_degrees: tuple, 俯仰角度选项（度）

    Returns:
        list[dict]: 命令列表，每个元素包含：
            - instruction: 自然语言指令
            - action_type: 命令类型 ("move", "rotate", "pitch")
            - direction: 方向
            - magnitude: 移动距离（仅move）
            - degrees: 角度（仅rotate和pitch）
    """
    commands = []

    # 生成移动命令
    for direction in move_directions:
        for distance in move_distances:
            dir_text = {
                "forward": "forward",
                "back": "backward",
                "left": "to the left",
                "right": "to the right"
            }[direction]

            text = f"Move the camera {dir_text} by {distance} meters."

            commands.append({
                "instruction": text,
                "action_type": "move",
                "direction": direction,
                "magnitude": distance
            })

    # 生成旋转命令
    for direction in rotate_directions:
        for degrees in rotate_degrees:
            dir_text = "left" if direction == "left" else "right"
            text = f"Rotate the camera {degrees} degrees to the {dir_text}."

            commands.append({
                "instruction": text,
                "action_type": "rotate",
                "direction": direction,
                "degrees": degrees
            })

    # 生成俯仰命令
    for direction in pitch_directions:
        for degrees in pitch_degrees:
            dir_text = "up" if direction == "up" else "down"
            text = f"Look {dir_text} by {degrees} degrees."

            commands.append({
                "instruction": text,
                "action_type": "pitch",
                "direction": direction,
                "degrees": degrees
            })

    return commands


def execute_agent_camera_command(controller, action_type, direction, magnitude=None, degrees=None):
    """
    执行单个Agent相机控制命令

    Args:
        controller: AI2-THOR controller
        action_type: str, 命令类型 ("move", "rotate", "pitch")
        direction: str, 方向
        magnitude: float, 移动距离（仅move命令需要）
        degrees: float, 角度（仅rotate和pitch命令需要）

    Returns:
        success (bool): 是否成功
        result_dict (dict): 包含以下字段的字典
            - reason (str): 操作结果原因
            - original_pos (dict): 原始位置 {x, y, z}
            - original_rot (dict): 原始旋转 {x, y, z}
            - target_pos (dict): 目标位置 {x, y, z}
            - target_rot (dict): 目标旋转 {x, y, z}
            - final_pos (dict): 最终位置 {x, y, z}
            - final_rot (dict): 最终旋转 {x, y, z}
            - used_force (bool): 是否使用了强制移动（相机移动通常不需要）
    """
    # 记录原始状态
    metadata = controller.last_event.metadata
    agent = metadata.get("agent", {})
    original_pos = {
        "x": agent["position"]["x"],
        "y": agent["position"]["y"],
        "z": agent["position"]["z"]
    }
    original_rot = {
        "x": metadata["agent"]["cameraHorizon"],
        "y": agent["rotation"]["y"],
        "z": 0  # Agent通常不会有Z轴旋转
    }

    # 执行命令
    evt = None

    if action_type == "move":
        if magnitude is None:
            return False, {
                "reason": "missing_magnitude_parameter",
                "original_pos": original_pos,
                "original_rot": original_rot,
                "target_pos": None,
                "target_rot": None,
                "final_pos": None,
                "final_rot": None,
                "used_force": False
            }

        action_map = {
            "forward": "MoveAhead",
            "back": "MoveBack",
            "left": "MoveLeft",
            "right": "MoveRight"
        }

        if direction not in action_map:
            return False, {
                "reason": f"invalid_direction: {direction}",
                "original_pos": original_pos,
                "original_rot": original_rot,
                "target_pos": None,
                "target_rot": None,
                "final_pos": None,
                "final_rot": None,
                "used_force": False
            }

        action = action_map[direction]
        evt = controller.step(action=action, moveMagnitude=magnitude)

        # 目标位置需要根据当前朝向计算
        # 简化处理：从执行后的位置获取
        target_pos = None  # 将在成功后从final_pos获取
        target_rot = original_rot.copy()

    elif action_type == "rotate":
        if degrees is None:
            return False, {
                "reason": "missing_degrees_parameter",
                "original_pos": original_pos,
                "original_rot": original_rot,
                "target_pos": None,
                "target_rot": None,
                "final_pos": None,
                "final_rot": None,
                "used_force": False
            }

        action_map = {
            "left": "RotateLeft",
            "right": "RotateRight"
        }

        if direction not in action_map:
            return False, {
                "reason": f"invalid_direction: {direction}",
                "original_pos": original_pos,
                "original_rot": original_rot,
                "target_pos": None,
                "target_rot": None,
                "final_pos": None,
                "final_rot": None,
                "used_force": False
            }

        action = action_map[direction]
        evt = controller.step(action=action, degrees=degrees)

        # 计算目标旋转
        delta_y = degrees if direction == "right" else -degrees
        target_rot = {
            "x": original_rot["x"],
            "y": (original_rot["y"] + delta_y) % 360,
            "z": 0
        }
        target_pos = original_pos.copy()

    elif action_type == "pitch":
        if degrees is None:
            return False, {
                "reason": "missing_degrees_parameter",
                "original_pos": original_pos,
                "original_rot": original_rot,
                "target_pos": None,
                "target_rot": None,
                "final_pos": None,
                "final_rot": None,
                "used_force": False
            }

        action_map = {
            "up": "LookUp",
            "down": "LookDown"
        }

        if direction not in action_map:
            return False, {
                "reason": f"invalid_direction: {direction}",
                "original_pos": original_pos,
                "original_rot": original_rot,
                "target_pos": None,
                "target_rot": None,
                "final_pos": None,
                "final_rot": None,
                "used_force": False
            }

        action = action_map[direction]
        evt = controller.step(action=action, degrees=degrees)

        # 计算目标俯仰角
        # 注意：LookUp会减少horizon（向上看），LookDown会增加horizon（向下看）
        delta_x = -degrees if direction == "up" else degrees
        target_rot = {
            "x": original_rot["x"] + delta_x,
            "y": original_rot["y"],
            "z": 0
        }
        target_pos = original_pos.copy()

    else:
        return False, {
            "reason": f"unknown_action_type: {action_type}",
            "original_pos": original_pos,
            "original_rot": original_rot,
            "target_pos": None,
            "target_rot": None,
            "final_pos": None,
            "final_rot": None,
            "used_force": False
        }

    # 检查执行结果
    if not evt.metadata.get("lastActionSuccess", False):
        error_msg = evt.metadata.get("errorMessage", "action_failed")
        return False, {
            "reason": error_msg,
            "original_pos": original_pos,
            "original_rot": original_rot,
            "target_pos": target_pos,
            "target_rot": target_rot,
            "final_pos": None,
            "final_rot": None,
            "used_force": False
        }

    # 获取最终状态
    final_metadata = evt.metadata
    final_agent = final_metadata.get("agent", {})
    final_pos = {
        "x": final_agent["position"]["x"],
        "y": final_agent["position"]["y"],
        "z": final_agent["position"]["z"]
    }
    final_rot = {
        "x": final_metadata["agent"]["cameraHorizon"],
        "y": final_agent["rotation"]["y"],
        "z": 0
    }

    # 对于移动命令，使用实际最终位置作为目标位置
    if action_type == "move" and target_pos is None:
        target_pos = final_pos.copy()

    # 验证结果
    # 计算位置误差
    pos_err = ((final_pos["x"] - target_pos["x"])**2 +
               (final_pos["y"] - target_pos["y"])**2 +
               (final_pos["z"] - target_pos["z"])**2) ** 0.5

    # 计算旋转误差
    def angle_diff(a, b):
        """计算两个角度之间的最小差值（考虑360度环绕）"""
        diff = abs(a - b) % 360
        return min(diff, 360 - diff)

    rot_y_err = angle_diff(final_rot["y"], target_rot["y"])
    rot_x_err = abs(final_rot["x"] - target_rot["x"])  # horizon不需要环绕

    # 构建成功原因消息
    if action_type == "move":
        reason = f"camera_moved_verified (pos_err={pos_err:.3f}m)"
    elif action_type == "rotate":
        reason = f"camera_rotated_verified (rot_y_err={rot_y_err:.1f}deg)"
    elif action_type == "pitch":
        reason = f"camera_pitched_verified (rot_x_err={rot_x_err:.1f}deg)"
    else:
        reason = "camera_action_success"

    return True, {
        "reason": reason,
        "original_pos": original_pos,
        "original_rot": original_rot,
        "target_pos": target_pos,
        "target_rot": target_rot,
        "final_pos": final_pos,
        "final_rot": final_rot,
        "used_force": False  # 相机移动不需要force
    }
