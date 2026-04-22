# object_relative.py
import random
from .spawn_utils import (
    _pos_dict, _dist3, get_object_by_id,
    get_first_receptacle_id_of, get_spawn_coords_on_receptacle,
    pick_nearest_coord, place_object_at_coord, get_horizontal_axes, pick_best_coord,
    get_closest_receptacle_id
)


def estimate_max_move_distance_for_object_relative(
        controller,
        obj_meta,
        anchor_meta,
        relations=("to the right of", "to the left of", "in front of", "behind"),
        test_distances=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
        dist_threshold=0.15
):
    """
    估算物体相对于锚点物体在各个方向上的最大可移动距离。

    Args:
        controller: AI2-THOR controller
        obj_meta: 要移动的物体元数据
        anchor_meta: 锚点物体元数据
        relations: 相对方向元组
        test_distances: 测试距离列表（米）
        dist_threshold: 目标位置与最近有效坐标的最大距离阈值（米）

    Returns:
        dict: {relation: max_valid_distance}
    """
    results = {}

    # 获取锚点位置
    base = _pos_dict(anchor_meta["position"])

    # 获取相机方向向量
    right, forward = get_horizontal_axes(controller.last_event.metadata["agent"])

    # 找到可用的 receptacle（优先使用锚点所在的 receptacle）
    rec_id = get_closest_receptacle_id(controller, anchor_meta)
    if rec_id is None:
        # 如果锚点没有 receptacle，尝试使用物体的
        rec_id = get_closest_receptacle_id(controller, obj_meta)

    if rec_id is None:
        return results

    # 获取 receptacle 表面的 spawn 点
    coords = get_spawn_coords_on_receptacle(controller, rec_id, anywhere=False)
    if not coords:
        coords = get_spawn_coords_on_receptacle(controller, rec_id, anywhere=True)
    if not coords:
        return results

    # 获取表面高度（使用中位数）
    ys = sorted(c["y"] for c in coords)
    surface_y = ys[len(ys)//2]

    # 定义方向映射
    direction_map = {
        "to the right of": right,
        "to the left of": {"x": -right["x"], "y": 0.0, "z": -right["z"]},
        "in front of": {"x": -forward["x"], "y": 0.0, "z": -forward["z"]},
        "behind": forward
    }

    # 测试每个方向
    for relation in relations:
        if relation not in direction_map:
            continue

        dir_vec = direction_map[relation]
        best_valid_dist = 0.0

        for dist in test_distances:
            # 计算目标位置（只考虑中心点距离）
            target = {
                "x": base["x"] + dir_vec["x"] * dist,
                "y": surface_y,
                "z": base["z"] + dir_vec["z"] * dist
            }

            # 找到最近的有效坐标
            nearest = pick_nearest_coord(coords, target)
            if nearest is None:
                break

            # 检查距离是否在阈值内
            if _dist3(nearest, target) < dist_threshold:
                best_valid_dist = dist
            else:
                break

        if best_valid_dist > 0:
            results[relation] = best_valid_dist

    return results


def generate_object_relative_commands(
        obj_meta,
        anchor_meta,
        controller,
        relations=("to the right of", "to the left of", "in front of", "behind"),
        sample_distances=(0.10, 0.15, 0.20),
        test_distances=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
):
    """
    生成基于相机坐标系（camera-centric）的无歧义物体相对指令。
    只生成经过验证可行的指令。

    Args:
        obj_meta: dict, 要移动的物体元数据
        anchor_meta: dict, 参照物体元数据
        controller: AI2-THOR controller
        relations: 相对方向
        sample_distances: 距离样本 (米)
        test_distances: 用于测试的距离范围

    Returns:
        list[dict]: 每个元素为一个完整的可执行指令配置
    """
    obj_type = obj_meta["objectType"]
    anchor_type = anchor_meta["objectType"]
    anchor_id = anchor_meta["objectId"]

    # 估算每个方向的最大有效移动距离
    max_ranges = estimate_max_move_distance_for_object_relative(
        controller, obj_meta, anchor_meta,
        relations=relations,
        test_distances=test_distances
    )

    # 只生成有效范围内的命令
    commands = []
    for rel, max_d in max_ranges.items():
        for dist in sample_distances:
            if dist <= max_d:
                dist_cm = int(dist * 100)
                text = (
                    f"Move the {obj_type}'s center {rel} the {anchor_type}'s center "
                    f"by {dist_cm} centimeters, relative to the camera view."
                )

                commands.append({
                    "instruction": text,             # 无歧义自然语言指令
                    "relation": rel,                 # 相对方向
                    "anchor_id": anchor_id,          # 参照物体 ID
                    "dist_m": dist,                  # 移动距离 (m)，表示中心点之间的距离
                    "reference": "camera"            # 明确标注相机坐标系
                })

    return commands


# def execute_object_relative_command(
#         controller,
#         obj_id,
#         anchor_id,
#         relation="to the right of",
#         dist_m=0.1
# ):
#     """
#     移动物体到相对于锚点物体的位置，使用 TeleportObject 直接定位。

#     Args:
#         controller: AI2-THOR controller
#         obj_id: 要移动的物体ID
#         anchor_id: 锚点物体ID
#         relation: 相对关系 ("to the right of", "to the left of", "in front of", "behind")
#         dist_m: 距离（米）

#     Returns:
#         success (bool): 是否成功
#         result_dict (dict): 包含以下字段的字典
#             - reason (str): 操作结果原因
#             - original_pos (dict): 原始位置 {x, y, z}
#             - original_rot (dict): 原始旋转 {x, y, z}
#             - target_pos (dict): 目标位置 {x, y, z}
#             - target_rot (dict): 目标旋转 {x, y, z}（与原始旋转相同）
#             - final_pos (dict): 最终位置 {x, y, z}
#             - final_rot (dict): 最终旋转 {x, y, z}
#             - used_force (bool): 是否使用了强制移动
#     """
#     # === Step 1. 获取物体元数据 ===
#     obj_meta = get_object_by_id(controller, obj_id)
#     anchor_meta = get_object_by_id(controller, anchor_id)

#     if obj_meta is None:
#         return False, {
#             "reason": "Object not found",
#             "original_pos": None,
#             "original_rot": None,
#             "target_pos": None,
#             "target_rot": None,
#             "final_pos": None,
#             "final_rot": None,
#             "used_force": False
#         }

#     if anchor_meta is None:
#         return False, {
#             "reason": "Anchor object not found",
#             "original_pos": _pos_dict(obj_meta["position"]),
#             "original_rot": obj_meta["rotation"],
#             "target_pos": None,
#             "target_rot": None,
#             "final_pos": None,
#             "final_rot": None,
#             "used_force": False
#         }

#     # 记录原始位置和旋转
#     original_pos = _pos_dict(obj_meta["position"])
#     original_rot = obj_meta["rotation"]

#     # 获取锚点位置
#     base = _pos_dict(anchor_meta["position"])

#     # === Step 2. 计算目标位置 ===
#     # 根据相机坐标系计算水平 forward/right 向量
#     right, forward = get_horizontal_axes(controller.last_event.metadata["agent"])

#     # 选择偏移方向（相对相机）
#     if "right" in relation:
#         dir_vec = right
#     elif "left" in relation:
#         dir_vec = {"x": -right["x"], "y": 0.0, "z": -right["z"]}
#     elif "front" in relation:
#         dir_vec = forward
#     elif "behind" in relation:
#         dir_vec = {"x": -forward["x"], "y": 0.0, "z": -forward["z"]}
#     else:
#         return False, {
#             "reason": f"Unknown relation: {relation}",
#             "original_pos": original_pos,
#             "original_rot": original_rot,
#             "target_pos": None,
#             "target_rot": None,
#             "final_pos": None,
#             "final_rot": None,
#             "used_force": False
#         }

#     # 基于相机方向生成目标点
#     target = {
#         "x": base["x"] + dir_vec["x"] * dist_m,
#         "y": base["y"],  # 使用锚点的 y 坐标
#         "z": base["z"] + dir_vec["z"] * dist_m,
#     }

#     target_rot = original_rot  # 目标旋转与原始旋转相同

#     # === Step 3. 尝试传送物体 ===
#     evt = controller.step(
#         action="TeleportObject",
#         objectId=obj_id,
#         position=target,
#         rotation=original_rot,
#         forceAction=False
#     )
#     used_force = False
#     used_pos = target

#     if not evt.metadata.get("lastActionSuccess", False):
#         # 若失败则尝试抬高 + forceAction
#         bumped = dict(target)
#         bumped["y"] += 0.01
#         evt = controller.step(
#             action="TeleportObject",
#             objectId=obj_id,
#             position=bumped,
#             rotation=original_rot,
#             forceAction=True
#         )
#         used_force = True
#         used_pos = bumped

#         if not evt.metadata.get("lastActionSuccess", False):
#             return False, {
#                 "reason": evt.metadata.get("errorMessage", "teleport_failed"),
#                 "original_pos": original_pos,
#                 "original_rot": original_rot,
#                 "target_pos": target,
#                 "target_rot": target_rot,
#                 "final_pos": None,
#                 "final_rot": None,
#                 "used_force": used_force
#             }

#     # === Step 4. 验证位置与旋转 ===
#     after_meta = get_object_by_id(controller, obj_id)
#     if after_meta is None:
#         return False, {
#             "reason": "teleport_lost_object",
#             "original_pos": original_pos,
#             "original_rot": original_rot,
#             "target_pos": target,
#             "target_rot": target_rot,
#             "final_pos": used_pos,
#             "final_rot": None,
#             "used_force": used_force
#         }

#     final_pos = _pos_dict(after_meta["position"])
#     final_rot = after_meta["rotation"]

#     # 检查位置偏差
#     dist_err = ((final_pos["x"] - used_pos["x"])**2 +
#                 (final_pos["y"] - used_pos["y"])**2 +
#                 (final_pos["z"] - used_pos["z"])**2) ** 0.5

#     # 检查旋转偏差
#     def compute_rotation_error(final_rot, original_rot):
#         def angle_diff(a, b):
#             diff = abs(a - b) % 360
#             return min(diff, 360 - diff)

#         rot_x = angle_diff(final_rot["x"], original_rot["x"])
#         rot_y = angle_diff(final_rot["y"], original_rot["y"])
#         rot_z = angle_diff(final_rot["z"], original_rot["z"])

#         tilt_err = max(abs(final_rot["x"]), abs(final_rot["z"]))
#         total_err = rot_x + rot_y + rot_z
#         return total_err, tilt_err

#     rot_err, tilt_err = compute_rotation_error(final_rot, original_rot)

#     if dist_err > 0.05:
#         reason = f"teleport_mismatch_pos ({dist_err:.3f}m)"
#         return False, {
#             "reason": reason,
#             "original_pos": original_pos,
#             "original_rot": original_rot,
#             "target_pos": target,
#             "target_rot": target_rot,
#             "final_pos": final_pos,
#             "final_rot": final_rot,
#             "used_force": used_force
#         }

#     if rot_err > 5.0:
#         reason = f"teleport_mismatch_rot ({rot_err:.1f} deg)"
#         return False, {
#             "reason": reason,
#             "original_pos": original_pos,
#             "original_rot": original_rot,
#             "target_pos": target,
#             "target_rot": target_rot,
#             "final_pos": final_pos,
#             "final_rot": final_rot,
#             "used_force": used_force
#         }

#     # === Step 5. 最终返回结果 ===
#     if used_force:
#         reason = f"teleport_verified_force (err={dist_err:.3f}m)"
#     else:
#         reason = f"teleport_verified (err={dist_err:.3f}m)"

#     return True, {
#         "reason": reason,
#         "original_pos": original_pos,
#         "original_rot": original_rot,
#         "target_pos": target,
#         "target_rot": target_rot,
#         "final_pos": final_pos,
#         "final_rot": final_rot,
#         "used_force": used_force
#     }


def execute_object_relative_command(
        controller,
        obj_id,
        anchor_id,
        relation="to the right of",
        dist_m=0.1
):
    """
    将物体移动到相对于锚点物体的位置（基于物体中心点距离）。

    Args:
        controller: AI2-THOR controller
        obj_id: 要移动的物体ID
        anchor_id: 锚点物体ID
        relation: 相对关系 ("to the right of", "to the left of", "in front of", "behind")
        dist_m: 物体中心之间的距离（米）

    Returns:
        success (bool), result_dict
    """

    # === Step 1. 获取物体元数据 ===
    obj_meta = get_object_by_id(controller, obj_id)
    anchor_meta = get_object_by_id(controller, anchor_id)

    def _fail(reason, **extra):
        return False, {
            "reason": reason,
            **extra,
            "used_force": extra.get("used_force", False)
        }

    if obj_meta is None:
        return _fail("Object not found")

    if anchor_meta is None:
        return _fail("Anchor object not found", original_pos=_pos_dict(obj_meta["position"]),
                     original_rot=obj_meta["rotation"])

    original_pos = _pos_dict(obj_meta["position"])
    original_rot = obj_meta["rotation"]
    base = _pos_dict(anchor_meta["position"])

    # === Step 2. 计算方向向量 ===
    right, forward = get_horizontal_axes(controller.last_event.metadata["agent"])

    if "right" in relation:
        dir_vec = right
    elif "left" in relation:
        dir_vec = {"x": -right["x"], "y": 0.0, "z": -right["z"]}
    elif "front" in relation:
        dir_vec = {"x": -forward["x"], "y": 0.0, "z": -forward["z"]}
    elif "behind" in relation:
        dir_vec = forward
    else:
        return _fail(f"Unknown relation: {relation}", original_pos=original_pos, original_rot=original_rot)

    # === Step 3. 计算目标位置（只考虑中心点距离） ===
    target = {
        "x": base["x"] + dir_vec["x"] * dist_m,
        "y": base["y"],  # 保持同一高度
        "z": base["z"] + dir_vec["z"] * dist_m,
    }
    target_rot = original_rot

    # === Step 5. Teleport 尝试 ===
    evt = controller.step(
        action="TeleportObject",
        objectId=obj_id,
        position=target,
        rotation=original_rot,
        forceAction=False
    )

    used_force = False
    used_pos = target

    if not evt.metadata.get("lastActionSuccess", False):
        bumped = dict(target)
        bumped["y"] += 0.02  # 稍微抬高一点
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
                         original_pos=original_pos, original_rot=original_rot,
                         target_pos=target, target_rot=target_rot)

    # === Step 6. 验证最终位置并检查可见性 ===
    after_meta = get_object_by_id(controller, obj_id)
    if after_meta is None:
        return _fail("teleport_lost_object",
                     original_pos=original_pos, original_rot=original_rot,
                     target_pos=target, target_rot=target_rot,
                     final_pos=used_pos)

    # 检查物体是否可见
    visible = after_meta.get("visible", False)
    if not visible:
        return _fail("object_not_visible_after_teleport",
                     original_pos=original_pos, original_rot=original_rot,
                     target_pos=target, target_rot=target_rot,
                     final_pos=used_pos, final_rot=original_rot)

    final_pos = _pos_dict(after_meta["position"])
    final_rot = after_meta["rotation"]

    # 距离偏差
    dist_err = ((final_pos["x"] - used_pos["x"]) ** 2 +
                (final_pos["y"] - used_pos["y"]) ** 2 +
                (final_pos["z"] - used_pos["z"]) ** 2) ** 0.5

    # 旋转偏差
    def compute_rotation_error(final_rot, original_rot):
        def angle_diff(a, b):
            diff = abs(a - b) % 360
            return min(diff, 360 - diff)
        rot_x = angle_diff(final_rot["x"], original_rot["x"])
        rot_y = angle_diff(final_rot["y"], original_rot["y"])
        rot_z = angle_diff(final_rot["z"], original_rot["z"])
        tilt_err = max(abs(final_rot["x"]), abs(final_rot["z"]))
        return rot_x + rot_y + rot_z, tilt_err

    rot_err, tilt_err = compute_rotation_error(final_rot, original_rot)

    if dist_err > 0.05:
        return _fail(f"teleport_mismatch_pos ({dist_err:.3f}m)",
                     original_pos=original_pos, original_rot=original_rot,
                     target_pos=target, target_rot=target_rot,
                     final_pos=final_pos, final_rot=final_rot)

    if rot_err > 5.0:
        return _fail(f"teleport_mismatch_rot ({rot_err:.1f} deg)",
                     original_pos=original_pos, original_rot=original_rot,
                     target_pos=target, target_rot=target_rot,
                     final_pos=final_pos, final_rot=final_rot)

    # === Step 8. 成功返回 ===
    reason = f"teleport_verified{'_force' if used_force else ''} (err={dist_err:.3f}m)"
    return True, {
        "reason": reason,
        "original_pos": original_pos,
        "original_rot": original_rot,
        "target_pos": target,
        "target_rot": target_rot,
        "final_pos": final_pos,
        "final_rot": final_rot,
        "used_force": used_force,
        "visible": visible
    }
