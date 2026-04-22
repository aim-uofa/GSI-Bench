# camera_relative.py
import random
from .spawn_utils import (
    _pos_dict, _dist3, get_object_by_id, get_first_receptacle_id_of, get_closest_receptacle_id,
    get_spawn_coords_on_receptacle, pick_nearest_coord, place_object_at_coord,
    pick_nearest_visible_receptacle, get_camera_axes, _dist2, get_horizontal_axes
)
from .utils_record import save_current_view
def estimate_max_move_distance_from_camera(
        controller, obj_meta,
        directions=("left", "right", "forward", "back"),
        test_distances=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
        dist_threshold=0.15
):
    results = {}
    obj_pos = _pos_dict(obj_meta["position"])
    rec_id = get_first_receptacle_id_of(obj_meta)
    if rec_id is None:
        return results

    # 获取物体所在 receptacle 表面的 spawn 点
    coords = get_spawn_coords_on_receptacle(controller, rec_id, anywhere=False)
    if not coords:
        coords = get_spawn_coords_on_receptacle(controller, rec_id, anywhere=True)
    if not coords:
        return results

    # ✅ 关键修改：用 receptacle 的 surface_y，而不是物体本身 y
    ys = sorted(c["y"] for c in coords)
    surface_y = ys[len(ys)//2]   # 中位数，最稳妥，避免奇异点

    # right, forward, up, _ = get_camera_axes(controller)
    right, forward = get_horizontal_axes(controller)
    

    for d in directions:
        best_valid_dist = 0.0
        for dist in test_distances:
            if d == "left":
                target = {
                    "x": obj_pos["x"] - right["x"]*dist,
                    "y": surface_y,
                    "z": obj_pos["z"] - right["z"]*dist
                }
            elif d == "right":
                target = {
                    "x": obj_pos["x"] + right["x"]*dist,
                    "y": surface_y,
                    "z": obj_pos["z"] + right["z"]*dist
                }
            elif d == "forward":
                target = {
                    "x": obj_pos["x"] + forward["x"]*dist,
                    "y": surface_y,
                    "z": obj_pos["z"] + forward["z"]*dist
                }
            elif d == "back":
                target = {
                    "x": obj_pos["x"] - forward["x"]*dist,
                    "y": surface_y,
                    "z": obj_pos["z"] - forward["z"]*dist
                }
            else:
                continue

            nearest = pick_nearest_coord(coords, target)
            if nearest is None:
                break

            if _dist3(nearest, target) < dist_threshold:
                best_valid_dist = dist
            else:
                break

        if best_valid_dist > 0:
            results[d] = best_valid_dist
    # save_current_view(controller, f"debug")
    return results



def generate_camera_relative_commands(
        obj_meta,
        controller,
        sample_distances=(0.10, 0.15, 0.20),
        test_directions=("left","right","forward","back"),
        test_distances=(0.05,0.10,0.15,0.20,0.25,0.30)
):
    obj_type = obj_meta["objectType"]

    max_ranges = estimate_max_move_distance_from_camera(
        controller, obj_meta,
        directions=test_directions,
        test_distances=test_distances
    )

    commands = []
    for d, max_d in max_ranges.items():
        for dist in sample_distances:
            if dist <= max_d:
                text = f"Move the {obj_type} {int(dist*100)} centimeters {d} relative to the camera view."
                commands.append({
                    "instruction": text,
                    "direction": d,
                    "dist_m": dist
                })
    return commands


# def execute_camera_relative_command(controller, obj_id, direction, dist_m):
#     # use place object at coord
#     obj_meta = get_object_by_id(controller, obj_id)
#     if obj_meta is None:
#         return False, "Object not found", None

#     # right, forward, up, _ = get_camera_axes(controller)
#     right, forward = get_horizontal_axes(controller)
#     base_pos = _pos_dict(obj_meta["position"])

#     if direction == "left":
#         target = {"x": base_pos["x"] - right["x"]*dist_m, "y": base_pos["y"], "z": base_pos["z"] - right["z"]*dist_m}
#     elif direction == "right":
#         target = {"x": base_pos["x"] + right["x"]*dist_m, "y": base_pos["y"], "z": base_pos["z"] + right["z"]*dist_m}
#     elif direction == "forward":
#         target = {"x": base_pos["x"] + forward["x"]*dist_m, "y": base_pos["y"], "z": base_pos["z"] + forward["z"]*dist_m}
#     elif direction == "back":
#         target = {"x": base_pos["x"] - forward["x"]*dist_m, "y": base_pos["y"], "z": base_pos["z"] - forward["z"]*dist_m}
#     else:
#         return False, f"Unknown direction: {direction}", None

#     # rec_id = get_first_receptacle_id_of(obj_meta) # 第一个 receptacle 未必是直接接触的
#     rec_id = get_closest_receptacle_id(controller, obj_meta)
#     if rec_id is None:
#         nearest_rec = pick_nearest_visible_receptacle(controller, target)
#         if nearest_rec is None:
#             return False, "No receptacle found", None
#         rec_id = nearest_rec["objectId"]

#     coords = get_spawn_coords_on_receptacle(controller, rec_id, anywhere=False)
#     if not coords:
#         coords = get_spawn_coords_on_receptacle(controller, rec_id, anywhere=True)
#     if not coords:
#         return False, "No spawn coords", None

#     best_coord = pick_nearest_coord(coords, target)
#     if best_coord is None:
#         return False, "No valid coordinate", None
    
#     # 检查一下和原始 target 的距离 不能过大
#     if _dist2(best_coord, target) > 0.05:
#         return False, "No suitable coordinate near target", None

#     ok, reason, used = place_object_at_coord(controller, obj_id, best_coord)
#     # save_current_view(controller, f"debug_place_{obj_id}")
#     return ok, reason, used


def execute_camera_relative_command(controller, obj_id, direction, dist_m):
    """
    移动物体相对于相机视角的平面方向位置，使用 TeleportObject 直接定位。
    优点：保持原始旋转角度，逻辑简单。
    缺点：若目标位置与其他物体重叠可能失败。

    返回值：
        success (bool): 是否成功
        result_dict (dict): 包含以下字段的字典
            - reason (str): 操作结果原因
            - original_pos (dict): 原始位置 {x, y, z}
            - original_rot (dict): 原始旋转 {x, y, z}
            - target_pos (dict): 目标位置 {x, y, z}
            - target_rot (dict): 目标旋转 {x, y, z}（与原始旋转相同）
            - final_pos (dict): 最终位置 {x, y, z}
            - final_rot (dict): 最终旋转 {x, y, z}
            - used_force (bool): 是否使用了强制移动
    """
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

    # 计算方向向量（只在水平面上）
    right, forward = get_horizontal_axes(controller)

    # 计算目标位置
    if direction == "left":
        target = {
            "x": original_pos["x"] - right["x"] * dist_m,
            "y": original_pos["y"],
            "z": original_pos["z"] - right["z"] * dist_m
        }
    elif direction == "right":
        target = {
            "x": original_pos["x"] + right["x"] * dist_m,
            "y": original_pos["y"],
            "z": original_pos["z"] + right["z"] * dist_m
        }
    elif direction == "forward":
        target = {
            "x": original_pos["x"] + forward["x"] * dist_m,
            "y": original_pos["y"],
            "z": original_pos["z"] + forward["z"] * dist_m
        }
    elif direction == "back":
        target = {
            "x": original_pos["x"] - forward["x"] * dist_m,
            "y": original_pos["y"],
            "z": original_pos["z"] - forward["z"] * dist_m
        }
    else:
        return False, {
            "reason": f"Unknown direction: {direction}",
            "original_pos": original_pos,
            "original_rot": original_rot,
            "target_pos": None,
            "target_rot": None,
            "final_pos": None,
            "final_rot": None,
            "used_force": False
        }

    target_rot = original_rot  # 目标旋转与原始旋转相同

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
        # 若失败则尝试抬高 + forceAction
        bumped = dict(target)
        bumped["y"] += 0.01
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
            return False, {
                "reason": evt.metadata.get("errorMessage", "teleport_failed"),
                "original_pos": original_pos,
                "original_rot": original_rot,
                "target_pos": target,
                "target_rot": target_rot,
                "final_pos": None,
                "final_rot": None,
                "used_force": used_force
            }

    # === Step 4. 验证位置与旋转 ===
    after_meta = get_object_by_id(controller, obj_id)
    if after_meta is None:
        return False, {
            "reason": "teleport_lost_object",
            "original_pos": original_pos,
            "original_rot": original_rot,
            "target_pos": target,
            "target_rot": target_rot,
            "final_pos": used_pos,
            "final_rot": None,
            "used_force": used_force
        }

    final_pos = _pos_dict(after_meta["position"])
    final_rot = after_meta["rotation"]

    # 检查位置偏差
    dist_err = ((final_pos["x"] - used_pos["x"])**2 +
                (final_pos["y"] - used_pos["y"])**2 +
                (final_pos["z"] - used_pos["z"])**2) ** 0.5

    # 检查旋转偏差
    # rot_err = abs(final_rot["x"] - original_rot["x"]) + abs(final_rot["z"] - original_rot["z"])
    def compute_rotation_error(final_rot, original_rot):
        def angle_diff(a, b):
            diff = abs(a - b) % 360
            return min(diff, 360 - diff)

        rot_x = angle_diff(final_rot["x"], original_rot["x"])
        rot_y = angle_diff(final_rot["y"], original_rot["y"])
        rot_z = angle_diff(final_rot["z"], original_rot["z"])

        tilt_err = max(abs(final_rot["x"]), abs(final_rot["z"]))
        total_err = rot_x + rot_y + rot_z
        return total_err, tilt_err
    rot_err, tilt_err = compute_rotation_error(final_rot, original_rot)
    if dist_err > 0.05:
        reason = f"teleport_mismatch_pos ({dist_err:.3f}m)"
        return False, {
            "reason": reason,
            "original_pos": original_pos,
            "original_rot": original_rot,
            "target_pos": target,
            "target_rot": target_rot,
            "final_pos": final_pos,
            "final_rot": final_rot,
            "used_force": used_force
        }

    if rot_err > 5.0:
        reason = f"teleport_mismatch_rot ({rot_err:.1f} deg)"
        return False, {
            "reason": reason,
            "original_pos": original_pos,
            "original_rot": original_rot,
            "target_pos": target,
            "target_rot": target_rot,
            "final_pos": final_pos,
            "final_rot": final_rot,
            "used_force": used_force
        }

    # === Step 5. 最终返回结果 ===
    if used_force:
        reason = f"teleport_verified_force (err={dist_err:.3f}m)"
    else:
        reason = f"teleport_verified (err={dist_err:.3f}m)"

    return True, {
        "reason": reason,
        "original_pos": original_pos,
        "original_rot": original_rot,
        "target_pos": target,
        "target_rot": target_rot,
        "final_pos": final_pos,
        "final_rot": final_rot,
        "used_force": used_force
    }
