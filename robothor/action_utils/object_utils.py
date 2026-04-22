"""
Object-related utility functions for RoboTHOR environment
"""
import random
from collections import Counter


def get_visible_interactables(controller, pickup_only=False, max_objects=5):
    """
    获取当前视野中可交互的、顶层（无上层覆盖）的物体。

    Args:
        controller: AI2-THOR controller
        pickup_only: 是否只获取可拾取的物体
        max_objects: 最大返回物体数量

    Returns:
        符合条件的物体列表（随机抽样）
    """
    objs = controller.last_event.metadata["objects"]
    visible_objs = [o for o in objs if o.get("visible") and not o.get("isPickedUp")]

    # === 歧义检测阶段 ===
    type_counts = Counter(o["objectType"] for o in visible_objs)
    ambiguous_types = {t for t, c in type_counts.items() if c > 1}
    for o in visible_objs:
        o["is_ambiguous"] = (o["objectType"] in ambiguous_types)

    # 找出所有"被放置在其它物体上"的 receptacle ID
    occupied_receptacles = set()
    for o in visible_objs:
        parents = o.get("parentReceptacles") or []
        for pid in parents:
            occupied_receptacles.add(pid)

    # 过滤：仅保留没有其它物体放在自己上面的
    candidates = []
    for o in visible_objs:
        if o["objectId"] in occupied_receptacles:
            continue  # 被覆盖的物体跳过
        if pickup_only and not o.get("pickupable", False):
            continue
        if (o.get("pickupable") or o.get("moveable")):
            candidates.append(o)

    # 过滤:去除有歧义的物体
    candidates = [o for o in candidates if not o.get("is_ambiguous", False)]

    # 随机截取数量
    random.shuffle(candidates)
    return candidates[:max_objects]


def _pick_nearest_visible_receptacle(controller, target_pos):
    """
    从可见对象里挑选receptacle=True的，选离target_pos最近的

    Args:
        controller: AI2-THOR controller
        target_pos: 目标位置字典 {"x": float, "y": float, "z": float}

    Returns:
        最近的receptacle物体，如果没有则返回None
    """
    from robothor_utils import _pos_dict, _dist3

    cands = []
    for o in controller.last_event.metadata.get("objects", []):
        if not o.get("visible"):
            continue
        if o.get("receptacle", False):
            cands.append(o)
    if not cands:
        return None
    return min(cands, key=lambda o: _dist3(_pos_dict(o["position"]), target_pos))


def reset_object_position(controller, obj_id, original_pos, original_rot, view_info=None, disable_physics=False):
    """
    尝试重置物体位置，优先使用TeleportObject（快速），失败时才使用完整环境重置

    Args:
        controller: AI2-THOR controller
        obj_id: 物体ID
        original_pos: 原始位置
        original_rot: 原始旋转
        view_info: 视角信息，用于环境重置后恢复视角
        disable_physics: 是否禁用物理模拟

    Returns:
        success: 是否成功重置
        method: 使用的方法 ("teleport" 或 "full_reset")
    """
    # 方法1: 尝试使用TeleportObject（快速）
    controller.step(
        action="TeleportObject",
        objectId=obj_id,
        position=original_pos,
        rotation=original_rot
    )

    if controller.last_event.metadata.get("lastActionSuccess", False):
        return True, "teleport"

    # 方法2: 尝试稍微提高Y坐标
    new_pos = dict(original_pos)
    new_pos["y"] += 0.02
    controller.step(
        action="TeleportObject",
        objectId=obj_id,
        position=new_pos,
        rotation=original_rot
    )

    if controller.last_event.metadata.get("lastActionSuccess", False):
        return True, "teleport_raised"

    # 方法3: 尝试使用forceAction
    controller.step(
        action="TeleportObject",
        objectId=obj_id,
        position=original_pos,
        rotation=original_rot,
        forceAction=True
    )

    if controller.last_event.metadata.get("lastActionSuccess", False):
        return True, "teleport_forced"

    # 方法4: 最后手段 - 完整环境重置（慢）
    print(f"[Warning] TeleportObject failed for {obj_id}, using full environment reset")
    controller.reset()

    # 导入并使用环境工具函数
    from action_utils.environment_utils import apply_physics_settings, restore_view

    # 重新应用物理设置（reset 会恢复物理）
    apply_physics_settings(controller, disable_physics)

    # 恢复到原来的视角
    if view_info is not None:
        restore_view(controller, view_info)

    return True, "full_reset"
