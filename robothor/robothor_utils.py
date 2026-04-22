#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RoboTHOR utility functions.
General-purpose helper functions for AI2-THOR/RoboTHOR environments.
Author: zmz @ Zhejiang University
"""
import os
import math
import numpy as np
from PIL import Image


# =====================
# Rendering & Visualization
# =====================

def save_current_view(controller, output_dir="./output", prefix="frame"):
    """保存当前AI2-THOR的视图（RGB、深度、分割）"""
    os.makedirs(output_dir, exist_ok=True)
    event = controller.last_event

    rgb = Image.fromarray(event.frame)
    depth = event.depth_frame
    seg = event.instance_segmentation_frame

    rgb.save(f"{output_dir}/{prefix}_rgb.png")
    Image.fromarray((depth / np.max(depth) * 255).astype(np.uint8)).save(f"{output_dir}/{prefix}_depth.png")
    Image.fromarray(seg).save(f"{output_dir}/{prefix}_seg.png")

    print(f"[Info] Saved frames to {output_dir}/{prefix}_*.png")


def get_top_down_frame(controller) -> Image.Image:
    """获取场景的俯视图"""
    import copy
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    # 更新相机参数
    pose.update({
        "fieldOfView": 50,
        "position": {
            "x": pose["position"]["x"],
            "y": pose["position"]["y"] + 1.1 * max_bound,
            "z": pose["position"]["z"]
        },
        "orthographic": False,
        "farClippingPlane": 50,
    })

    # 删除 orthographicSize 参数，防止冲突
    pose.pop("orthographicSize", None)

    # 添加第三方相机
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True
    )

    return Image.fromarray(event.third_party_camera_frames[-1])


def set_render_quality(controller, quality="Ultra", width=1024, height=1024):
    """设置渲染质量与分辨率（自动兼容旧版本）"""
    try:
        controller.step(action="ChangeQuality", quality=quality)
    except Exception:
        controller.step(action="SetQuality", quality=quality)

    try:
        controller.step(action="SetResolution", width=width, height=height)
    except Exception:
        controller.step(action="ChangeResolution", x=width, y=height)

    print(f"[Info] Rendering upgraded: quality={quality}, resolution={width}x{height}")


# =====================
# Basic Math Utilities
# =====================

def _dist3(a, b):
    """计算两个3D点之间的欧氏距离"""
    return math.sqrt((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2 + (a["z"]-b["z"])**2)


def _angle_diff(a, b):
    """计算两个角度之间的最小差值（0-180度）"""
    d = abs((a - b) % 360)
    return min(d, 360 - d)


def _pos_dict(p):
    """确保位置是标准的dict格式，包含x,y,z"""
    return {"x": float(p["x"]), "y": float(p["y"]), "z": float(p["z"])}


# =====================
# Object Manipulation
# =====================

def reposition_object_on_same_receptacle(controller, object_id):
    """
    保证物体仍放在同一个容器上，但位置更新到容器表面的一个新可见点。
    """
    # Step 1: 找到目标物体
    objects = controller.last_event.metadata["objects"]
    target = next((o for o in objects if o["objectId"] == object_id), None)
    if not target:
        raise ValueError(f"Object {object_id} not found in metadata.")

    receptacle_id = target.get("parentReceptacles")
    if not receptacle_id:
        raise ValueError(f"Object {object_id} is not currently on any receptacle.")

    # Step 2: 获取容器上方可视范围内的放置点
    coords_event = controller.step(
        action="GetSpawnCoordinatesAboveReceptacle",
        objectId=receptacle_id[0],
        anywhere=False,  # 只取视野中可见位置
        raise_for_failure=True,
    )

    coords = coords_event.metadata["actionReturn"]
    if not coords:
        raise RuntimeError("No valid spawn coordinates found (object may be out of view).")

    # Step 3: 选择第一个坐标并重新放置物体
    new_position = coords[0]
    result = controller.step(
        action="PlaceObjectAtPoint",
        objectId=object_id,
        position=new_position,
        raise_for_failure=True,
    )

    if not result.metadata["lastActionSuccess"]:
        raise RuntimeError(f"Failed to reposition object: {result.metadata.get('errorMessage')}")

    return result


def visualize_reposition_on_receptacle(controller, output_dir="./output", prefix="reposition_demo"):
    """触发 reposition_object_on_same_receptacle 并保存前后帧用于可视化"""
    event = controller.last_event
    if event is None:
        raise RuntimeError("Controller has no event; perform a step before visualization.")

    metadata_objects = event.metadata.get("objects", [])
    candidates = [
        obj for obj in metadata_objects
        if obj.get("visible")
        and (obj.get("pickupable") or obj.get("moveable"))
        and obj.get("parentReceptacles")
    ]
    if not candidates:
        raise RuntimeError("No visible movable object on a receptacle found for visualization.")

    candidate = candidates[0]
    print("Found candidates:", [obj["objectId"] for obj in candidates])

    object_id = candidate["objectId"]
    before_position = candidate["position"].copy()

    os.makedirs(output_dir, exist_ok=True)
    save_current_view(controller, output_dir, prefix=f"{prefix}_before")

    reposition_object_on_same_receptacle(controller, object_id=object_id)

    save_current_view(controller, output_dir, prefix=f"{prefix}_after")

    after_event = controller.last_event
    moved_metadata = None
    for obj in after_event.metadata.get("objects", []):
        if obj.get("objectId") == object_id:
            moved_metadata = obj
            break

    after_position = moved_metadata["position"] if moved_metadata else None
    print(f"[Info] Reposition demo moved {object_id} from {before_position} to {after_position}.")

    return object_id, before_position, after_position


# =====================
# Room Clustering & View Selection
# =====================

def _nms_views(candidates, min_pos_dist=0.6, min_rot_diff=45):
    """对视角做简单NMS，避免位置太近&角度太接近的重复视角"""
    selected = []
    for cand in sorted(candidates, key=lambda x: (-x["score"])):  # 高分优先
        keep = True
        for s in selected:
            if _dist3(cand["pos"], s["pos"]) < min_pos_dist and _angle_diff(cand["rot"], s["rot"]) < min_rot_diff:
                keep = False
                break
        if keep:
            selected.append(cand)
    return selected


def cluster_positions_greedy(positions, eps=1.5, min_members=15, max_cluster_size=99999):
    """
    简易"在线聚类"：按位置顺序遍历，靠近已有簇中心（<eps）就并入并更新中心；否则新开簇。
    - eps：房间半径阈值（越小越容易分成多个房间）
    - min_members：过滤掉太小的簇（走廊、门缝噪声）
    注意：RoboTHOR 多数是一个公寓单元，此方法能把房间/区域粗分出来即可。
    """
    clusters = []  # 每个元素：{"center":pos, "members":[pos,...]}
    for p in positions:
        p = _pos_dict(p)
        best_i, best_d = -1, 1e9
        for i, c in enumerate(clusters):
            d = _dist3(p, c["center"])
            if d < best_d:
                best_d, best_i = d, i
        if best_d <= eps and len(clusters[best_i]["members"]) < max_cluster_size:
            # 并入并更新中心（简单均值更新）
            c = clusters[best_i]
            c["members"].append(p)
            n = len(c["members"])
            cx, cy, cz = c["center"]["x"], c["center"]["y"], c["center"]["z"]
            cx = (cx*(n-1) + p["x"])/n
            cy = (cy*(n-1) + p["y"])/n
            cz = (cz*(n-1) + p["z"])/n
            c["center"] = {"x":cx, "y":cy, "z":cz}
        else:
            clusters.append({"center": p, "members": [p]})

    # 过滤小簇
    clusters = [c for c in clusters if len(c["members"]) >= min_members]
    # 按成员数降序
    clusters.sort(key=lambda c: -len(c["members"]))
    return clusters


def count_visible_interactables(controller, require_pickupable=False):
    """
    统计当前视野内 (visible==True) 且 可交互 的物体数：
    - require_pickupable=True 时，只计 pickupable 物体
    - 否则：pickupable 或 moveable 任一为 True
    返回：(count, 可见物体objectId集合) 方便后续调试
    """
    objs = controller.last_event.metadata.get("objects", [])
    ids = []
    for o in objs:
        if not o.get("visible"):
            continue
        if require_pickupable:
            ok = o.get("pickupable", False)
        else:
            ok = o.get("pickupable", False) or o.get("moveable", False)
        if ok:
            ids.append(o["objectId"])
    return len(ids), set(ids)


def find_top_k_views_in_cluster(
    controller,
    cluster_members,
    k=5,
    rotations=(0, 90, 180, 270, 45, 135, 225, 315),  # 水平旋转角
    horizons=(0, 15, 30),                           # 俯视角度（pitch）
    sample_positions=60,
    require_pickupable=False,
    min_pos_dist=0.6,
    min_rot_diff=45
):
    """
    在给定房间 cluster_members 内，搜索视角（位置 + rotation + pitch），
    选出"可见可交互物体最多"的 top-k 视角。

    支持 horizon（俯视摄像机角度）
    自动评分并做 NMS 去除过近重复视角
    """
    import random

    if len(cluster_members) == 0:
        return []

    # 抽样位置减少计算
    members = list(cluster_members)
    random.shuffle(members)
    members = members[:sample_positions]

    candidates = []

    for pos in members:
        pos = _pos_dict(pos)  # 确保是 {x,y,z} 格式
        for rot in rotations:
            for hor in horizons:
                event = controller.step(
                    action="TeleportFull",
                    position=pos,
                    rotation={"x": 0, "y": rot, "z": 0},
                    horizon=hor,
                    standing=True,
                    forceAction=True
                )

                if not event.metadata.get("lastActionSuccess", False):
                    continue

                score, _ = count_visible_interactables(
                    controller,
                    require_pickupable=require_pickupable
                )
                if score <= 0:
                    continue

                candidates.append({
                    "pos": pos,
                    "rot": int(rot),
                    "horizon": int(hor),
                    "score": int(score),
                })

    if not candidates:
        return []

    # 使用 NMS 去除太近/角度重复的视角
    filtered = _nms_views(
        candidates,
        min_pos_dist=min_pos_dist,
        min_rot_diff=min_rot_diff
    )

    # 输出前 k 个视角
    return filtered[:k]


def generate_room_views(controller,
                        k_per_room=5,
                        eps=1.6,
                        min_members=15,
                        sample_positions_per_room=80,
                        require_pickupable=False):
    """
    主函数：
    1) 获取可达点
    2) 聚类成"房间/区域"
    3) 每个房间找 k 个最佳视角
    4) 返回 views 列表（含 room_id / score / pos / rot）
    """
    # 1) 所有可达位置
    evt = controller.step(action="GetReachablePositions")
    reachable_positions = evt.metadata["actionReturn"]
    if not reachable_positions:
        print("[Warn] No reachable positions in this scene.")
        return []

    # 2) 聚类
    clusters = cluster_positions_greedy(reachable_positions, eps=eps, min_members=min_members)
    if not clusters:
        print("[Warn] Clustering produced no rooms; using whole space as one room.")
        clusters = [{"center": {"x": 0., "y": 0., "z": 0.}, "members": [ _pos_dict(p) for p in reachable_positions ]}]

    print(f"[Info] Detected {len(clusters)} room(s) by clustering.")
    views = []
    for rid, c in enumerate(clusters):
        best = find_top_k_views_in_cluster(
            controller,
            c["members"],
            k=k_per_room,
            rotations=(0, 90, 180, 270, 45, 135, 225, 315),
            sample_positions=sample_positions_per_room,
            require_pickupable=require_pickupable,
            min_pos_dist=0.6,
            min_rot_diff=45
        )
        # 标注 room_id
        for v in best:
            v["room_id"] = rid
        print(f"[Info] Room {rid}: selected {len(best)} view(s).")
        views.extend(best)

    # 可选：全局再做一次轻量NMS（避免跨房间边界过近的重复）
    # 这里假设每个房间保留自己的5个即可，不做全局NMS。
    return views
