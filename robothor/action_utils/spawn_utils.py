# spawn_utils.py
import math
import random
import os 
def _pos_dict(pos):
    return {"x": float(pos["x"]), "y": float(pos["y"]), "z": float(pos["z"])}

def _dist3(a, b):
    return math.sqrt((a["x"] - b["x"])**2 + (a["y"] - b["y"])**2 + (a["z"] - b["z"])**2)

def _dist2(a, b):
    return math.sqrt((a["x"] - b["x"])**2 + (a["z"] - b["z"])**2)

def _norm2(v):
    return math.sqrt(v["x"] ** 2 + v["z"] ** 2)

def _dot2(a, b):
    return a["x"] * b["x"] + a["z"] * b["z"]

def get_object_by_id(controller, object_id):
    for obj in controller.last_event.metadata.get("objects", []):
        if obj["objectId"] == object_id:
            return obj
    return None

def get_first_receptacle_id_of(obj_meta):
    recs = obj_meta.get("parentReceptacles")
    if recs and len(recs) > 0:
        return recs[0]
    return None


def get_closest_receptacle_id(controller, obj_meta, max_height_diff=0.5):
    recs = obj_meta.get("parentReceptacles") or []
    if not recs:
        return None

    obj_y = obj_meta["position"]["y"]
    valid = []
    for r_id in recs:
        rec = next((o for o in controller.last_event.metadata["objects"]
                    if o["objectId"] == r_id), None)
        if rec:
            y_diff = abs(rec["position"]["y"] - obj_y)
            valid.append((y_diff, r_id))
    if not valid:
        return None

    valid.sort(key=lambda x: x[0])  # 按高度差排序
    best_y_diff, best_rid = valid[0]
    if best_y_diff < max_height_diff:
        return best_rid
    return None

def get_camera_axes(controller):
    agent = controller.last_event.metadata["agent"]
    yaw = math.radians(agent["rotation"]["y"])
    pitch = math.radians(agent.get("cameraHorizon", 0.0))

    forward = {
        "x": math.sin(yaw) * math.cos(pitch),
        "y": -math.sin(pitch),
        "z": math.cos(yaw) * math.cos(pitch),
    }
    right = {"x": math.cos(yaw), "y": 0.0, "z": -math.sin(yaw)}
    up = {"x": 0.0, "y": 1.0, "z": 0.0}

    def norm(v):
        n = math.sqrt(v["x"]**2 + v["y"]**2 + v["z"]**2) or 1.0
        return {"x": v["x"]/n, "y": v["y"]/n, "z": v["z"]/n}

    return norm(right), norm(forward), norm(up), _pos_dict(agent["position"])


def get_horizontal_axes(meta):
    """从物体或相机元信息中提取水平 forward/right 向量"""
    # 如果传入的是 Controller 对象，获取 agent 的 metadata
    if hasattr(meta, 'last_event'):
        meta = meta.last_event.metadata['agent']
    # 如果传入的是字典但没有 rotation 键，也尝试获取 agent
    elif isinstance(meta, dict) and 'rotation' not in meta:
        if 'agent' in meta:
            meta = meta['agent']

    import math
    yaw = meta["rotation"]["y"]
    t = math.radians(yaw)
    forward = {"x": math.sin(t), "y": 0.0, "z": math.cos(t)}
    right   = {"x": math.cos(t), "y": 0.0, "z": -math.sin(t)}
    return right, forward

def get_spawn_coords_on_receptacle(controller, receptacle_id, anywhere=False):
    coords = []
    try:
        evt = controller.step(
            action="GetSpawnCoordinatesAboveReceptacle",
            objectId=receptacle_id,
            anywhere=anywhere,
            raise_for_failure=True
        )
        coords = evt.metadata["actionReturn"] or []
    except Exception:
        evt = controller.step(
            action="GetSpawnCoordinates",
            objectId=receptacle_id,
            anywhere=anywhere
        )
        coords = evt.metadata.get("actionReturn", []) or []
    return coords


def get_spawn_coords_on_receptacle(controller, receptacle_id, anywhere=False):
    """
    ✅ 更稳健的版本：
    - 如果 receptacle 不可见或不存在 → 返回 []
    - 只尝试 GetSpawnCoordinatesAboveReceptacle（AI2-THOR 支持且安全）
    - 如果失败，不再调用 GetSpawnCoordinates（因为 RoboTHOR 环境中没有）
    """
    # 1. 先找 receptacle 有没有在 metadata 中
    objs = controller.last_event.metadata.get("objects", [])
    receptacle = next((o for o in objs if o["objectId"] == receptacle_id), None)
    if receptacle is None:
        print(f"[spawn_utils] Receptacle {receptacle_id} not found in current metadata.")
        return []

    # 2. 如果要求必须可见（anywhere=False），但它不可见，直接返回空
    if not receptacle.get("visible", False) and not anywhere:
        print(f"[spawn_utils] Receptacle {receptacle_id} is not visible; cannot get spawn coordinates with anywhere=False.")
        return []

    # 3. 尝试安全调用 GetSpawnCoordinatesAboveReceptacle
    try:
        evt = controller.step(
            action="GetSpawnCoordinatesAboveReceptacle",
            objectId=receptacle_id,
            anywhere=anywhere,
            raise_for_failure=True  # 允许抛异常，方便捕获
        )
        coords = evt.metadata.get("actionReturn", []) or []
        return coords
    except Exception:
        # 4. 不调用 GetSpawnCoordinates（在 RoboTHOR 中无效）
        try:
            evt = controller.step(
                action="GetSpawnCoordinates",
                objectId=receptacle_id,
                anywhere=anywhere,
                raise_for_failure=False  # 不要抛错误
            )
            return evt.metadata.get("actionReturn", []) or []
        except Exception:
            return []



def pick_nearest_coord(coords, target_pos):
    if not coords:
        return None
    return min(coords, key=lambda c: _dist3(c, target_pos))


def pick_best_coord(coords, target, base, dir_vec, max_dist_diff=0.1, max_angle_diff_deg=10):
    best, best_score = None, float("inf")
    for c in coords:
        dx = c["x"] - base["x"]
        dz = c["z"] - base["z"]
        dist = math.sqrt(dx * dx + dz * dz)
        if dist < 1e-5:
            continue

        # 方向相似度
        cand_vec = {"x": dx / dist, "z": dz / dist}
        cos_angle = _dot2(cand_vec, dir_vec) / (_norm2(cand_vec) * _norm2(dir_vec))
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = math.degrees(math.acos(cos_angle))
        if angle_deg > max_angle_diff_deg:
            continue

        # 距离误差过滤
        target_dist = math.sqrt(
            (target["x"] - base["x"]) ** 2 + (target["z"] - base["z"]) ** 2
        )
        if abs(dist - target_dist) > max_dist_diff:
            continue

        # 与 target 距离最小的优先
        d_err = math.sqrt(
            (target["x"] - c["x"]) ** 2 + (target["z"] - c["z"]) ** 2
        )
        if d_err < best_score:
            best, best_score = c, d_err
    return best

def place_object_at_coord(controller, object_id, coord):
    evt = controller.step(
        action="PlaceObjectAtPoint",
        objectId=object_id,
        position=_pos_dict(coord),
        raise_for_failure=False
    )
    if not evt.metadata.get("lastActionSuccess", False):
        return False, evt.metadata.get("errorMessage", ""), None
    return True, "", coord

def pick_nearest_visible_receptacle(controller, target_pos):
    objs = controller.last_event.metadata.get("objects", [])
    receptacles = [o for o in objs if o.get("visible") and o.get("receptacle", False)]
    if not receptacles:
        return None
    return min(receptacles, key=lambda o: _dist3(_pos_dict(o["position"]), target_pos))


import matplotlib.pyplot as plt
import os

def get_spawn_coords_on_receptacle_safe(controller, receptacle_id, anywhere=True):
    """安全调用 Spawn API，不报错，失败返回空列表"""
    try:
        evt = controller.step(
            action="GetSpawnCoordinatesAboveReceptacle",
            objectId=receptacle_id,
            anywhere=anywhere,
            raise_for_failure=True
        )
        return evt.metadata.get("actionReturn", []) or []
    except Exception:
        return []

def find_all_receptacles(controller):
    """返回当前 metadata 中所有 receptacle 物体（不要求可见）"""
    objs = controller.last_event.metadata["objects"]
    return [o for o in objs if o.get("receptacle")]

def save_all_receptacle_spawn_plot(controller,
                                   out_path="./vis/spawn_all_recs.png",
                                   anywhere=True,
                                   max_recs=None):
    """
    ✅ 可视化地面 reachable + 全部 receptacle 的 spawn(x,z) 点
    ✅ 无需可见（不限制 visible）
    ✅ 不 show，直接保存
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 1) 地面的 reachable positions
    evt_r = controller.step(action="GetReachablePositions")
    reachable_positions = evt_r.metadata["actionReturn"]
    xs = [p["x"] for p in reachable_positions]
    zs = [p["z"] for p in reachable_positions]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(xs, zs, s=4, c="#BBBBBB", alpha=0.5, label="Reachable Floor")

    # 2) 所有 receptacle 物体
    all_recs = find_all_receptacles(controller)
    if max_recs:  # 限制数量
        all_recs = all_recs[:max_recs]

    print(f"[Info] Total Receptacles Found: {len(all_recs)}")

    colors = [
        "tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown",
        "tab:pink","tab:olive","tab:cyan","gold","teal","navy"
    ]

    for i, rec in enumerate(all_recs):
        rid = rec["objectId"]
        coords = get_spawn_coords_on_receptacle_safe(controller, rid, anywhere=anywhere)
        if not coords:
            continue

        cx = [c["x"] for c in coords]
        cz = [c["z"] for c in coords]

        ax.scatter(cx, cz, s=5, c=colors[i % len(colors)], alpha=0.8, label=rec["objectType"])

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal")
    ax.set_title("Spawn Coordinates of All Receptacles (x–z)")
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] {out_path}")
