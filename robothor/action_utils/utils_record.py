# utils_record.py
import os, json
import numpy as np
from PIL import Image

def get_camera_info(controller):
    """
    获取相机的pose和内参信息

    Returns:
        dict: 包含相机位置、旋转、内参等信息
    """
    event = controller.last_event
    metadata = event.metadata

    # 获取相机pose
    camera_position = metadata.get("cameraPosition", {})
    camera_rotation = metadata.get("agent", {}).get("cameraHorizon", 0)
    agent_rotation = metadata.get("agent", {}).get("rotation", {})

    # 获取图像尺寸
    frame_shape = event.frame.shape
    height, width = frame_shape[0], frame_shape[1]

    # 获取FOV
    fov = metadata.get("fov", 90.0)

    # 计算相机内参矩阵
    # fx = fy = (width / 2) / tan(fov/2)
    import math
    fov_rad = math.radians(fov)
    fx = fy = (width / 2.0) / math.tan(fov_rad / 2.0)
    cx = width / 2.0
    cy = height / 2.0

    camera_info = {
        "position": {
            "x": camera_position.get("x", 0.0),
            "y": camera_position.get("y", 0.0),
            "z": camera_position.get("z", 0.0)
        },
        "rotation": {
            "x": camera_rotation,  # 相机俯仰角(horizon)
            "y": agent_rotation.get("y", 0.0),  # agent的y轴旋转
            "z": 0.0  # RoboTHOR通常不使用z轴旋转
        },
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
            "fov": fov
        },
        # 相机内参矩阵 K (3x3)
        "K": [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ]
    }

    return camera_info

import math
def get_camera_info_gpt5(controller):
    """
    获取 AI2-THOR 当前相机的位姿 (pose) 与内参 (intrinsics)。

    Returns:
        dict: {
            position, rotation, intrinsics, K
        }
    """
    event = controller.last_event
    metadata = event.metadata
    agent = metadata.get("agent", {})

    # === 1️⃣ 相机 Pose（位置与旋转） ===
    # AI2-THOR 没有直接暴露 cameraPosition，
    # 通常相机位置与 agent["position"] 一致（除非使用 drone 或特殊相机模式）
    camera_pos = agent.get("position", {})
    agent_rot = agent.get("rotation", {})
    camera_horizon = agent.get("cameraHorizon", 0.0)

    # === 2️⃣ 图像尺寸 ===
    height, width = event.frame.shape[0:2]

    # === 3️⃣ FOV（水平视场角） ===
    fov = metadata.get("fov", 90.0)

    # === 4️⃣ 内参计算 ===
    fov_rad = math.radians(fov)
    fx = fy = (width / 2.0) / math.tan(fov_rad / 2.0)
    cx, cy = width / 2.0, height / 2.0

    # === 5️⃣ 汇总结果 ===
    return {
        "position": {
            "x": camera_pos.get("x", 0.0),
            "y": camera_pos.get("y", 0.0),
            "z": camera_pos.get("z", 0.0)
        },
        "rotation": {
            "x": camera_horizon,           # pitch (俯仰角)
            "y": agent_rot.get("y", 0.0),  # yaw (水平旋转)
            "z": agent_rot.get("z", 0.0)   # roll（通常为0）
        },
        "intrinsics": {
            "fx": fx, "fy": fy,
            "cx": cx, "cy": cy,
            "width": width, "height": height,
            "fov": fov
        },
        "K": [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ]
    }


def save_current_view(controller, output_dir='vis', prefix='current'):
    os.makedirs(output_dir, exist_ok=True)
    event = controller.last_event

    paths = {}
    rgb_path = os.path.join(output_dir, f"{prefix}_rgb.png")
    Image.fromarray(event.frame).save(rgb_path)
    paths["rgb"] = rgb_path

    if getattr(event, "depth_frame", None) is not None:
        depth = event.depth_frame
        dpath = os.path.join(output_dir, f"{prefix}_depth.png")
        if depth.max() > 0:
            scaled = (depth / (depth.max() + 1e-6) * 255).astype(np.uint8)
        else:
            scaled = (depth * 0).astype(np.uint8)
        Image.fromarray(scaled).save(dpath)
        paths["depth"] = dpath

    if getattr(event, "instance_segmentation_frame", None) is not None:
        seg = event.instance_segmentation_frame
        spath = os.path.join(output_dir, f"{prefix}_seg.png")
        Image.fromarray(seg).save(spath)
        paths["seg"] = spath

    return paths

def append_jsonl(record, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")



from PIL import Image
import copy

def get_top_down_frame(controller):
    """
    ✅ 获取当前场景的俯视图（Top-down camera）
    - 使用 AI2-THOR 内置的 GetMapViewCameraProperties
    - 自动在场景中心正上方生成第三方相机
    - 返回 PIL.Image 格式
    """
    # Step 1: 获取 MapView 相机属性
    event = controller.step(
        action="GetMapViewCameraProperties",
        raise_for_failure=True
    )
    cam_props = copy.deepcopy(event.metadata["actionReturn"])
    
    # Step 2: 抬高相机到场景上方，保证俯视
    bounds = event.metadata["sceneBounds"]["size"]
    max_range = max(bounds["x"], bounds["z"])

    cam_props.update({
        "fieldOfView": 60,
        "position": {
            "x": cam_props["position"]["x"],
            "y": cam_props["position"]["y"] + max_range * 1.3,
            "z": cam_props["position"]["z"]
        },
        "rotation": {"x": 90, "y": 0, "z": 0},  # ✅ 俯视图朝下
        "orthographic": False,  # 保留透视图（必要时可改 True）
        "farClippingPlane": 50,
    })

    # 必须删除 orthographicSize，否则部分版本报错
    cam_props.pop("orthographicSize", None)

    # Step 3: 添加第三方相机并获取图像
    event = controller.step(action="AddThirdPartyCamera", **cam_props, skyboxColor="white")
    frame = event.third_party_camera_frames[-1]  # 最新添加的相机画面

    return Image.fromarray(frame)

