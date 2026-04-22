import numpy as np
import cv2
from shapely.geometry import Polygon

from .train_utils import *
from .img_utils import draw_orientation_arrow

def get_corners_3d(bbox, rot_mat=None):
    x, y, z, w, h, l, yaw = bbox
    corners = np.array([
        [w/2, -h/2, l/2],
        [w/2, -h/2, -l/2],
        [-w/2, -h/2, -l/2],
        [-w/2, -h/2, l/2],
        [w/2, h/2, l/2],
        [w/2, h/2, -l/2],
        [-w/2, h/2, -l/2],
        [-w/2, h/2, l/2],
    ])
    if rot_mat is not None:
        R = np.array(rot_mat)
    else:
        R = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)],
        ])
    corners = (R @ corners.T).T + np.array([x, y, z])
    return corners

def project_bbox(bbox, K):
    corners = get_corners_3d(bbox)
    corners_2d = (K @ corners.T).T
    corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]
    x_min, y_min = np.min(corners_2d, axis=0)
    x_max, y_max = np.max(corners_2d, axis=0)
    return [x_min, y_min, x_max, y_max]

def bbox_3d_to_2d(bbox_3d, rot_mat, K):
    x, y, z, w, h, l, yaw = bbox_3d
    vertices_3d, _ = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rot_mat)
    vertices_2d = project_to_image(vertices_3d, K)
    bbox_2d = np.array([np.min(vertices_2d[:, 0]), np.min(vertices_2d[:, 1]), np.max(vertices_2d[:, 0]), np.max(vertices_2d[:, 1])])
    return bbox_2d
    
def calc_volume(bbox_3d):
    w, h, l = bbox_3d[3: 6]
    return w * h * l

def iou_2d(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Calculate the Intersection over Union (IoU) of two 2D bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def iou_3d(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    corners1 = get_corners_3d(bbox1)
    corners2 = get_corners_3d(bbox2)
    poly1 = Polygon(corners1[:4, [0,2]])
    poly2 = Polygon(corners2[:4, [0,2]])
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    try:
        inter_area = poly1.intersection(poly2).area
    except Exception:
        print("Invalid polygon detected!", corners1[:4, [0,2]], corners2[:4, [0,2]])
        return 0.0
    
    ymin1, ymax1 = corners1[:,1].min(), corners1[:,1].max()
    ymin2, ymax2 = corners2[:,1].min(), corners2[:,1].max()
    inter_ymin = max(ymin1, ymin2)
    inter_ymax = min(ymax1, ymax2)

    inter_vol = inter_area * max(0.0, inter_ymax - inter_ymin)
    vol1 = poly1.area * (ymax1 - ymin1)
    vol2 = poly2.area * (ymax2 - ymin2)
    union_vol = vol1 + vol2 - inter_vol

    return inter_vol / union_vol if union_vol > 0 else 0.0

def move_bbox(bbox, direction, distance, rot_mat=None):
    distance /= 100.0 # cm to m
    x_dir, y_dir, z_dir = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
    if rot_mat is not None:
        rot_mat = np.array(rot_mat)
        x_dir = rot_mat[:, np.argmax(np.abs(rot_mat[:, 0]))]
        y_dir = rot_mat[:, np.argmax(np.abs(rot_mat[:, 1]))]
        z_dir = rot_mat[:, np.argmax(np.abs(rot_mat[:, 2]))]
    xyz, w, h, l, yaw = bbox[:3], bbox[3], bbox[4], bbox[5], bbox[6]
    if direction == "left":
        xyz -= x_dir * distance
    elif direction == "right":
        xyz += x_dir * distance
    elif direction == "front":
        xyz -= z_dir * distance
    elif direction == "behind":
        xyz += z_dir * distance
    elif direction == "above":
        xyz -= y_dir * distance
    elif direction == "below":
        xyz += y_dir * distance
    x, y, z = xyz
    return [float(x), float(y), float(z), float(w), float(h), float(l), float(yaw)]


def rotate_along_y_axis(rot_mat, angle, direction="clockwise"):
    assert 0 <= angle <= 360 and direction in ["clockwise", "counterclockwise"]
    angle = np.pi * angle / 180.0 if direction == "counterclockwise" else np.pi * (360.0 - angle) / 180.0
    R_y = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])
    new_rot_mat = rot_mat @ R_y.T
    return new_rot_mat

def visualize_bbox(bbox, rot_mat, K, img, mode="Original", show_orientation=False, show_2d=False):
    color_dict = {
        "Original": (0, 255, 0),
        "Modified": (0, 0, 255),
        "Reference": (255, 0, 0),
    }

    x, y, z, w, h, l, yaw = bbox
    vertices_3d, for_plain_center = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rot_mat)
    vertices_2d = project_to_image(vertices_3d, K)
    origin = np.array([-0.05, -0.05, 1])
    origin_2d = project_to_image(origin.reshape(1, 3), K)[0]
    if show_2d:
        print("first 3d vertex: ", vertices_3d[0])
        print("first 2d vertex: ", vertices_2d[0])
        print(origin_2d)
    # cv2.circle(img, (int(origin_2d[0]), int(origin_2d[1])), 5, (255, 255, 0), 5)
    draw_bbox_2d(img, vertices_2d, color=color_dict[mode])
    cv2.putText(img, mode, (int(vertices_2d[0][0]), int(vertices_2d[0][1]+10)), cv2.FONT_HERSHEY_SIMPLEX, 1, color_dict[mode], 2)
    if show_orientation:
        center_3d = np.array([x, y, z])
        draw_orientation_arrow(img, center_3d, rot_mat, K, arrow_length=1, color=(0, 255, 0))
