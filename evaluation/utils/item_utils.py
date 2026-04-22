import numpy as np
from .box_utils import iou_2d
import random

def dict2np(item):
    arrs = []
    for key in item.keys():
        if isinstance(item[key], list):
            arrs.append(np.array(item[key], dtype=float).flatten())
        elif isinstance(item[key], np.ndarray):
            arrs.append(item[key].flatten())
    if arrs:
        return np.concatenate(arrs)
    else:
        return np.array([], dtype=float)

def check_same(item1, item2, threshold=1e-1):
    """check if two items are the same within a threshold"""
    if isinstance(item1, dict):
        data1 = dict2np(item1)
        data2 = dict2np(item2)
        diff = np.sum((data1 - data2) ** 2)
    elif isinstance(item1, np.ndarray):
        diff = np.sum((item1 - item2) ** 2)
    elif isinstance(item1, list):
        diff = np.sum((np.array(item1) - np.array(item2)) ** 2)
    else:
        raise ValueError("Unsupported type for check_same")
    return diff < threshold  

def find_attach(target, objects):
    max_iou = 0.0
    attach = None
    for item in objects:
        if check_same(target, item):
            continue
        iou = iou_2d(np.array(target["bbox_2d"]), np.array(item["bbox_2d"]))
        if iou > max_iou:
            max_iou = iou
            attach = item
    if max_iou < 0.1:
        attach = None
    return attach

def get_objs(objects, label):
    """get all objects with the same label, removing duplicates"""
    obj_list = []
    for item in objects:
        if item["label"] != label:
            continue
        flag = True
        for obj in obj_list:
            if check_same(item, obj):
                flag = False
                break
        if flag:
            obj_list.append(item)
    return obj_list

def find_refer(objs, refer_list):
    """find the refer objects in objs according to refer_list"""
    result_dir = {refer: None for refer in refer_list}
    for obj in objs:
        x, y, z, w, h, l, yaw = obj["bbox_3d"]
        if result_dir["the leftest"] is None or x < result_dir["the leftest"]["bbox_3d"][0]:
            result_dir["the leftest"] = obj
        if result_dir["the rightest"] is None or x > result_dir["the rightest"]["bbox_3d"][0]:
            result_dir["the rightest"] = obj
        if result_dir["the frontest"] is None or z < result_dir["the frontest"]["bbox_3d"][2]:
            result_dir["the frontest"] = obj
        if result_dir["the backest"] is None or z > result_dir["the backest"]["bbox_3d"][2]:
            result_dir["the backest"] = obj
        if result_dir["the highest"] is None or y < result_dir["the highest"]["bbox_3d"][1]:
            result_dir["the highest"] = obj
        if result_dir["the lowest"] is None or y > result_dir["the lowest"]["bbox_3d"][1]:
            result_dir["the lowest"] = obj
        dist = np.sqrt(x**2 + y**2 + z**2)
        nearest_dist = np.sqrt(result_dir["the nearest"]["bbox_3d"][0]**2 + result_dir["the nearest"]["bbox_3d"][1]**2 + result_dir["the nearest"]["bbox_3d"][2]**2) if result_dir["the nearest"] is not None else float('inf')
        furthest_dist = np.sqrt(result_dir["the furthest"]["bbox_3d"][0]**2 + result_dir["the furthest"]["bbox_3d"][1]**2 + result_dir["the furthest"]["bbox_3d"][2]**2) if result_dir["the furthest"] is not None else 0.0
        if result_dir["the nearest"] is None or dist < nearest_dist:
            result_dir["the nearest"] = obj
        if result_dir["the furthest"] is None or dist > furthest_dist:  
            result_dir["the furthest"] = obj
    return result_dir

def decide_reference(objects, target, refer_list, refer=False):
    """choose a reference object from objects that overlaps with target in 2D bbox
    if refer is True, choose randomly according to refer_list"""
    reference_candidates = []
    for obj in objects:
        exist = check_same(obj["bbox_3d"], target["bbox_3d"])
        for item in reference_candidates:
            if check_same(obj["bbox_3d"], item["bbox_3d"]):
                exist = True
                break
        if not exist and iou_2d(np.array(target["bbox_2d"]), np.array(obj["bbox_2d"])) > 0:
            reference_candidates.append(obj)
    if not refer:
        reference = random.choice(reference_candidates) if len(reference_candidates) > 0 else None
        return reference
    else:
        refer = random.choice(refer_list)
        refer_dict = find_refer(reference_candidates, refer_list)
        reference = refer_dict[refer] if refer_dict[refer] is not None else None
        return reference, refer