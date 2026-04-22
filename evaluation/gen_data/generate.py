import os
import json
import numpy as np
import cv2
import argparse
import yaml
from pathlib import Path
import random
from datasets.dataloader import Dataset
from utils.train_utils import *
from utils.img_utils import *
from utils.box_utils import get_corners_3d, calc_volume,  move_bbox, rotate_along_y_axis
from utils.item_utils import get_objs, find_refer, decide_reference

from models.model import DetAny3DInference

actions = ["move", "rotate", "remove"]
directions = ["left", "right", "front", "behind", "above", "below"]
weights = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
rot_directions = ["clockwise", "counterclockwise"]
refer_list = ["the leftest", "the rightest", "the frontest", "the backest", "the highest", "the lowest", "the furthest", "the nearest"]
distances = [10, 20, 30, 50, 70, 90, 110]
angles = [30, 45, 60, 90, 120, 150, 180]
vol_threshold = 0.4
fixed_objs = ["door", "floor", "ceiling", "wall", "window frame", "window", "doorframe", "door", "blinds", "ceiling light", "object", "bed"]
    

def gen_move(camera_intrinsics, objects, target, target_label, use_table=False):
    # 1. choose target 
    reference, refer = None, None

    # use table as reference to improve success rate
    if use_table:
        reference = random.choice(get_objs(objects, "table"))
        target, refer = decide_reference(objects, reference, refer_list, refer=True)
        if len(get_objs(objects, reference["label"])) <= 1:
            refer = None
        if target is None:
            return
    else:
        objs = get_objs(objects, target_label)
        if len(objs) > 1:
            refer = random.choice(refer_list)
            target = find_refer(objs, refer_list)[refer]
    
    # too large to move
    if calc_volume(target["bbox_3d"]) > vol_threshold:
        return

    eps = random.uniform(0, 1)
    if eps < 0.5:
        # move a random direction
        direction = random.choice(["left", "right", "front", "behind"]) if use_table else random.choices(directions, weights=weights, k=1)[0]
        dist = random.choice(distances)
        original_bbox_3d = target["bbox_3d"]
        rot_mat = target["rotation_matrix"]
        new_bbox_3d = move_bbox(original_bbox_3d, direction, dist, rot_mat)
        
        prompt = f"Move {refer} {target_label} {dist} centimeters to the {direction}, while keeping other objects unchanged."
        if refer is None:
            prompt = f"Move the {target_label} {dist} centimeters to the {direction}, while keeping other objects unchanged." 
        return {
            "prompt": prompt,
            "target": target_label,
            "original_bbox_3d": original_bbox_3d,
            "new_bbox_3d": new_bbox_3d,
            "camera_intrinsics": camera_intrinsics,
            "rotation_matrix": rot_mat
        }
    elif eps < 0.8:
        if not use_table:
            reference = decide_reference(objects, target, refer_list)
        if reference is None:
            return
        original_bbox_3d = target["bbox_3d"]
        direction = random.choice(["left", "right", "front", "behind"]) if use_table else random.choices(directions, weights=weights, k=1)[0]
        dist = random.choice(distances)
        rot_mat = target["rotation_matrix"]
        new_bbox_3d = move_bbox(original_bbox_3d, direction, dist, rot_mat=reference["rotation_matrix"])
        prompt = f"Move the {target_label} {dist} centimeters to the {direction} referencing the {reference['label']}, while keeping other objects unchanged."
        if refer is not None:
            prompt = f"Move {refer} {target_label} {dist} centimeters to the {direction} referencing the {reference['label']}, while keeping other objects unchanged."
        return {
            "prompt": prompt,
            "target": target_label,
            "original_bbox_3d": original_bbox_3d,
            "new_bbox_3d": new_bbox_3d,
            "camera_intrinsics": camera_intrinsics,
            "rotation_matrix": rot_mat,
            "reference": reference
        }
    else:
        if not use_table:
            reference = decide_reference(objects, target, refer_list)
        if reference is None:
            return
        corners = get_corners_3d(reference["bbox_3d"], reference["rotation_matrix"])
        corner = random.choice(corners)
        x, y, z, w, h, l, yaw = target["bbox_3d"]
        if np.abs(y - corner[1]) > 0.2:
            return
        new_bbox_3d = [corner[0], corner[1], corner[2], w, h, l, yaw]
        prompt = f"Move the {target_label} to the position of the {reference['label']}, while keeping other objects unchanged."
        if refer is not None:
            prompt = f"Move {refer} {target_label} to the position of the {reference['label']}, while keeping other objects unchanged."
        return {
            "prompt": prompt,
            "target": target_label,
            "original_bbox_3d": target["bbox_3d"],
            "new_bbox_3d": new_bbox_3d,
            "camera_intrinsics": camera_intrinsics,
            "rotation_matrix": target["rotation_matrix"],
            "reference": reference,
            "reference_point": corner.tolist()
        }

def gen_rotate(camera_intrinsics, objects, target, target_label):
    objs = get_objs(objects, target_label)
    refer = None
    if len(objs) > 1:
        refer = random.choice(refer_list)
        target = find_refer(objs, refer_list)[refer]
    rot_direction = random.choice(rot_directions)
    angle = random.choice(angles)
    if refer is None:
        prompt = f"Rotate the {target_label} {angle} degrees {rot_direction} along its vertical axis, while keeping other objects unchanged."
    else:
        prompt = f"Rotate {refer} {target_label} {angle} degrees {rot_direction} along its vertical axis, while keeping other objects unchanged."
    
    original_bbox_3d = target["bbox_3d"]
    rotation_matrix = target["rotation_matrix"]
    new_rotation_matrix = rotate_along_y_axis(rotation_matrix, angle, rot_direction)
    return {
        "prompt": prompt,
        "target": target_label,
        "original_bbox_3d": original_bbox_3d,
        "new_bbox_3d": original_bbox_3d, 
        "camera_intrinsics": camera_intrinsics,
        "rotation_matrix": rotation_matrix,
        "new_rotation_matrix": new_rotation_matrix.tolist()
    }
                

def gen_remove(camera_intrinsics, objects, target, target_label):
    objs = get_objs(objects, target_label)
    if len(objs) <= 1:
        return
    refer = random.choice(refer_list)
    target = find_refer(objs, refer_list)[refer]
    prompt = f"Remove {refer} {target_label} from the scene, while keeping other objects unchanged."
    return {
        "prompt": prompt,
        "target": target_label,
        "original_bbox_3d": target["bbox_3d"],
        "camera_intrinsics": camera_intrinsics,
        "rotation_matrix": target["rotation_matrix"],
    }            


def gen_edit(camera_intrinsics, pose, result, img_id, frame_id, output_dir):
    edit_dict = {}
    cnt = random.randint(20, 28)
    # cnt = random.randint(5, 10)
    objects = result['detections']
    table_exist = len(get_objs(objects, "table")) > 0
    if objects is None or len(objects) == 0:
        return
    for i in range(cnt):
        for j in range(20):
            action = "move" if table_exist and i < 20 else random.choice(actions) 
            target = random.choice(objects)
            target_label = target["label"]
            new_gen = None
            if action == "move":
                new_gen = gen_move(camera_intrinsics, objects, target, target_label, use_table=table_exist and i < 20)
            elif action == "rotate":
                edit_dict[str(i)] = gen_rotate(camera_intrinsics, objects, target, target_label)
            elif action == "remove":
                new_gen = gen_remove(camera_intrinsics, objects, target, target_label) 
            
            if new_gen is not None:
                for key in edit_dict.keys():
                    if edit_dict[key]["prompt"] == new_gen["prompt"]:
                        new_gen = None
                        break
            if new_gen is not None:
                edit_dict[str(i)] = new_gen
                break
    try:
        json.dump(edit_dict, open(os.path.join(output_dir, f"{img_id}_frame_{frame_id}_edit.json"), 'w'), indent=4)
    except Exception as e:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", "-s", type=str, required=True)
    parser.add_argument("--dst", "-d", type=str, required=True)
    parser.add_argument("--dataset", "-D", type=str, default="ScanNet++")
    parser.add_argument("--cfg", "-c", type=str, default="./detect_anything/configs/demo.yaml")
    args = parser.parse_args()

    dataset = Dataset(args.src, args.dataset)
    print("Dataset size:", len(dataset))
    inference = DetAny3DInference(args.cfg)
    for i in range(len(dataset)):
        sample = dataset[i]
        img_path = sample["img_path"]
        img_id = sample["img_id"]
        frame_id = sample["frame_id"]

        # Prepare pose and intrinsics
        pose = sample["pose"]
        intrinsics = sample["intrinsics"]

        # Prepare text prompts
        text_prompts = set()
        anno = sample["anno"]
        for item in anno["segGroups"]:
            text_prompts.add(item["label"])
        text_prompts = list(text_prompts)

        result = inference.predict(
            img_path,
            text_prompts=text_prompts,
            point_coords=None,
            bbox_coords=None,
            output_dir=args.dst,
            show_orientation=False,
            gt_camera_intrinsics=intrinsics
        )
        gen_edit(intrinsics.tolist(), pose, result, img_id, frame_id, args.dst)

if __name__ == "__main__":
    main()