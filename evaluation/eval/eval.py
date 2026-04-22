import os
import json
import logging
import numpy as np
import cv2
import argparse
from PIL import Image
from glob import glob
from copy import deepcopy
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from models.model import DetAny3DInference
from utils.box_utils import iou_3d, bbox_3d_to_2d
from utils.item_utils import check_same
from utils.img_utils import img_process, update_mask_2d
from utils.train_utils import *
from .eval_utils import (
    get_img_data,
    get_operation,
    load_detection,
    get_origin_bbox_2d,
    masked_ssim, 
    masked_lpips, 
    masked_mse, 
    load_config, 
    check_locality,
    check_viewpoint_similarity, 
    # sam_img
)
from .statistics_utils import compute_statistics

def _is_truthy(value):
    if value is None:
        return False
    return str(value).lower() in ("1", "true", "yes", "on")

def setup_debugpy(enabled=False, host="127.0.0.1", port=5678, wait_for_client=False):
    if not enabled:
        return
    try:
        import debugpy
    except ImportError as exc:
        raise ImportError("debugpy is not installed. Please install it with `pip install debugpy`.") from exc

    debugpy.listen((host, port))
    print(f"[debugpy] Listening on {host}:{port}")
    if wait_for_client:
        print("[debugpy] Waiting for debugger to attach...")
        debugpy.wait_for_client()

def rotation_matrix_distance(R1, R2):
    """Calculate the angular distance between two rotation matrices."""
    R = np.dot(R1.T, R2)
    trace = np.trace(R)
    theta = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
    return theta

def dist_between_centers(bbox1, bbox2):
    """Calculate the Euclidean distance between the centers of two 3D bounding boxes."""
    if isinstance(bbox1, list):
        bbox1 = np.array(bbox1)
    if isinstance(bbox2, list):
        bbox2 = np.array(bbox2)
    center1 = bbox1[:3]
    center2 = bbox2[:3]
    return np.linalg.norm(center1 - center2)

def eval_move(edit_meta, origin_res, result, mode, check):
    if not check or len(result["detections"]) == 0:
        return 0.0
        
    # Get original 2D bbox and compute ground-truth 2D movement
    origin_bbox_2d = get_origin_bbox_2d(edit_meta, origin_res)
    if origin_bbox_2d is None:
        return 0.0

    origin_detections, detections = origin_res["detections"], result["detections"]
    gt_bbox = np.array(edit_meta["new_bbox_3d"])
    gt_bbox_2d = bbox_3d_to_2d(gt_bbox, np.array(edit_meta["rotation_matrix"]), np.array(edit_meta["camera_intrinsics"]))
    gt_2d_move = gt_bbox_2d - origin_bbox_2d
    
    # find the most possible moved bbox
    thres_2d = 0.2
    min_dist, target_pred_bbox = float('inf'), None
    for detection in result["detections"]:
        if not any(
            check_same(detection["bbox_3d"], od["bbox_3d"], 1e-1) or
            check_same(detection["bbox_2d"], od["bbox_2d"], 100)
            for od in origin_detections
        ):
            ratio_2d = np.linalg.norm(np.array(detection["bbox_2d"]) - origin_bbox_2d) / (np.linalg.norm(gt_2d_move) + 1e-5)
            dist_to_gt = dist_between_centers(detection["bbox_3d"], gt_bbox)
            if dist_to_gt < min_dist and ratio_2d > thres_2d:
                target_pred_bbox = detection["bbox_3d"]
                min_dist = dist_to_gt

    # compute the iou and distance-based scores
    score1 = iou_3d(gt_bbox, np.array(target_pred_bbox)) if target_pred_bbox is not None else 0.0
    pred_dist = dist_between_centers(edit_meta["original_bbox_3d"], target_pred_bbox) if target_pred_bbox is not None else 0.0
    gt_dist = dist_between_centers(edit_meta["original_bbox_3d"], gt_bbox)
    ratio_3d = pred_dist / (gt_dist + 1e-5)
    score2 = np.exp(-np.log(np.sqrt(ratio_3d) + 1e-5)**2)
    score = 0.3 * score1 + 0.7 * score2

    # number consistency check
    num_check = (len(result["detections"]) == len(origin_res["detections"]))
    if not num_check:
        score *= 0.5

    if mode == "instruction-compliance":
        return 1.0 if score > 0.05 else 0.0
    elif mode == "spatial-accuracy":
        return score
    

def eval_rotate(origin_img, edited_img, edit_meta, origin_res, result, mode, check, model=None):
    if not check or len(result["detections"]) == 0:
        return 0.0

    # Get original 2D bbox and compute ground-truth 2D movement
    origin_bbox_2d = get_origin_bbox_2d(edit_meta, origin_res)
    if origin_bbox_2d is None:
        return 0.0
    gt_bbox = np.array(edit_meta["original_bbox_3d"])
    gt_bbox_2d = bbox_3d_to_2d(gt_bbox, np.array(edit_meta["new_rotation_matrix"]), np.array(edit_meta["camera_intrinsics"]))
    gt_2d_move = gt_bbox_2d - origin_bbox_2d

    # find the most possible rotated bbox
    thres_2d = 0.2
    min_dist, target_pred_bbox, target_pred_rot = float('inf'), None, None
    for i, detection in enumerate(result["detections"]):
        if not any(
            check_same(detection["bbox_3d"], od["bbox_3d"], 1e-3) or
            check_same(detection["bbox_2d"], od["bbox_2d"], 100)
            for od in origin_res["detections"]
        ):
            ratio_2d = np.linalg.norm(np.array(detection["bbox_2d"]) - origin_bbox_2d) / (np.linalg.norm(gt_2d_move) + 1e-5)
            dist = dist_between_centers(detection["bbox_3d"], gt_bbox)
            if dist < min_dist and ratio_2d > thres_2d:
                target_pred_bbox = detection["bbox_3d"]
                target_pred_rot = detection["rotation_matrix"]
                min_dist = dist

    # compute rotation-based score
    score = 0.0
    if target_pred_rot is not None and min_dist < 0.5:
        R_origin, R_gt = np.array(edit_meta["rotation_matrix"]), np.array(edit_meta["new_rotation_matrix"])
        R_pred = np.array(target_pred_rot)
        theta_gt = rotation_matrix_distance(R_origin, R_gt)
        theta_pred = rotation_matrix_distance(R_origin, R_pred)
        theta_diff = rotation_matrix_distance(R_gt, R_pred)
        ratio = theta_pred / (theta_gt + 1e-5)
        score = 0.5 * np.exp(-np.log(ratio + 1e-5)**2) + 0.5 * np.exp(-np.abs(theta_diff))

    if mode == "instruction-compliance":
        if target_pred_rot is not None:
            return 1.0 if not check_same(edit_meta["rotation_matrix"], target_pred_rot, 1e-4) else 0.0
        else:
            return 0.0
    elif mode == "spatial-accuracy":
        return score

def eval_remove(edit_meta, origin_res, result, mode, check):
    if not check:
        return 0.0

    # Get original 2D bbox
    origin_bbox_2d = get_origin_bbox_2d(edit_meta, origin_res)
    if origin_bbox_2d is None:
        return 0.0

    # Check if the object is removed
    removed = not any(
        iou_3d(edit_meta["original_bbox_3d"], np.array(detection["bbox_3d"])) > 0.1 or
        check_same(origin_bbox_2d, detection["bbox_2d"], 100)
        for detection in result["detections"]
    )

    # compute score
    score = 0.0
    if removed:
        # TODO: score formula doesn't penalize extra detections; can clip to 1.0 even with false positives.
        score =  1 - 0.3 * (len(origin_res["detections"]) - len(result["detections"]) - 1)
        score = np.clip(score, 0.0, 1.0)

    if mode == "instruction-compliance":
        return 1.0 if score >= 0.5 else 0.0
    elif mode == "spatial-accuracy":
        return score

def eval_resize(edit_meta, origin_res, result, mode, check):
    if not check or len(result["detections"]) == 0:
        return 0.0
    original_bbox = np.array(edit_meta["original_bbox_3d"])
    thres_2d = 0.2
    # get original 2D bbox and compute ground-truth 2D movement
    origin_bbox_2d = get_origin_bbox_2d(edit_meta, origin_res)
    if origin_bbox_2d is None:
        return 0.0
    gt_bbox = np.array(edit_meta["new_bbox_3d"])
    gt_bbox_2d = bbox_3d_to_2d(gt_bbox, np.array(edit_meta["rotation_matrix"]), np.array(edit_meta["camera_intrinsics"]))
    gt_2d_move = gt_bbox_2d - origin_bbox_2d

    # find the most possible resized bbox
    thres_2d = 0.2
    min_dist, target_pred_bbox = float('inf'), None
    for i, detection in enumerate(result["detections"]):
        if not any(
            check_same(detection["bbox_3d"], od["bbox_3d"], 1e-3) or
            check_same(detection["bbox_2d"], od["bbox_2d"], 100)
            for od in origin_res["detections"]
        ):
            ratio_2d = np.linalg.norm(np.array(detection["bbox_2d"]) - origin_bbox_2d) / (np.linalg.norm(gt_2d_move) + 1e-5)
            dist = dist_between_centers(detection["bbox_3d"], gt_bbox)
            if dist < min_dist and ratio_2d > thres_2d:
                target_pred_bbox = detection["bbox_3d"]
                min_dist = dist

    # compute scale-based score
    scale_gt = np.array(edit_meta["new_bbox_3d"])[3: 6] / (original_bbox[3: 6] + 1e-5)
    scale_pred = np.array(target_pred_bbox)[3: 6] / (original_bbox[3: 6] + 1e-5) if target_pred_bbox is not None else np.array([1.0, 1.0, 1.0])
    ratio = scale_pred / (scale_gt + 1e-5)
    score = np.exp(-np.mean(np.log(ratio + 1e-5)**2))
            
    if mode == "instruction-compliance":
        return 1.0 if np.linalg.norm(scale_pred - 1.0) > 0.2 else 0.0
    elif mode == "spatial-accuracy":
        return score

def eval_view(img, gt_img):
    assert img.shape == gt_img.shape, "Image and GT image must have the same shape."
    mask = np.ones(img.shape[:2], dtype=np.uint8)
    mask = mask.astype(bool)
    ssim = masked_ssim(img, gt_img, mask)
    lpips = masked_lpips(img, gt_img, mask)
    return 0.5 * ssim + 0.5 * (1 - lpips)

def eval_locality(img1, img2, origin_res, result, edit_meta, config_path=None, img_path=None, show_masked=False):
    # preprocess images
    cfg = load_config(config_path)
    vis_img1, vis_img2 = deepcopy(img1), deepcopy(img2)
    if vis_img1.shape != vis_img2.shape:
        vis_img2 = cv2.resize(vis_img2, (vis_img1.shape[1], vis_img1.shape[0]))
    vis_img1 = img_process(vis_img1, cfg)
    vis_img2 = img_process(vis_img2, cfg)

    # get original 2D bbox
    origin_bbox_2d = get_origin_bbox_2d(edit_meta, origin_res)
    # get new 2D bbox
    # TODO: new_bbox_2d picks the last non-origin detection; multi-object scenes can select wrong target.
    new_bbox_2d = None
    for detection in result["detections"]:
        K = np.array(edit_meta["camera_intrinsics"])
        # print(edit_meta.keys())
        if not (
            check_same(detection["bbox_3d"], np.array(edit_meta["original_bbox_3d"]), 1e-1) or
            check_same(detection["bbox_2d"], bbox_3d_to_2d(np.array(edit_meta["original_bbox_3d"]), np.array(edit_meta["rotation_matrix"]), K), 100)
        ):
            new_bbox_2d = np.array(detection["bbox_2d"])
   
   # generate mask
    mask = np.zeros(vis_img1.shape[:2], dtype=np.uint8)
    if origin_bbox_2d is not None:
        update_mask_2d(vis_img1.shape, mask, origin_bbox_2d)
    if new_bbox_2d is not None:
        update_mask_2d(vis_img1.shape, mask, new_bbox_2d)
    mask = mask.astype(bool)
    mask = ~mask

    masked_img1 = deepcopy(vis_img1)
    masked_img1[~mask] = 0
    masked_img2 = deepcopy(vis_img2)
    masked_img2[~mask] = 0
    # save masked images for visualization
    if show_masked:
        import matplotlib.pyplot as plt
        
        cv2.imwrite("masked_img1.png", masked_img1)
        cv2.imwrite("masked_img2.png", masked_img2)

    # compute metrics
    masked_img1 = vis_img1
    masked_img2 = vis_img2
    ssim = masked_ssim(masked_img1, masked_img2, mask)
    lpips = masked_lpips(masked_img1, masked_img2, mask)
    mse = masked_mse(masked_img1, masked_img2, mask)
    return ssim, lpips, mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=str, required=True, help="Directory containing original images")
    parser.add_argument("--edited", type=str, required=True, help="Directory containing edited images")
    parser.add_argument("--edit", type=str, required=True, help="JSON file containing edit details")
    parser.add_argument("--output", type=str, default=None, help="Directory to save results")
    parser.add_argument("--mode", type=str, default="spatial-accuracy", help="Evaluation mode")
    parser.add_argument("--dataset", type=str, default="GSI-Real", help="Dataset name for specific checks")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume from existing DetAny3D results (default)"
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not resume from existing DetAny3D results"
    )
    parser.set_defaults(resume=True)

    parser.add_argument('--config', type=str, default='./utils/detect_anything/configs/demo.yaml', help='Path to config file')
    parser.add_argument("--debugpy", action="store_true", help="Enable debugpy listen server")
    parser.add_argument("--debugpy-host", type=str, default="127.0.0.1", help="debugpy host")
    parser.add_argument("--debugpy-port", type=int, default=5678, help="debugpy port")
    parser.add_argument("--debugpy-wait", action="store_true", help="Wait for debugger attach before running")
    args = parser.parse_args()
    debugpy_enabled = args.debugpy or _is_truthy(os.getenv("DEBUGPY"))
    debugpy_wait = args.debugpy_wait or _is_truthy(os.getenv("DEBUGPY_WAIT_FOR_CLIENT"))
    setup_debugpy(debugpy_enabled, args.debugpy_host, args.debugpy_port, debugpy_wait)

    debug = os.getenv("GSI_DEBUG", "").lower() in ("1", "true", "yes")
    original = args.original
    edited = args.edited
    edit_path = args.edit
    output_dir = args.output
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
    progress_file = os.path.join(output_dir, f"{args.mode}_progress.json") if output_dir is not None else None
    # if progress_file and os.path.exists(progress_file):
    #     with open(progress_file, "r") as f:
    #         eval_result = json.load(f)
    # else:
    #     eval_result = {}
    eval_result = {}
    infer_cache_file = os.path.join(output_dir, "infer_cache.json") if output_dir is not None else None
    infer_cache = {}
    if infer_cache_file and os.path.exists(infer_cache_file) and args.resume:
        with open(infer_cache_file, "r") as f:
            infer_cache = json.load(f)
        print("Loaded inference cache from", infer_cache_file)

    # TODO: output_dir can be None; avoid calling os.path.exists(None) (TypeError).
    update_cache = (infer_cache_file and not args.resume) or not os.path.exists(infer_cache_file)
    inference = DetAny3DInference(args.config)
    # img_files = glob(os.path.join(args.original, "*.jpg")) + glob(os.path.join(args.original, "*.png"))
    img_files = glob(os.path.join(edited, "*.jpg")) + glob(os.path.join(edited, "*.png"))
    operation_dict = {
        "move": ["move", "shift", "relocate", "place", "put", "push", "pull"],
        "rotate": ["rotate", "turn", "spin"],
        "remove": ["remove"],
        "scale": ["scale", "resize", "make", "shrink", "reduce", "enlarge", "expand", "increase"],
        "view": ["view", "changeview", "look"]
    }
    for img_file in tqdm(img_files):
        # Load required data
        data_dict = get_img_data(img_file, args.original)
        img_id, query_id = data_dict["meta"]["img_id"], data_dict["meta"]["query_id"]
        origin_img_path, img_path = data_dict["original_img"], data_dict["edited_img"]
        gt_edited_img_file, json_file = data_dict["gt_edited_img"], data_dict["edit_json"]
        
        if f"{img_id}_{query_id}" in eval_result:
            continue

        with open(json_file, 'r') as f:
            edit_meta = json.load(f)

        # TODO: img_path can become invalid if filename normalization (e.g., "_rgb") modifies the real path.
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            logging.warning(f"Cannot open image: {img_path}, skip.")
            continue
        
        # Get DetAny3D results for original and edited images
        target = edit_meta["target"]
        operation = get_operation(edit_meta, operation_dict)
        cache_key = f"{img_id}_{query_id}"
        origin_res = load_detection(infer_cache, cache_key, origin_img_path, edit_meta, inference, restype="origin_result")
        result = load_detection(infer_cache, cache_key, img_path, edit_meta, inference, restype="new_result")
        if debug:
            origin_cnt = len(origin_res.get("detections", [])) if isinstance(origin_res, dict) else 0
            new_cnt = len(result.get("detections", [])) if isinstance(result, dict) else 0
            print(f"[GSI_DEBUG] {img_file} target={edit_meta.get('target')} origin_det={origin_cnt} new_det={new_cnt}")
        
        # Read images and ensure the sizes match
        original_img = Image.open(origin_img_path).convert("RGB")
        edited_img = Image.open(img_file).convert("RGB")
        if original_img.size != edited_img.size:
            edited_img = edited_img.resize(original_img.size)
        original_img = np.array(original_img)
        edited_img = np.array(edited_img)

        # Calculate edit-locality metrics
        ssim, lpips, mse = eval_locality(original_img, edited_img, origin_res, result, edit_meta, args.config, img_file)
        if args.mode in ["instruction-compliance", "spatial-accuracy"]:
            check = check_locality(ssim, lpips, args.dataset)
            if operation == "move":
                eval_res = eval_move(edit_meta, origin_res, result, args.mode, check)
            
            elif operation == "rotate":
                # TODO: guard check_viewpoint_similarity for None/exception to avoid aborting a batch.
                check = check and check_viewpoint_similarity(original_img, edited_img)
                eval_res = eval_rotate(original_img, edited_img, edit_meta, origin_res, result, args.mode, check, inference)
            
            elif operation == "remove":
                eval_res = eval_remove(edit_meta, origin_res, result, args.mode, check)

            elif operation == "scale":
                eval_res = eval_resize(edit_meta, origin_res, result, args.mode, check)
            
            elif operation == "view":
                try:
                    gt_edited_img = np.array(Image.open(gt_edited_img_file).convert("RGB"))
                    eval_res = eval_view(edited_img, gt_edited_img)
                except Exception as e:
                    logging.warning(f"View evaluation failed for {img_file} with error: {e}")
                    eval_res = 0.0
            
            else:
                print(edit_meta["prompt"])
                raise NotImplementedError(f"Operation {operation} not implemented yet.")

        
        if args.mode == "instruction-compliance":
            eval_result[f"{img_id}_{query_id}"] = {
                "operation": operation,
                "compliance": float(eval_res),
            }
        elif args.mode == "spatial-accuracy":    
            eval_result[f"{img_id}_{query_id}"] = {
                "operation": operation,
                "edit_score": float(eval_res),
            }
        elif args.mode == "edit-locality":
            eval_result[f"{img_id}_{query_id}"] = {
                "operation": operation,
                "ssim": float(ssim),
                "lpips": float(lpips),
                "mse": float(mse)
            }

        # save infer cache
        if update_cache:
            with open(infer_cache_file, "w") as f:
                json.dump(infer_cache, f, indent=4)

    stats = compute_statistics(eval_result, args.mode, args.output, show=True)

    if output_dir is not None:
        json.dump(eval_result, open(os.path.join(output_dir, f"{args.mode}_eval_results.json"), 'w'), indent=4)
        json.dump(stats, open(os.path.join(output_dir, f"{args.mode}_eval_stats.json"), "w"), indent=4)

if __name__ == "__main__":
    main()
