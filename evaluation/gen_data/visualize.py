import json
import cv2
from pathlib import Path
import numpy as np
from glob import glob
import argparse
import torch
from PIL import Image
import torch.nn.functional as F
import yaml
from box import Box
from copy import deepcopy
from tqdm import tqdm

from utils.train_utils import *
from utils.utils import ResizeLongestSide
from utils.img_utils import *
from utils.box_utils import visualize_bbox

def load_config(config_path="./detect_anything/configs/demo.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg = Box(cfg)
    return cfg

def crop_hw(img):
    if img.dim() == 4:
        img = img.squeeze(0)
    h, w = img.shape[1:3]
    assert max(h, w) % 112 == 0, "target_size must be divisible by 112"
        
    new_h = (h // 14) * 14
    new_w = (w // 14) * 14
        
    center_h, center_w = h // 2, w // 2
    start_h = center_h - new_h // 2
    start_w = center_w - new_w // 2
        
    img_cropped = img[:, start_h:start_h + new_h, start_w:start_w + new_w]
    return img_cropped.unsqueeze(0)

def preprocess(x, cfg):
    """Preprocess image for SAM"""
    sam_pixel_mean = torch.Tensor(cfg.dataset.pixel_mean).view(-1, 1, 1)
    sam_pixel_std = torch.Tensor(cfg.dataset.pixel_std).view(-1, 1, 1)
    x = (x - sam_pixel_mean) / sam_pixel_std
        
    h, w = x.shape[-2:]
    padh = cfg.model.pad - h
    padw = cfg.model.pad - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def visualize(vis_img, data, edit_id, output_path, prefix, show_reference=False, show_orientation=False, show_2d=False):
    if "original_bbox_3d" not in data.keys():
        pass

    rot_mat = np.array(data["rotation_matrix"])
    K = np.array(data["camera_intrinsics"]) if "camera_intrinsics" in data.keys() else np.eye(3)
    visualize_bbox(data["original_bbox_3d"], rot_mat, K, vis_img, mode="Original", show_orientation=show_orientation, show_2d=show_2d)
    
    if "new_bbox_3d" in data.keys():
        if "new_rotation_matrix" in data.keys():
            rot_mat = np.array(data["new_rotation_matrix"])
        visualize_bbox(data["new_bbox_3d"], rot_mat, K, vis_img, mode="Modified", show_orientation=show_orientation, show_2d=show_2d)
            
    if "reference" in data.keys() and show_reference:
        item = data["reference"]
        visualize_bbox(item["bbox_3d"], np.array(item["rotation_matrix"]), K, vis_img, mode="Reference", show_orientation=show_orientation, show_2d=show_2d)
        if "reference_point" in data.keys() and show_reference:
            corner = np.array(data["reference_point"])
            corner_2d = project_to_image(corner.reshape(1, 3), K)
            cv2.circle(vis_img, (int(corner_2d[0][0]), int(corner_2d[0][1])), 5, (0, 255, 255), -1)
            cv2.putText(vis_img, "Ref Point", (int(corner_2d[0][0]), int(corner_2d[0][1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
    output_img_file = output_path / f"{prefix}_edit_{edit_id}_vis.jpg"
    if f"{prefix}_edit_{edit_id}_vis.jpg" == "0a76e06478_frame_000670_edit_5_vis.jpg":
        print(data)
    cv2.imwrite(str(output_img_file), image_cut(vis_img))
    # print(f"Saved visualization to {output_img_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--show_reference", type=bool, default=False)
    parser.add_argument("--show_orientation", type=bool, default=False)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    show_orientation = args.show_orientation
    show_reference = args.show_reference

    modify_jsons = glob(str(input_path / "*_edit*.json"))
    
    cfg = load_config()
    sam_trans = ResizeLongestSide(cfg.model.pad)

    for modify_json in tqdm(modify_jsons, desc="Processing JSON files"):
        with open(modify_json, 'r') as f:
            modify_data = json.load(f)
        prefix = modify_json.split("/")[-1].split("_edit")[0]
        edit_id = modify_json.split("/")[-1].split("_edit")[-1].split(".")[0]
        if edit_id != "":
            edit_id = edit_id[1:]
        # id = modify_json.split("/")[-1].split("_")[0]
        # frame_id = modify_json.split("/")[-1].split("_")[-1].split(".")[0]
        img_file = input_path / f"{prefix}.jpg"
        img = np.array(Image.open(img_file).convert("RGB"))
        image_h, image_w = img.shape[0], img.shape[1]
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0)
        img_tensor = sam_trans.apply_image_torch(img_tensor)
        img_tensor = crop_hw(img_tensor)
        img_for_sam = preprocess(img_tensor, cfg)
        origin_img = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1) * img_for_sam[0, :, :image_h, :image_w].squeeze(0).detach().cpu() + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        vis_img = cv2.cvtColor(origin_img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
        if edit_id != "":
            tmp_vis_img = deepcopy(vis_img)
            visualize(tmp_vis_img, modify_data, edit_id, output_path, prefix, show_reference=show_reference, show_orientation=show_orientation)
        else:
            for key in modify_data.keys():
                tmp_vis_img = deepcopy(vis_img)
                visualize(tmp_vis_img, modify_data[key], key, output_path, prefix, show_reference=show_reference, show_orientation=show_orientation)
            

if __name__ == "__main__":
    main()
