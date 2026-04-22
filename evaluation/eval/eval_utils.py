import sys
import os
import re
import logging
from typing import Literal
from glob import glob
from skimage.metrics import structural_similarity as ssim
import numpy as np
import lpips
import torch
import torch.nn.functional as F
import yaml
from box import Box
import cv2
from copy import deepcopy
# from groundingdino.util.inference import predict as dino_predict
from torchvision.ops import box_convert
import warnings
from utils.item_utils import check_same
from pathlib import Path

def _resolve_config_path(config_path: str) -> str:
    path = Path(config_path)
    if path.is_file():
        return str(path)

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "utils" / "detect_anything" / "configs" / "demo.yaml",
        repo_root / "detect_anything" / "configs" / "demo.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(f"Config file not found: {config_path}")


def load_config(config_path="./detect_anything/configs/demo.yaml"):
    config_path = _resolve_config_path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg = Box(cfg)
    return cfg

def get_img_data(img_file, origin_dir):
    """Get the original image file and editing json file based on the edited image file path."""
    # the edited image file
    # TODO: avoid altering the on-disk path when stripping "_rgb"; keep path vs. logical name separate.
    img_file = img_file.replace("_rgb", "")

    # get img_id and query_id
    img_id = os.path.basename(img_file).split("_edit_")[0]
    try:
        query_id = os.path.basename(img_file).split("_edit_")[1].split(".")[0]
    except:
        raise ValueError(f"Image file name format incorrect: {img_file}")
    # the editing json file
    json_file = re.sub(r'\.(jpg|jpeg|png|bmp|gif)$', '.json', img_file, flags=re.IGNORECASE)
    json_file = os.path.join(origin_dir, os.path.basename(json_file))

    # the original image file
    # Prefer exact stem match (e.g. "..._frame_007520.jpg"), and never use *_edit_*_vis files.
    exact_candidates = [
        os.path.join(origin_dir, f"{img_id}.jpg"),
        os.path.join(origin_dir, f"{img_id}.jpeg"),
        os.path.join(origin_dir, f"{img_id}.png"),
    ]
    original_img_file = next((p for p in exact_candidates if os.path.exists(p)), None)

    if original_img_file is None:
        fuzzy_candidates = glob(os.path.join(origin_dir, f"{img_id}*.jpg")) + \
                           glob(os.path.join(origin_dir, f"{img_id}*.jpeg")) + \
                           glob(os.path.join(origin_dir, f"{img_id}*.png"))
        fuzzy_candidates = [
            p for p in fuzzy_candidates
            if "_edit_" not in os.path.basename(p)
        ]
        if len(fuzzy_candidates) == 0:
            raise FileNotFoundError(f"Original image file not found for {img_file} in {origin_dir}")
        original_img_file = sorted(fuzzy_candidates)[0]

    # the ground-truth editing image file
    gt_edited_img_file = glob(os.path.join(origin_dir, f"{img_id}_*gtedit_{query_id}*.*"))
    if len(gt_edited_img_file) > 0:
        gt_edited_img_file = gt_edited_img_file[0]
    else:
        gt_edited_img_file = None

    # pack into a dict
    data_dict = {
        "meta": {
            "img_id": img_id,
            "query_id": query_id
        },
        "original_img": original_img_file,
        "edited_img": img_file,
        "gt_edited_img": gt_edited_img_file,
        "edit_json": json_file
    }
    # check whether the files exist
    for key, value in data_dict.items():
        # print(key, value)
        if value is not None and (key != "meta" and not os.path.exists(value)):
            raise FileNotFoundError(f"{key} file not found: {value}")

    return data_dict

def get_operation(edit_meta, operation_dict={}):
    """Extract the operation type from the editing prompt."""
    assert "prompt" in edit_meta, "edit_meta must contain 'prompt' key."
    operation = edit_meta["prompt"].split(" ")[0].lower()
    if operation == "please":
        operation = edit_meta["prompt"].split(" ")[1].lower()
    for op_key in operation_dict.keys():
        if operation in operation_dict[op_key]:
            operation = op_key
            break
    if "camera" in edit_meta["prompt"] and "from the camera" not in edit_meta["prompt"] and "to the camera" not in edit_meta["prompt"]:
        operation = "view"
    return operation

def load_detection(infer_cache, cache_key, img_path, edit_meta, inference, restype: Literal["origin_result", "new_result"]):
    """Load detection results from infer_cache or run inference online."""
    try:
        # try to load from cache
        res = infer_cache[cache_key][restype]
        if os.getenv("GSI_DEBUG", "").lower() in ("1", "true", "yes"):
            det_count = len(res.get("detections", [])) if isinstance(res, dict) else 0
            print(f"[GSI_DEBUG] cache hit {restype}: {img_path} det={det_count}")
    except:
        # run inference online otherwise
        print(f"Running inference for image: {img_path}")
        res = inference.predict(
            img_path,
            text_prompts=[edit_meta["target"]],
            point_coords=None,
            bbox_coords=None,
            output_dir=None,
            show_orientation=False,
            gt_camera_intrinsics=np.array(edit_meta["camera_intrinsics"])
        )
        det_count = len(res.get("detections", [])) if isinstance(res, dict) else 0
        if det_count == 0:
            logging.warning(f"Inference produced 0 detections for {img_path} (target={edit_meta.get('target')})")
        if cache_key not in infer_cache:
            infer_cache[cache_key] = {}
        infer_cache[cache_key][restype] = {
            "detections": res["detections"]
        }
    return res

def get_origin_bbox_2d(edit_meta, origin_res):
    """Get the original 2D bounding box from origin_res based on edit_meta."""
    origin_bbox_2d = None
    # TODO: strict 3D equality can miss the original bbox under detector noise; consider a looser match.
    for origin_detection in origin_res["detections"]:
        if check_same(origin_detection["bbox_3d"], edit_meta["original_bbox_3d"]):
            origin_bbox_2d = np.array(origin_detection["bbox_2d"])
            break
    return origin_bbox_2d

def masked_mse(img1, img2, mask=None):
    """
    img1, img2: numpy arrays or torch tensors, shape (H, W) or (H, W, C)
    mask: numpy array or torch tensor, shape (H, W), values 0/1 or bool
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if mask is not None:
        diff = (img1 - img2) ** 2
        mse = diff[mask > 0].mean()
    else:
        mse = ((img1 - img2) ** 2).mean()
    return mse

def masked_ssim(img1, img2, mask=None):
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    # Compute SSIM and retrieve the full SSIM map
    ssim_score, ssim_map = ssim(
        img1,
        img2,
        data_range=img2.max() - img2.min(),
        channel_axis=-1,
        full=True
    )

    if mask is None:
        return ssim_score

    # Ensure mask shape matches (H, W)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.astype(bool)

    # Only average SSIM over masked region
    masked_score = ssim_map[mask].mean()
    return masked_score

class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def masked_lpips(img1, img2, mask=None, net='alex'):
    """
    net: 'alex', 'vgg', 'squeeze'
    img1, img2: numpy arrays or torch tensors
    mask: numpy array or torch tensor
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    # if mask is not None and isinstance(mask, torch.Tensor):
    #     mask = mask.detach().cpu().numpy()
    if mask is not None and isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    with SuppressStdout():
        loss_fn = lpips.LPIPS(net=net, spatial=True)
    def preprocess(x):
        if x.max() > 1.0:
            x = x / 255.0
        x = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        return x

    img1_t = preprocess(img1)
    img2_t = preprocess(img2)

    # TODO: LPIPS returns a scalar by default; current masking has no effect. Use spatial LPIPS or mask inputs.
    if mask is not None:
        # mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        lpips_map = loss_fn.forward(img1_t, img2_t, normalize=False)
        mask_resized = F.interpolate(mask.float(), size=lpips_map.shape[-2:], mode='nearest')
        masked_score = (lpips_map * mask_resized).sum() / mask_resized.sum().clamp(min=1e-6)
        return masked_score.item()
    else:
        score = loss_fn.forward(img1_t, img2_t, normalize=False)
        return score.item()

def check_locality(ssim, lpips, dataset):
    if dataset == "GSI-Real":
        return ssim > 0.6 or lpips < 0.3
    else:
        return ssim > 0.6 and lpips < 0.3

def check_viewpoint_similarity(img1, img2, angle_thresh=10.0):
    # TODO: guard for des1/des2 None and findEssentialMat failures to avoid crashing evaluation.
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)

    # Lowe ratio test
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    E, mask = cv2.findEssentialMat(pts1, pts2, focal=1000.0, pp=(img1.shape[1]/2, img1.shape[0]/2),
                                   method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2)

    angle_diff = np.degrees(np.arccos((np.trace(R) - 1) / 2))
    return angle_diff < angle_thresh

# def sam_img(model, image, target):
#     image_source_dino, image_dino = model.convert_image_for_dino(image)
#     boxes, logits, phrses = dino_predict(
#         model=model.dino_model,
#         image=image_dino,
#         caption=target,
#         box_threshold=model.box_threshold,
#         text_threshold=model.text_threshold
#     )
#     bbox_2d_list = []
#     if len(boxes) > 0:
#         h, w, _ = image_source_dino.shape
#         boxes = boxes * torch.Tensor([w, h, w, h])
#         xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy"
#         )
#         for i, box in enumerate(xyxy):
#             bbox_2d_list.append(box.to(torch.int).cpu().numpy().tolist())
#     if len(bbox_2d_list) == 0:
#         return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
#     img = image
#     if True:
#         original_size = tuple(img.shape[:-1])
#         adjusted_colors = model.generate_colors(img)

#          # Prepare image for SAM
#         img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0)
#         img_tensor = model.sam_trans.apply_image_torch(img_tensor)
#         img_tensor = model.crop_hw(img_tensor)
#         before_pad_size = tuple(img_tensor.shape[2:])
        
#         img_for_sam = model.preprocess(img_tensor).to('cuda:0')
#         img_for_dino = model.preprocess_dino(img_tensor).to('cuda:0')
        
#         image_h, image_w = int(before_pad_size[0]), int(before_pad_size[1])
        
#         if model.cfg.model.vit_pad_mask:
#             vit_pad_size = (before_pad_size[0] // model.cfg.model.image_encoder.patch_size, 
#                            before_pad_size[1] // model.cfg.model.image_encoder.patch_size)
#         else:
#             vit_pad_size = (model.cfg.model.pad // model.cfg.model.image_encoder.patch_size, 
#                            model.cfg.model.pad // model.cfg.model.image_encoder.patch_size)
        
#         # Prepare visualization
#         origin_img = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1) * img_for_sam[0, :, :image_h, :image_w].squeeze(0).detach().cpu() + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
#         vis_img = cv2.cvtColor(origin_img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)


#     bbox_2d_tensor = torch.tensor(bbox_2d_list)
#     bbox_2d_tensor = model.sam_trans.apply_boxes_torch(bbox_2d_tensor, original_size).to(torch.int).to('cuda:0')
#     input_dict = {
#         "images": img_for_sam,
#         "vit_pad_size": torch.tensor(vit_pad_size).to('cuda:0').unsqueeze(0),
#         "images_shape": torch.tensor(before_pad_size).to('cuda:0').unsqueeze(0),
#         "image_for_dino": img_for_dino,
#         "boxes_coords": bbox_2d_tensor
#     }
#     with torch.no_grad():
#         ret_dict = model.sam_model(input_dict)
#     mask1 = ret_dict["masks"][0, 0].cpu().numpy()
#     binary_mask = (mask1 > 0).astype(np.uint8)
#     return binary_mask
