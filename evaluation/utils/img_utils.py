import cv2
import torch
import numpy as np

from .train_utils import *
from .wrap_model import WrapModel
from .utils import ResizeLongestSide



def is_blurry(image_path, high_freq_ratio=0.1, threshold=0.6):
    """
    Evaluate image sharpness by the proportion of high-frequency energy in the frequency domain.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Fourier transform
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    h, w = img.shape
    crow, ccol = h // 2, w // 2
    radius = int(min(h, w) * high_freq_ratio)

    # Use np.zeros() to explicitly specify dtype and shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)

    # High-frequency region (mask==0)
    high_freq = magnitude_spectrum * (1 - mask)

    total_energy = np.sum(magnitude_spectrum)
    high_energy = np.sum(high_freq)
    score = high_energy / total_energy

    is_blurry = score < threshold
    return is_blurry, score

def image_cut(img):
    h = img.shape[1]
    return img[:h-220, ...]

def draw_orientation_arrow(img, center_3d, rotation_matrix, K, arrow_length=0.5, color=(0, 255, 0)):
        """Draw orientation arrow showing object's forward direction"""
        try:
            # Define forward direction in object coordinate system (usually +X or +Z axis)
            # We'll use +X axis as forward direction
            forward_3d = np.array([arrow_length, 0, 0])  # Arrow pointing in +X direction
            up_3d = np.array([0, -arrow_length*0.5, 0])  # Smaller arrow pointing in -Y direction
            right_3d = np.array([0, 0, arrow_length*0.5])  # Smaller arrow pointing in +Z direction
            
            # Transform directions by rotation matrix
            forward_rotated = rotation_matrix @ forward_3d
            up_rotated = rotation_matrix @ up_3d  
            right_rotated = rotation_matrix @ right_3d
            
            center_2d = project_to_image(center_3d.reshape(1, 3), K)[0]
            # Calculate arrow endpoints in 3D
            forward_end_3d = center_3d + forward_rotated
            up_end_3d = center_3d + up_rotated
            right_end_3d = center_3d + right_rotated
            
            # Project to 2D
            forward_end_2d = project_to_image(forward_end_3d.reshape(1, 3), K)[0]
            up_end_2d = project_to_image(up_end_3d.reshape(1, 3), K)[0]
            right_end_2d = project_to_image(right_end_3d.reshape(1, 3), K)[0]
            
            # Draw arrows with different colors
            # Forward direction (main arrow) - Green
            cv2.arrowedLine(img, tuple(center_2d.astype(int)), tuple(forward_end_2d.astype(int)), 
                           (0, 255, 0), 3, tipLength=0.3)
            cv2.putText(img, 'X', tuple(forward_end_2d.astype(int) + [10, 0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Up direction - Red  
            cv2.arrowedLine(img, tuple(center_2d.astype(int)), tuple(up_end_2d.astype(int)), 
                           (0, 0, 255), 2, tipLength=0.3)
            cv2.putText(img, 'Y', tuple(up_end_2d.astype(int) + [10, 0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Right direction - Blue
            cv2.arrowedLine(img, tuple(center_2d.astype(int)), tuple(right_end_2d.astype(int)), 
                           (255, 0, 0), 2, tipLength=0.3)
            cv2.putText(img, 'Z', tuple(right_end_2d.astype(int) + [10, 0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                           
        except Exception as e:
            print(f"Failed to draw orientation arrow: {e}")
            # Fallback: draw a simple forward arrow
            try:
                forward_2d = center_2d + np.array([50, 0])  # Simple horizontal arrow
                cv2.arrowedLine(img, tuple(center_2d.astype(int)), tuple(forward_2d.astype(int)), 
                               color, 2, tipLength=0.3)
            except:
                pass

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

def img_process(img, cfg):
    sam_trans = ResizeLongestSide(cfg.model.pad)

    image_h, image_w = img.shape[0], img.shape[1]
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0)
    img_tensor = sam_trans.apply_image_torch(img_tensor)
    img_tensor = crop_hw(img_tensor)
    img_for_sam = preprocess(img_tensor, cfg)
    origin_img = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1) * img_for_sam[0, :, :image_h, :image_w].squeeze(0).detach().cpu() + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    vis_img = cv2.cvtColor(origin_img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    return vis_img

def update_mask_3d(mask, bbox_3d, rot_mat, K):
    x, y, z, w, h, l, yaw = bbox_3d
    vertices_3d, _ = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rot_mat)
    vertices_2d = project_to_image(vertices_3d, K)
    bbox_2d = [np.min(vertices_2d[:, 0]), np.min(vertices_2d[:, 1]), np.max(vertices_2d[:, 0]), np.max(vertices_2d[:, 1])]
    # cv2.fillPoly(mask, [vertices_2d.astype(np.int32)], 0)
    # print("bbox_2d:", bbox_2d)
    try:
        x_min, y_min, x_max, y_max = map(int, bbox_2d)
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 0, thickness=-1)
    except Exception as e:
        pass
    # return mask

def update_mask_2d(sz, mask, bbox_2d):
    x1, y1, x2, y2 = bbox_2d
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(sz[1], x2))
    y2 = int(min(sz[0], y2))
    mask[y1:y2, x1:x2] = 1


