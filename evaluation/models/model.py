import json
import os
import yaml
import torch
from box import Box
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision.ops import box_convert

from utils.train_utils import *
from utils.wrap_model import WrapModel

# Try to import scipy for rotation conversions, fallback if not available
try:
    import scipy.spatial.transform as spt
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy not available, 6D pose will be limited to rotation matrix only")
    SCIPY_AVAILABLE = False

# Try to import GroundingDINO, if not available, create placeholder
try:
    from groundingdino.util.inference import load_model
    from groundingdino.util.inference import predict as dino_predict
    import groundingdino.datasets.transforms as T
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    # print("GroundingDINO not available, text prompts will be disabled")
    GROUNDING_DINO_AVAILABLE = False
    
    def load_model(config_path, weights_path):
        return None
    
    def dino_predict(model, image, caption, box_threshold, text_threshold):
        return torch.tensor([]), torch.tensor([]), []

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


class DetAny3DInference:
    def __init__(self, config_path='./detect_anything/configs/demo.yaml'):
        """Initialize DetAny3D inference pipeline"""

        config_path = _resolve_config_path(config_path)
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.cfg = Box(self.cfg)
        
        # Disable distributed training
        torch.distributed.is_available = lambda: False
        torch.distributed.is_initialized = lambda: False
        torch.distributed.get_world_size = lambda group=None: 1
        torch.distributed.get_rank = lambda group=None: 0
        
        # Initialize SAM model
        print("Loading SAM model...")
        self.sam_model = WrapModel(self.cfg)
        checkpoint = torch.load(self.cfg.resume, map_location='cuda:0')
        new_model_dict = self.sam_model.state_dict()
        for k, v in new_model_dict.items():
            if k in checkpoint['state_dict'].keys() and checkpoint['state_dict'][k].size() == new_model_dict[k].size():
                new_model_dict[k] = checkpoint['state_dict'][k].detach()
        self.sam_model.load_state_dict(new_model_dict)
        self.sam_model.to('cuda:0')
        self.sam_model.setup()
        self.sam_model.eval()
        self.sam_trans = ResizeLongestSide(self.cfg.model.pad)
        
        # Initialize GroundingDINO model
        self.dino_model = None
        self.dino_available = GROUNDING_DINO_AVAILABLE
        if self.dino_available:
            try:
                print("Loading GroundingDINO model...")
                self.dino_model = load_model(
                    "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                    "GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
                )
                self.dino_model.eval()
                print("GroundingDINO loaded successfully")
            except Exception as e:
                print(f"Failed to load GroundingDINO: {e}")
                self.dino_available = False
        
        self.box_threshold = 0.47
        self.text_threshold = 0.25
        
    def convert_image_for_dino(self, img):
        """Convert image for GroundingDINO processing"""
        if not self.dino_available or self.dino_model is None:
            return img, None
            
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_source = Image.fromarray(img, 'RGB')
        image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)
        return image, image_transformed
    
    def crop_hw(self, img):
        """Crop image to ensure divisible by 14"""
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
    
    def preprocess(self, x):
        """Preprocess image for SAM"""
        sam_pixel_mean = torch.Tensor(self.cfg.dataset.pixel_mean).view(-1, 1, 1)
        sam_pixel_std = torch.Tensor(self.cfg.dataset.pixel_std).view(-1, 1, 1)
        x = (x - sam_pixel_mean) / sam_pixel_std
        
        h, w = x.shape[-2:]
        padh = self.cfg.model.pad - h
        padw = self.cfg.model.pad - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def preprocess_dino(self, x):
        """Preprocess image for GroundingDINO"""
        x = x / 255
        IMAGENET_DATASET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        IMAGENET_DATASET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        x = (x - IMAGENET_DATASET_MEAN) / IMAGENET_DATASET_STD
        return x
    
    def adjust_brightness(self, color, factor=1.5, v_min=0.3):
        """Adjust color brightness in HSV space"""
        r, g, b = color
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        v = max(v, v_min) * factor
        v = min(v, 1.0)
        return colorsys.hsv_to_rgb(h, s, v)
    
    def draw_orientation_arrow(self, img, center_2d, rotation_matrix, K, arrow_length=0.5, color=(0, 255, 0)):
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
            
            # Get object center in 3D (assuming it's available from the calling context)
            # This will be passed from the calling function
            center_3d = getattr(self, '_temp_center_3d', np.array([0, 0, 1]))
            
            # Calculate arrow endpoints in 3D
            forward_end_3d = center_3d + forward_rotated
            up_end_3d = center_3d + up_rotated
            right_end_3d = center_3d + right_rotated
            
            # Project to 2D
            forward_end_2d = project_to_image(forward_end_3d.reshape(1, 3), K.squeeze(0))[0]
            up_end_2d = project_to_image(up_end_3d.reshape(1, 3), K.squeeze(0))[0]
            right_end_2d = project_to_image(right_end_3d.reshape(1, 3), K.squeeze(0))[0]
            
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
    
    def generate_colors(self, img, num_colors=100):
        """Generate bright colors for visualization"""
        pixels = np.array(img).reshape(-1, 3) / 255.0
        brightness = pixels.mean(axis=1)
        prob = brightness / brightness.sum()
        sampled_indices = np.random.choice(pixels.shape[0], min(num_colors, len(pixels)), p=prob, replace=False)
        sampled_colors = pixels[sampled_indices]
        sampled_colors = sorted(sampled_colors, key=lambda c: colorsys.rgb_to_hsv(*c)[2])
        adjusted_colors = [self.adjust_brightness(c, factor=2.0, v_min=0.4) for c in sampled_colors]
        return adjusted_colors
    
    def predict(self, image_path, text_prompts=None, point_coords=None, bbox_coords=None, output_dir="./results", show_orientation=True, gt_camera_intrinsics=None):
        """
        Perform 3D detection inference
        
        Args:
            image_path: Path to input image
            text_prompt: Text description for detection (e.g., "a car")
            point_coords: List of (x, y) coordinates for point prompts
            bbox_coords: List of (x1, y1, x2, y2) for bbox prompts  
            output_dir: Directory to save results
            show_orientation: Whether to draw orientation arrows on visualization
            gt_camera_intrinsics: GT camera intrinsic matrix (3x3) or None to use predicted intrinsics
        """
        
        # Load image
        if isinstance(image_path, str):
            img = np.array(Image.open(image_path).convert('RGB'))
        else:
            img = image_path
            
        original_size = tuple(img.shape[:-1])
        adjusted_colors = self.generate_colors(img)

         # Prepare image for SAM
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0)
        img_tensor = self.sam_trans.apply_image_torch(img_tensor)
        img_tensor = self.crop_hw(img_tensor)
        before_pad_size = tuple(img_tensor.shape[2:])
        
        img_for_sam = self.preprocess(img_tensor).to('cuda:0')
        img_for_dino = self.preprocess_dino(img_tensor).to('cuda:0')
        
        image_h, image_w = int(before_pad_size[0]), int(before_pad_size[1])
        
        if self.cfg.model.vit_pad_mask:
            vit_pad_size = (before_pad_size[0] // self.cfg.model.image_encoder.patch_size, 
                           before_pad_size[1] // self.cfg.model.image_encoder.patch_size)
        else:
            vit_pad_size = (self.cfg.model.pad // self.cfg.model.image_encoder.patch_size, 
                           self.cfg.model.pad // self.cfg.model.image_encoder.patch_size)
        
        # Prepare visualization
        origin_img = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1) * img_for_sam[0, :, :image_h, :image_w].squeeze(0).detach().cpu() + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        vis_img = cv2.cvtColor(origin_img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
        
        # Process prompts
        label_list = []
        bbox_2d_list = []
        point_coords_list = []
        
        results = []
        K = torch.tensor([])
        # Text-based detection using GroundingDINO
        if len(text_prompts) > 0 and self.dino_available and self.dino_model is not None:
            image_source_dino, image_dino = self.convert_image_for_dino(img)
            for text_prompt in text_prompts:
                boxes, logits, phrases = dino_predict(
                    model=self.dino_model,
                    image=image_dino,
                    caption=text_prompt,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                )
                if len(boxes) > 0:
                    h, w, _ = image_source_dino.shape
                    boxes = boxes * torch.Tensor([w, h, w, h])
                    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
                    
                    for i, box in enumerate(xyxy):
                        bbox_2d_list.append(box.to(torch.int).cpu().numpy().tolist())
                        label_list.append(phrases[i] if i < len(phrases) else "Detected")
        
                # Check if we have any prompts
                if len(bbox_2d_list) == 0 and len(point_coords_list) == 0:
                    print(f"Warning: GroundingDINO returned 0 boxes for prompt '{text_prompt}' on {image_path}")
                    # print("No objects detected or provided. Please provide text prompt, point coordinates, or bbox coordinates.")
                    continue
        
                # Prepare input dictionary
                bbox_2d_tensor = torch.tensor(bbox_2d_list)
                bbox_2d_tensor = self.sam_trans.apply_boxes_torch(bbox_2d_tensor, original_size).to(torch.int).to('cuda:0')
                
                input_dict = {
                    "images": img_for_sam,
                    'vit_pad_size': torch.tensor(vit_pad_size).to('cuda:0').unsqueeze(0),
                    "images_shape": torch.Tensor(before_pad_size).to('cuda:0').unsqueeze(0),
                    "image_for_dino": img_for_dino,
                    "boxes_coords": bbox_2d_tensor,
                }
        
                # Run inference
                with torch.no_grad():
                    ret_dict = self.sam_model(input_dict)
        
                # Decode results
                K_pred = ret_dict['pred_K']
                
                # Use GT camera intrinsics if provided, otherwise use predicted intrinsics
                if gt_camera_intrinsics is not None:
                    # Convert GT intrinsics to tensor if needed
                    if isinstance(gt_camera_intrinsics, np.ndarray):
                        gt_camera_intrinsics = torch.from_numpy(gt_camera_intrinsics).float()
                    if len(gt_camera_intrinsics.shape) == 2:
                        gt_camera_intrinsics = gt_camera_intrinsics.unsqueeze(0)  # Add batch dimension
                    
                    K_gt = gt_camera_intrinsics.to(K_pred.device)
                    # print(f"Using GT camera intrinsics: \n{K_gt[0].cpu().numpy()}")
                    
                    # Use decode_bboxes_virtual_to_real with GT intrinsics
                    decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes_virtual_to_real(ret_dict, self.cfg, K_gt, K_pred)
                    
                    # Use GT intrinsics for all subsequent operations
                    K = K_gt
                else:
                    # Use standard decoding with predicted intrinsics
                    decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes(ret_dict, self.cfg, K_pred)
                    K = K_pred
                    # print("Using predicted camera intrinsics")
        
                pose_6d = ret_dict['pred_pose_6d']  # 6D pose representation
                rot_mat = rotation_6d_to_matrix(pose_6d)  # Convert to rotation matrix
                pred_box_ious = ret_dict.get('pred_box_ious', None)
        
                K_np = K.detach().cpu().numpy()

                # Visualize and collect results
                for i in range(len(decoded_bboxes_pred_2d)):
                    x, y, z, w, h, l, yaw = decoded_bboxes_pred_3d[i].detach().cpu().numpy()
                    rot_mat_i = rot_mat[i].detach().cpu().numpy()
                    vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rot_mat_i)
                    vertices_2d = project_to_image(vertices_3d, K_np.squeeze(0))
                    
                    color = adjusted_colors[i % len(adjusted_colors)]
                    color = [min(255, c*255) for c in color]
                    
                    best_j = torch.argmax(pred_box_ious[i]) if pred_box_ious is not None else 0
                    iou_score = pred_box_ious[i][best_j].item() if pred_box_ious is not None else 0.0
                    
                    # Draw 3D bounding box
                    draw_bbox_2d(vis_img, vertices_2d, color=(255, 255, 0), thickness=3)
                    
                    # Draw orientation arrows if requested
                    if show_orientation:
                        # Store center_3d temporarily for arrow drawing
                        center_3d = np.array([x, y, z])
                        self._temp_center_3d = center_3d
                        
                        # Calculate 2D center for arrow origin
                        center_2d = project_to_image(center_3d.reshape(1, 3), K_np.squeeze(0))[0]
                        
                        # Draw orientation arrows
                        self.draw_orientation_arrow(vis_img, center_2d, rot_mat_i, K_np, arrow_length=min(w, h, l)*0.8)
            
                    # Add text label
                    bbox_2d = box_cxcywh_to_xyxy(decoded_bboxes_pred_2d[i]).detach().cpu().numpy()
                    label = label_list[i] if i < len(label_list) else "Object"
                    size_info = [round(c, 2) for c in decoded_bboxes_pred_3d[i][3:6].detach().cpu().numpy()]
                    
                    cv2.putText(vis_img, f"{label} {size_info}", 
                            (int(bbox_2d[0]), int(bbox_2d[1]-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    
                    # Extract 6D pose for this object
                    pose_6d_i = pose_6d[i].detach().cpu().numpy()
            
                    # Base result data
                    result_data = {
                        'label': label,
                        'bbox_3d': [float(x), float(y), float(z), float(w), float(h), float(l), float(yaw)],
                        'bbox_2d': [float(coord) for coord in bbox_2d.tolist()],
                        'iou_score': float(iou_score),
                        'vertices_3d': [[float(coord) for coord in vertex] for vertex in vertices_3d.tolist()],
                        'vertices_2d': [[float(coord) for coord in vertex] for vertex in vertices_2d.tolist()],
                        'camera_intrinsics': [[float(val) for val in row] for row in K_np.squeeze(0).tolist()],
                        # 6D Pose Information - Core
                        'pose_6d': [float(val) for val in pose_6d_i.tolist()],  # Original 6D representation
                        'rotation_matrix': [[float(val) for val in row] for row in rot_mat_i.tolist()],  # 3x3 rotation matrix
                        'position_3d': [float(x), float(y), float(z)],  # 3D position (translation)
                    }
                
                    # Add additional rotation representations if scipy is available
                    if SCIPY_AVAILABLE:
                        try:
                            r = spt.Rotation.from_matrix(rot_mat_i)
                            euler_xyz = r.as_euler('xyz', degrees=True)  # Roll, Pitch, Yaw in degrees
                            quaternion = r.as_quat()  # x, y, z, w format
                            
                            result_data.update({
                                'euler_angles_xyz_deg': [float(val) for val in euler_xyz.tolist()],  # Roll, Pitch, Yaw (degrees)
                                'quaternion_xyzw': [float(val) for val in quaternion.tolist()],  # Quaternion (x,y,z,w)
                            })
                        except Exception as e:
                            print(f"Warning: Failed to convert rotation matrix: {e}")
                    else:
                        result_data.update({
                            'euler_angles_xyz_deg': None,  # Not available without scipy
                            'quaternion_xyzw': None,  # Not available without scipy
                        })
                    results.append(result_data)

        # Save results if output_dir is specified        
        vis_path, json_path = None, None
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename
            if isinstance(image_path, str):
                base_name = Path(image_path).stem
            else:
                base_name = "result"
                
            # Save visualization
            vis_path = os.path.join(output_dir, f"{base_name}_3d_detection.jpg")
            cv2.imwrite(vis_path, vis_img)
            
            # Save result data
            json_path = os.path.join(output_dir, f"{base_name}_results.json")
        
            # Prepare metadata
            metadata = {
                'image_path': str(image_path) if isinstance(image_path, str) else "array_input",
                'text_prompt': text_prompts,
                'point_coords': point_coords,
                'bbox_coords': bbox_coords,
                'used_gt_intrinsics': gt_camera_intrinsics is not None,
                # 'predicted_camera_intrinsics': [[float(val) for val in row] for row in K_pred.detach().cpu().numpy().squeeze(0).tolist()],
                'detections': results
            }
            
            if gt_camera_intrinsics is not None:
                try:
                    metadata['gt_camera_intrinsics'] = [[float(val) for val in row] for row in gt_camera_intrinsics.detach().cpu().numpy().squeeze(0).tolist()]
                    # metadata["gt_camera_intrinsics"] = gt_camera_intrinsics
                except:
                    pass
            
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
            print(f"Results saved to:")
            print(f"  Visualization: {vis_path}")
            print(f"  Data: {json_path}")
            print(f"  Detected {len(results)} objects")
        
        if len(results) == 0:
            print(f"Warning: No detections produced for image: {image_path} (prompts={text_prompts})")
        return {
            'img_shape': vis_img.shape[:2],
            'visualization_path': vis_path,
            'results_path': json_path,
            'detections': results,
            'visualization_image': cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        }
