import os
import json
import numpy as np
from glob import glob
import random

class Dataset:
    def __init__(self, data_dir, data_type):
        self.data_dir = data_dir
        self.data_type = data_type
        self.samples = self.load_samples()
    
    def load_samples(self):
        samples = []
        if self.data_type == "ScanNet++":
            for scene in os.listdir(self.data_dir):
                scene_path = os.path.join(self.data_dir, scene)
                anno_json = os.path.join(scene_path, "scans/segments_anno.json")
                pose_intrinsic_json = os.path.join(scene_path, "iphone/pose_intrinsic_imu.json")
                img_dir = os.path.join(scene_path, "iphone/rgb/")
                img_files = glob(os.path.join(img_dir, "*.jpg"))
                for i in range(0, len(img_files), 10):
                    group = img_files[i: i+10]
                    if group:
                        selected_img = random.choice(group)
                        frame_id = os.path.basename(selected_img).split(".")[0].split("_")[-1]
                        samples.append({
                            "img_path": selected_img,
                            "img_id": scene,
                            "frame_id": frame_id,
                            "anno_json": anno_json,
                            "pose_intrinsic_json": pose_intrinsic_json
                        })
                if len(samples) >= 50:
                    break

        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")

        return samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        with open(sample["anno_json"], 'r') as f:
            anno_data = json.load(f)
        with open(sample["pose_intrinsic_json"], 'r') as f:
            pose_intrinsic_data = json.load(f)
        frame_id = sample["frame_id"]
        return {
            "img_path": sample["img_path"],
            "img_id": sample["img_id"],
            "frame_id": frame_id,
            "anno": anno_data,
            "pose": np.array(pose_intrinsic_data[f"frame_{frame_id}"]["aligned_pose"]),
            "intrinsics": np.array(pose_intrinsic_data[f"frame_{frame_id}"]["intrinsic"])
        }