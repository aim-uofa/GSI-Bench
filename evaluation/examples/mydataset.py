import os
import json
from PIL import Image

def load_dataset_local(dataset_path):
    from glob import glob

    dataset = []
    json_files = glob(os.path.join(dataset_path, "*.json"))
    for json_file in json_files:
        instruction = ""
        with open(json_file, 'r') as f:
            data = json.load(f)
            instruction = data["prompt"]
        img_file = json_file.split("_edit")[0] + ".png"
        if not os.path.exists(img_file):
            img_file = img_file.replace(".png", ".jpg")
        image = Image.open(img_file).convert("RGB")
        dataset.append({
            "instruction": instruction,
            "input_image": image,
            "task_type": "custom",
            "key": json_file.split('/')[-1].replace('.json', ''),
            "instruction_language": "en",
        })
    return dataset