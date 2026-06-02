# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import copy
import json
import pickle
from collections import defaultdict
from io import BytesIO
from pathlib import Path as pth
from typing import Any, Dict, List, Union
import re

import cv2
import lmdb
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize
from torch.utils.data import Dataset
from prompts import my_prompt


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        if key not in ["pixel_values", "image_grid_thw"]:
            tensors[key] = torch.stack(value, dim=0)
        # else:
        #     tensors[key] = torch.cat(value, dim=0)

    return {**tensors, **non_tensors}


class DL3DV(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        prompt_key="prompt",
        max_prompt_length=2000,
        truncation="error",
        system_prompt=None,
        max_pixels=1920 * 1080,
        min_pixels=1280 * 720,
        max_sample=None,
        mode="default"
    ):
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.dataset_path = pth(data_path).absolute()
        self.mode = mode

        self.datadb = lmdb.open(
            str(self.dataset_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        # Get dataset size
        with self.datadb.begin() as txn:
            meta_value = txn.get(b"__meta__")
            if meta_value is None:
                raise ValueError("No metadata found in LMDB")
            self.meta_data = pickle.loads(meta_value)
            self.num_samples = (
                self.meta_data["num_samples"]
                if max_sample is None
                else min(max_sample, self.meta_data["num_samples"])
            )

        self.system_prompt = (
            "You are a helpful assistant." if system_prompt is None else system_prompt
        )

    def __len__(self):
        return self.num_samples

    def process_answer(self, sample: dict, h_ratio: float = 0.0, w_ratio: float = 0.0):
        matches = sample.get("matches", [])
        assert len(matches) > 0

        for pair in matches:
            x1, x2, y1, y2 = pair["x1"], pair["x2"], pair["y1"], pair["y2"]
            pair["x1"] = int(x1 * w_ratio)
            pair["x2"] = int(x2 * w_ratio)
            pair["y1"] = int(y1 * h_ratio)
            pair["y2"] = int(y2 * h_ratio)
    
    def find_numbers(self, s):
        return re.findall(r'-?\d+\.?\d*', s)

    def create_query(self, sample: dict) -> dict[dict]:
        """Create a query and answer from the last step information.

        Args:
            step (dict): the raw sample of one step in AndroidControl

        Returns:
            dict[dict]: the simplified user query
        """
        text = my_prompt
        query = {
            "role": "user",
            "content": {
                "text": text,
                "images": [sample["img1"], sample["img2"]]  
            },
        }
        return [query]

    def make_openai_messages(self, messages: List[Dict]):
        def encode_image(image: Union[str, Image.Image]) -> str:
            buffer = BytesIO()
            if isinstance(image, str):
                Image.open(image).save(buffer, format="JPEG")
            elif isinstance(image, Image.Image):
                image.save(buffer, format="JPEG")
            else:
                raise ValueError("Image should be a file path or PIL Image object.")
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("utf-8")

        openai_messages = []
        for msg in messages:
            if msg["role"] == "user":
                content = []
                for item in msg["content"]:
                    if item["type"] == "text":
                        content.append(item)
                    elif item["type"] == "image":
                        assert not isinstance(item["image"], list), (
                            "processed messages should not contain list of images"
                        )

                        base64_image = encode_image(item["image"])
                        content.append(
                            {
                                "type": "image_url",
                                "min_pixels": item["min_pixels"],
                                "max_pixels": item["max_pixels"],
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            }
                        )
                openai_messages.append({"role": "user", "content": content})

            elif msg["role"] in ["assistant", "system"]:
                openai_messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                raise ValueError(f"Unknown role {msg['role']}")
        return openai_messages

    def make_messages(self, messages: List[Dict]):
        prompt_messages = []
        total_turn = len(messages)
        for i, msg in enumerate(messages):
            min_pixels = (
                self.history_min_pixels if i < total_turn - 2 else self.min_pixels
            )
            max_pixels = (
                self.history_max_pixels if i < total_turn - 2 else self.max_pixels
            )

            if msg["role"] == "user":
                if msg["content"].get("images", None) is not None:
                    text = msg["content"].get("text", None)
                    content = []

                    for img in msg["content"]["images"]:
                        image = {"type": "image"}
                        image["image"] = img
                        image["min_pixels"] = min_pixels
                        image["max_pixels"] = max_pixels

                        content.append(image)
                    content.append({"type": "text", "text": text})

                    prompt_messages.append({"role": "user", "content": content})

                else:
                    raise ValueError("User message must contain an image.")

            elif msg["role"] == "assistant" or msg["role"] == "system":
                prompt_messages.append(
                    {
                        "role": msg["role"],
                        "content": [{"type": "text", "text": msg["content"]["text"]}],
                    }
                )
            else:
                raise ValueError(f"Unknown role {msg['role']}")

        return prompt_messages

    def __getitem__(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """

        safe_index = index % self.num_samples

        with self.datadb.begin() as txn:
            db_sample = txn.get(f"{safe_index:08d}".encode("utf-8"))

        try:
            sample = pickle.loads(bytes(db_sample))

        except Exception as e:
            raise ValueError(f"Sample {safe_index} not found") from e

        # Decode images
        img1 = Image.fromarray(cv2.cvtColor(cv2.imdecode(
            np.frombuffer(sample["img1"], np.uint8), cv2.IMREAD_COLOR
        ), cv2.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv2.cvtColor(cv2.imdecode(
            np.frombuffer(sample["img2"], np.uint8), cv2.IMREAD_COLOR
        ), cv2.COLOR_BGR2RGB))

        sample_data = {
            "img1": img1,
            "img2": img2,
            "prompt": sample["prompt"],
            "item_id": sample["item_id"],
            "dataset": sample["dataset"],
            "model": sample["model"]
        }

        system = [{"role": "system", "content": {"text": self.system_prompt}}]
        user_query = self.create_query(sample_data)

        messages = system + user_query

        prompt_messages = self.make_messages(messages)
        openai_copy = self.make_openai_messages(copy.deepcopy(prompt_messages))

        image_input, video_input = process_vision_info(
            prompt_messages,
        )

        return {
            "db_idx": safe_index,
            "images": image_input,
            "openai": openai_copy,
            "index": json.dumps(sample_data["item_id"]),
            "dataset": sample_data["dataset"],
            "model": sample_data["model"]
        }
