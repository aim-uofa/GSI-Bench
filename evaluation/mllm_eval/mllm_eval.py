import json
import traceback
import time
import os
from collections import defaultdict
from typing import Any, Dict, List
from multiprocessing import Pool
from functools import partial

import torch
from mydataset_eval import DL3DV
from torch.utils.data import DataLoader
from tqdm import tqdm

import requests


PATH = None  # Set via --model-path argument or override here

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
        else:
            tensors[key] = torch.cat(value, dim=0)

    return {**tensors, **non_tensors}


class VLLMModel:
    def __init__(self, base_url="http://localhost:8000", model_name="._pretrained_models"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.headers = {"Content-Type": "application/json"}
        print(f"VLLM client initialized: base_url={base_url}, model={model_name}")

    def _call_vllm_api(self, messages, max_tokens=16384, temperature=0, top_p=None):
        """Call vLLM chat completion API"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=360
            )

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                print(f"vLLM API call failed: {response.status_code}, {response.text}")
                return "_ERROR_"
        except Exception as e:
            print(f"Exception during vLLM API call: {str(e)}")
            return "_ERROR_"

    def infer(self, samples: dict[torch.Tensor | list]):
        """Inference method with the same interface as RemoteModel"""
        try:
            bsz = len(samples["openai"])
            ans = samples

            return_dict = []
            raw_response_text = []

            # Call vLLM API
            try:
                for msg in ans["openai"]:
                    response = self._call_vllm_api(messages=msg)
                    raw_response_text.append(response)
            except Exception as e:
                print(f"vLLM API call error: {e}")
                raw_response_text = ["_ERROR_"] * bsz

            for i in range(bsz):
                single_data = {}
                single_data["prediction"] = raw_response_text[i]
                single_data["meta"] = {
                    "db_idx": ans["db_idx"][i],
                    "index": json.loads(ans["index"][i]),
                    "dataset": ans["dataset"][i],
                    "model": ans["model"][i]
                }
                return_dict.append(single_data)

        except Exception as e:
            print(f"Error during vLLM inference: {e}")
            traceback.print_exc()
            return_dict = []

        print("Returning one batch of results from vLLM...")
        return return_dict


class RemoteModel:
    def __init__(self, model_name):
        from mllm_response import MLLMsResponse  # optional: only needed for commercial API
        self.client = MLLMsResponse(model_name, sk='')
        print("Commercial API client initialized.")

        # Debugging (optional)
        # import debugpy
        # debugpy.listen(5679)
        # print("Waiting for debugger attach...")
        # debugpy.wait_for_client()
        # print("Debugger attached.")

    def infer(self, samples: dict[torch.Tensor | list]):
        try:
            bsz = len(samples["openai"])
            ans = samples

            return_dict = []
            raw_response_text = []
            # Call commercial API
            try:
                for msg in ans["openai"]:
                    responses = self.client.get_chat_response_message(messages=msg)
                    raw_response_text.append(responses)
            except Exception as e:
                print(f"API call error: {e}")
                raw_response_text = ["_ERROR_"] * bsz

            for i in range(bsz):
                single_data = {}
                single_data["prediction"] = raw_response_text[i]
                single_data["meta"] = {
                    "db_idx": ans["db_idx"][i],
                    "index": json.loads(ans["index"][i]),
                }
                return_dict.append(single_data)

        except Exception as e:
            print(f"Error during inference: {e}")
            traceback.print_exc()
            return_dict = []

        print("Returning one batch of results...")
        return return_dict


def prepare_dataloader(data_path, min_pixels=400 * 300, max_pixels=1280 * 720, mode="default"):
    dataset = DL3DV(
        data_path=data_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        mode=mode
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn
    )

    return dataloader


def process_batch(batch_data, model_name, use_vllm=False, vllm_base_url="http://localhost:8000"):
    """Process a single batch (used in multiprocessing)"""
    import os
    process_id = os.getpid()

    try:
        print(f"[Process {process_id}] Start processing batch")
        if use_vllm:
            model = VLLMModel(base_url=vllm_base_url, model_name=model_name)
        else:
            model = RemoteModel(model_name)
        result = model.infer(batch_data)
        print(f"[Process {process_id}] Batch completed, {len(result)} results returned")
        return result
    except Exception as e:
        print(f"[Process {process_id}] Error during processing: {e}")
        return []


def main():
    # ---------- Configuration ----------
    from argparse import ArgumentParser

    argparser = ArgumentParser()
    argparser.add_argument("--model_name", type=str, default='./Qwen3-VL-235B-A22B-Instruct')
    argparser.add_argument("--mode", type=str, default="default")
    args = argparser.parse_args()

    model_name = args.model_name

    # Model backend configuration
    use_vllm = True  # True: use vLLM service; False: use commercial API
    vllm_base_url = "http://0.0.0.0:8000"  # vLLM service address

    if use_vllm:
        model_name = args.model_name
        print(f"Using vLLM service: {vllm_base_url}, model: {model_name}")
    else:
        model_name = 'gpt-5'
        print(f"Using commercial API, model: {model_name}")

    # ---------- Preparation ----------
    root_dir = "./EVAL_lmdb_dataset"

    for dir in os.listdir(root_dir):
        data_path = os.path.join(root_dir, dir)

        dataloader = prepare_dataloader(
            data_path=data_path,
            min_pixels=400 * 300,
            max_pixels=1000 * 1500,
            mode=args.mode
        )

        # ---------- Multiprocessing inference ----------
        all_predictions = []
        first_N = 2000
        num_processes = 15

        batches = []
        for i, batch in enumerate(dataloader):
            if i >= first_N:
                break
            batches.append(batch)

        print(f"Starting multiprocessing for {len(batches)} batches using {num_processes} processes...")
        start_time = time.time()

        with Pool(processes=num_processes) as pool:
            process_func = partial(
                process_batch,
                model_name=model_name,
                use_vllm=use_vllm,
                vllm_base_url=vllm_base_url
            )

            pbar = tqdm(
                total=len(batches),
                desc="Processing",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )

            results = []
            for result in pool.imap(process_func, batches):
                results.append(result)
                pbar.update(1)

                total_processed = sum(len(r) for r in results)
                pbar.set_postfix({
                    "processed_samples": total_processed,
                    "error_batches": sum(1 for r in results if len(r) == 0)
                })

            pbar.close()

        total_results = 0
        error_batches = 0
        for result in results:
            if len(result) == 0:
                error_batches += 1
            else:
                all_predictions.extend(result)
                total_results += len(result)

        elapsed_time = time.time() - start_time
        print("\nProcessing completed!")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Successful batches: {len(batches) - error_batches}/{len(batches)}")
        print(f"Total samples: {total_results}")
        if error_batches > 0:
            print(f"Error batches: {error_batches}")

        print("All predictions collected!")

        dataset = dir.split("_")[-2]
        model = dir.split("_")[-1]

        output_path = f"./infer_results/predictions_infer_{first_N}_{model_name}_{dataset}_{model}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_predictions, f, indent=4)


if __name__ == "__main__":
    main()
