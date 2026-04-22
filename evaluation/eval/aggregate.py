import argparse
import json
import os
import warnings
from glob import glob
from typing import Dict, Iterable, List, Optional, Tuple


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def normalize_dataset_name(dataset_dir: str) -> str:
    if dataset_dir.endswith("_eval"):
        return dataset_dir[:-5]
    return dataset_dir


def normalize_model_name(model_dir: str) -> str:
    return model_dir.replace("_", "")


def find_ac_path(pattern: str, dataset: str, model: str) -> Optional[str]:
    matches = glob(pattern.format(dataset=dataset, model=model))
    if not matches:
        return None
    if len(matches) > 1:
        warnings.warn(
            f"Multiple AC result files found for dataset {dataset} and model {model}, "
            f"using the first match: {matches[0]}",
            RuntimeWarning,
        )
    return matches[0]


def parse_ac_scores(ac_data: Iterable[dict]) -> Dict[str, Optional[float]]:
    ac_result: Dict[str, Optional[float]] = {}
    for item in ac_data:
        index = item["meta"]["index"]
        pred_str = item["prediction"]
        try:
            score = json.loads(pred_str.split("||V^=^V||")[1])["score"]
        except Exception:
            score = None
        ac_result[index] = score
    return ac_result


def resolve_ac_key(key: str, ac_result: Dict[str, Optional[float]]) -> Optional[str]:
    query_id = key.split("_")[-1]
    new_key = key.rsplit(query_id, 1)[0] + f"edit_{query_id}"
    if new_key in ac_result:
        return new_key
    if f"{new_key}.jpg" in ac_result:
        return f"{new_key}.jpg"
    if f"{new_key}.png" in ac_result:
        return f"{new_key}.png"
    return None


def compute_metrics(
    ic_data: dict,
    sa_data: dict,
    el_data: dict,
    ac_result: Dict[str, Optional[float]],
) -> Tuple[float, float, float, float, float, float]:
    filtered_ic: List[float] = []
    filtered_sa: List[float] = []
    filtered_el_ssim: List[float] = []
    filtered_el_lpips: List[float] = []
    filtered_ac: List[float] = []

    for key in ic_data:
        filtered_ic.append(ic_data[key]["compliance"])
        filtered_sa.append(sa_data[key]["edit_score"])
        if ic_data[key]["compliance"] > 0:
            if key in el_data:
                filtered_el_ssim.append(100 * el_data[key]["ssim"])
                filtered_el_lpips.append(100 - 100 * el_data[key]["lpips"])
            else:
                print(f"EL key not found: {key}")
                filtered_el_ssim.append(0)
                filtered_el_lpips.append(0)

            ac_key = resolve_ac_key(key, ac_result)
            if ac_key is not None:
                try:
                    filtered_ac.append(10 * ac_result[ac_key])
                except Exception:
                    print(f"Error converting AC score to float for key: {ac_key}")
                    filtered_ac.append(0)
            else:
                print(f"AC key not found: {key}")
                filtered_ac.append(0)

            if filtered_ac[-1] == 0:
                filtered_ic[-1] = 0
                filtered_sa[-1] = 0
                filtered_el_ssim[-1] = 0
                filtered_el_lpips[-1] = 0
        else:
            filtered_el_ssim.append(0)
            filtered_el_lpips.append(0)
            filtered_ac.append(0)

    el_ssim_mean = sum(filtered_el_ssim) / len(filtered_el_ssim) if filtered_el_ssim else 0
    el_lpips_mean = sum(filtered_el_lpips) / len(filtered_el_lpips) if filtered_el_lpips else 0
    ac_mean = sum(filtered_ac) / len(filtered_ac) if filtered_ac else 0
    ic_mean = sum(filtered_ic) / len(filtered_ic) if filtered_ic else 0
    sa_mean = sum(filtered_sa) / len(filtered_sa) if filtered_sa else 0
    ic_mean *= 100
    sa_mean *= 100
    avg_score = (ic_mean + sa_mean + el_lpips_mean + ac_mean) / 4
    return el_ssim_mean, el_lpips_mean, ac_mean, ic_mean, sa_mean, avg_score


def should_process_dataset(dataset_dir: str) -> bool:
    return dataset_dir.endswith("_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation metrics (IC/SA/EL/AC) into summary JSON files."
    )
    parser.add_argument("--root-dir", default="./eval/", help="Root folder containing model directories.")
    parser.add_argument("--output-dir", default="./EVAL_together", help="Directory to write output JSON files.")
    parser.add_argument(
        "--mllm-eval-dir",
        default="./infer_results_new",
        help="Directory containing MLLM AC result files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = args.root_dir
    output_dir = args.output_dir
    output_el_file = os.path.join(output_dir, "EVAL_output_edit_locality.json")
    output_ac_file = os.path.join(output_dir, "EVAL_output_ac.json")
    output_ic_file = os.path.join(output_dir, "EVAL_output_instruction_compliance.json")
    output_sa_file = os.path.join(output_dir, "EVAL_output_spatial_accuracy.json")
    output_avg_file = os.path.join(output_dir, "EVAL_output_average.json")
    output_summary_file = os.path.join(output_dir, "EVAL_output_summary.json")

    el_dict: Dict[str, dict] = {}
    ac_dict: Dict[str, dict] = {}
    ic_dict: Dict[str, dict] = {}
    sa_dict: Dict[str, dict] = {}
    avg_dict: Dict[str, dict] = {}
    summary_dict: Dict[str, dict] = {}

    for model_dir_name in os.listdir(root_dir):
        model_dir = os.path.join(root_dir, model_dir_name)
        if not os.path.isdir(model_dir):
            continue

        for dataset_dir_name in os.listdir(model_dir):
            if not should_process_dataset(dataset_dir_name):
                continue

            dataset_eval_dir = os.path.join(model_dir, dataset_dir_name)
            ic_json_path = os.path.join(dataset_eval_dir, "instruction-compliance_eval_results.json")
            sa_json_path = os.path.join(dataset_eval_dir, "spatial-accuracy_eval_results.json")
            el_json_path = os.path.join(dataset_eval_dir, "edit-locality_eval_results.json")

            dataset = normalize_dataset_name(dataset_dir_name)
            model = normalize_model_name(model_dir_name)

            ac_glob = os.path.join(args.mllm_eval_dir, "*{dataset}_{model}.json")
            ac_json_path = find_ac_path(ac_glob, dataset, model)
            if ac_json_path is None:
                warnings.warn(
                    f"AC result file not found for dataset {dataset} and model {model}",
                    RuntimeWarning,
                )
                continue

            required_files = [ic_json_path, sa_json_path, el_json_path, ac_json_path]
            if not all(os.path.exists(p) for p in required_files):
                continue

            ic_data = load_json(ic_json_path)
            sa_data = load_json(sa_json_path)
            el_data = load_json(el_json_path)
            ac_data = load_json(ac_json_path)

            ac_result = parse_ac_scores(ac_data)
            el_ssim_mean, el_lpips_mean, ac_mean, ic_mean, sa_mean, avg_score = compute_metrics(
                ic_data, sa_data, el_data, ac_result
            )

            el_dict.setdefault(dataset, {})[model] = {"ssim": el_ssim_mean, "lpips": el_lpips_mean}
            ac_dict.setdefault(dataset, {})[model] = ac_mean
            ic_dict.setdefault(dataset, {})[model] = ic_mean
            sa_dict.setdefault(dataset, {})[model] = sa_mean
            avg_dict.setdefault(dataset, {})[model] = avg_score
            summary_dict.setdefault(dataset, {})[model] = {
                "instruction_compliance": ic_mean,
                "spatial_accuracy": sa_mean,
                "edit_locality": {
                    "ssim": el_ssim_mean,
                    "lpips": el_lpips_mean,
                },
                "ac": ac_mean,
                "average": avg_score,
            }

    write_json(output_el_file, el_dict)
    write_json(output_ac_file, ac_dict)
    write_json(output_ic_file, ic_dict)
    write_json(output_sa_file, sa_dict)
    write_json(output_avg_file, avg_dict)
    write_json(output_summary_file, summary_dict)


if __name__ == "__main__":
    main()
