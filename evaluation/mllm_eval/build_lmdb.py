"""
Build EVAL_lmdb_dataset for MLLM-based AC scoring.

This script packages (original image, edited image, prompt) triplets into
LMDB databases that mllm_eval.py can read.

Usage:
    python build_lmdb.py \
        --dataset-dir ../fine_dataset \
        --generated-dir ../eval/BAGEL/generated_images_fine \
        --output-dir ./EVAL_lmdb_dataset/fine_BAGEL \
        --dataset-name fine \
        --model-name BAGEL

It will scan --generated-dir for files matching <img_id>_edit_<query_id>.*,
find the corresponding original image and JSON in --dataset-dir, and write
everything into a single LMDB.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

def find_original_image(dataset_dir: Path, img_id: str):
    """Find the original image in dataset_dir whose name starts with img_id."""
    for ext in (".jpg", ".png", ".jpeg"):
        candidate = dataset_dir / f"{img_id}{ext}"
        if candidate.exists():
            return candidate
    # Fallback: glob
    matches = list(dataset_dir.glob(f"{img_id}.*"))
    for m in matches:
        if m.suffix.lower() in (".jpg", ".png", ".jpeg"):
            return m
    return None


def encode_image(img_path: str) -> bytes:
    """Read image and encode to JPEG bytes (same format mydataset_eval.py expects)."""
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    _, buf = cv2.imencode(".jpg", img)  # noqa: F821
    return buf.tobytes()


def main():
    parser = argparse.ArgumentParser(description="Build LMDB for MLLM AC evaluation")
    parser.add_argument("--dataset-dir", required=True,
                        help="Directory with original images and edit JSONs (e.g. fine_dataset/)")
    parser.add_argument("--generated-dir", required=True,
                        help="Directory with model-generated edited images")
    parser.add_argument("--output-dir", required=True,
                        help="Output LMDB directory path")
    parser.add_argument("--dataset-name", required=True,
                        help="Dataset identifier (e.g. fine, mesatask, bathroom, robothor)")
    parser.add_argument("--model-name", required=True,
                        help="Model identifier (e.g. BAGEL)")
    args = parser.parse_args()

    try:
        import lmdb
    except ImportError:
        print("Error: lmdb is required. Install with: pip install lmdb")
        sys.exit(1)

    dataset_dir = Path(args.dataset_dir)
    generated_dir = Path(args.generated_dir)
    output_dir = Path(args.output_dir)

    if not dataset_dir.is_dir():
        print(f"Error: dataset directory not found: {dataset_dir}")
        sys.exit(1)
    if not generated_dir.is_dir():
        print(f"Error: generated images directory not found: {generated_dir}")
        sys.exit(1)

    # Collect all edited images
    edit_files = sorted([
        f for f in generated_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".png", ".jpeg") and "_edit_" in f.stem
    ])

    if not edit_files:
        print(f"No edited images found in {generated_dir}")
        sys.exit(1)

    print(f"Found {len(edit_files)} edited images in {generated_dir}")

    # Build samples
    samples = []
    skipped = 0
    for edit_path in edit_files:
        # Parse: <img_id>_edit_<query_id>.ext
        stem = edit_path.stem.replace("_rgb", "")  # strip _rgb if present
        if "_edit_" not in stem:
            skipped += 1
            continue

        parts = stem.rsplit("_edit_", 1)
        img_id = parts[0]
        item_id = stem  # e.g. "036bce3393_frame_008270_edit_0"

        # Find original image
        orig_img_path = find_original_image(dataset_dir, img_id)
        if orig_img_path is None:
            print(f"  Warning: original image not found for {img_id}, skipping")
            skipped += 1
            continue

        # Find edit JSON
        json_path = dataset_dir / f"{stem}.json"
        if not json_path.exists():
            # Try with original stem (before _rgb removal)
            json_path = dataset_dir / f"{edit_path.stem}.json"
        if not json_path.exists():
            print(f"  Warning: JSON not found for {stem}, skipping")
            skipped += 1
            continue

        with open(json_path) as f:
            meta = json.load(f)

        prompt = meta.get("prompt", "")
        if not prompt:
            print(f"  Warning: empty prompt in {json_path}, skipping")
            skipped += 1
            continue

        samples.append({
            "orig_img_path": str(orig_img_path),
            "edit_img_path": str(edit_path),
            "prompt": prompt,
            "item_id": item_id,
        })

    print(f"Valid samples: {len(samples)}, skipped: {skipped}")

    if not samples:
        print("No valid samples to write.")
        sys.exit(1)

    # Write LMDB
    output_dir.mkdir(parents=True, exist_ok=True)
    map_size = 50 * 1024 * 1024 * 1024  # 50 GB max

    env = lmdb.open(str(output_dir), map_size=map_size)

    with env.begin(write=True) as txn:
        for i, s in enumerate(samples):
            if i % 100 == 0:
                print(f"  Writing {i}/{len(samples)}...")

            img1_bytes = encode_image(s["orig_img_path"])
            img2_bytes = encode_image(s["edit_img_path"])

            record = {
                "img1": img1_bytes,
                "img2": img2_bytes,
                "prompt": s["prompt"],
                "item_id": s["item_id"],
                "dataset": args.dataset_name,
                "model": args.model_name,
            }

            key = f"{i:08d}".encode("utf-8")
            txn.put(key, pickle.dumps(record))

        # Write metadata
        meta = {"num_samples": len(samples)}
        txn.put(b"__meta__", pickle.dumps(meta))

    env.close()
    print(f"Done. LMDB written to {output_dir} with {len(samples)} samples.")


if __name__ == "__main__":
    main()
