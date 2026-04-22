#!/bin/bash
if [ $# -ne 4 ]; then
    echo "Usage: bash run_all.sh <source_dir> <dest_dir> <output_dir> <cuda_device>"
    exit 1
fi

source_dir=$1
dest_dir=$2
output_dir=$3
cuda_device=$4

echo "==== Step 1: Generating modification files ===="
CUDA_VISIBLE_DEVICES=$cuda_device python gen_modify.py -s "$source_dir" -d "$dest_dir"
if [ $? -ne 0 ]; then
    echo "Run gen_modify.py failed."
    exit 1
fi

echo "==== Step 2: Extracting images ===="
python get_img.py --dir "$dest_dir" --img "$source_dir"
if [ $? -ne 0 ]; then
    echo "Run get_img.py failed."
    exit 1
fi

echo "==== Step 3: Applying modifications ===="
python modify.py --input "$dest_dir" --output "$output_dir"
if [ $? -ne 0 ]; then
    echo "Run modify.py failed."
    exit 1
fi

echo "All steps completed successfully!"
