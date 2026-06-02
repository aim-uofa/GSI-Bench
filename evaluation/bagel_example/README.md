# BAGEL reproduction artifacts (download separately)

This folder is **not** included in the Git repository (~265MB). Download and extract here before following [`REPRODUCE_BAGEL_RESULTS.en.md`](../REPRODUCE_BAGEL_RESULTS.en.md) / [`REPRODUCE_BAGEL_RESULTS.md`](../REPRODUCE_BAGEL_RESULTS.md).

## Download

After we publish the archive, use one of:

```bash
cd evaluation

# Option A: Hugging Face CLI (recommended)
pip install -U huggingface_hub
huggingface-cli download aim-uofa/GSI-Bench-bagel-example \
  --local-dir bagel_example --repo-type dataset

# Option B: manual
# Download the release from the project page or Hugging Face dataset page,
# then unzip so that this directory contains fine_dataset/, generated_images_fine/, etc.
```

**Dataset page (update when live):** https://huggingface.co/datasets/aim-uofa/GSI-Bench-bagel-example

## Expected layout

```
bagel_example/
├── fine_dataset/
├── generated_images_fine/
├── BAGEL/fine_eval/
├── predictions_infer_2000_Qwen3-VL-235B-A22B-Instruct_fine_BAGEL.json
└── agg_results/
```

## Verify

```bash
test -d evaluation/bagel_example/fine_dataset && \
test -f evaluation/bagel_example/agg_results/EVAL_output_summary.json && \
echo "bagel_example OK"
```
