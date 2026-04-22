#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./prepare_weights.sh <weight_root> [--force]

Purpose:
  Create symlinks under ./checkpoints/detany3d/ to the required weights.

Arguments:
  <weight_root>  Root directory that contains the weight files (e.g. GSI-weight).
  --force        Replace existing symlinks.

Expected files (filenames):
  other_exp_ckpt.pth
  sam_vit_h_4b8939.pth
  dinov2_vitl14_pretrain.pth
  GroundingDINO_SwinB_cfg.py
  groundingdino_swinb_cogcoor.pth

Notes:
  The script searches common subpaths under <weight_root>. If not found,
  it will fallback to a filename search.
EOF
}

WEIGHT_ROOT=""
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -f|--force)
      FORCE=1
      shift
      ;;
    *)
      if [[ -z "$WEIGHT_ROOT" ]]; then
        WEIGHT_ROOT="$1"
        shift
      else
        echo "Unknown argument: $1"
        usage
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$WEIGHT_ROOT" ]]; then
  usage
  exit 1
fi

if [[ ! -d "$WEIGHT_ROOT" ]]; then
  echo "Weight root not found: $WEIGHT_ROOT"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHT_ROOT="$(cd "$WEIGHT_ROOT" && pwd)"

WEIGHT_ITEMS=(
  "other_exp_ckpt.pth|checkpoints/detany3d/detany3d_ckpts/other_exp_ckpt.pth|detany3d/detany3d_ckpts/other_exp_ckpt.pth detany3d_ckpts/other_exp_ckpt.pth checkpoints/detany3d/detany3d_ckpts/other_exp_ckpt.pth"
  "sam_vit_h_4b8939.pth|checkpoints/detany3d/sam_ckpts/sam_vit_h_4b8939.pth|detany3d/sam_ckpts/sam_vit_h_4b8939.pth sam_ckpts/sam_vit_h_4b8939.pth checkpoints/detany3d/sam_ckpts/sam_vit_h_4b8939.pth"
  "dinov2_vitl14_pretrain.pth|checkpoints/detany3d/dino_ckpts/dinov2_vitl14_pretrain.pth|detany3d/dino_ckpts/dinov2_vitl14_pretrain.pth dino_ckpts/dinov2_vitl14_pretrain.pth checkpoints/detany3d/dino_ckpts/dinov2_vitl14_pretrain.pth"
  "GroundingDINO_SwinB_cfg.py|GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py|GroundingDINO_SwinB_cfg.py groundingdino/config/GroundingDINO_SwinB_cfg.py"
  "groundingdino_swinb_cogcoor.pth|GroundingDINO/weights/groundingdino_swinb_cogcoor.pth|groundingdino_swinb_cogcoor.pth GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
)

resolve_source() {
  local filename="$1"
  local candidates="$2"
  local cand

  for cand in $candidates; do
    if [[ -f "$WEIGHT_ROOT/$cand" ]]; then
      echo "$WEIGHT_ROOT/$cand"
      return
    fi
  done

  local found
  found="$(find "$WEIGHT_ROOT" -type f -name "$filename" -print -quit 2>/dev/null || true)"
  if [[ -n "$found" ]]; then
    echo "$found"
    return
  fi

  echo ""
}

link_weight() {
  local src="$1"
  local dest_rel="$2"
  local dest="$ROOT_DIR/$dest_rel"
  local dest_dir
  dest_dir="$(dirname "$dest")"
  mkdir -p "$dest_dir"

  if [[ -L "$dest" ]]; then
    local cur_target
    cur_target="$(readlink "$dest")"
    if [[ "$cur_target" == "$src" ]]; then
      echo "Link exists: $dest -> $src"
      return
    fi
    if [[ "$FORCE" -eq 1 ]]; then
      rm "$dest"
    else
      echo "Link exists (use --force to replace): $dest"
      return
    fi
  elif [[ -e "$dest" ]]; then
    echo "Path exists and is not a symlink (skip): $dest"
    return
  fi

  ln -s "$src" "$dest"
  echo "Linked: $dest -> $src"
}

missing=0

for item in "${WEIGHT_ITEMS[@]}"; do
  IFS='|' read -r filename dest_rel candidates <<< "$item"
  src="$(resolve_source "$filename" "$candidates")"
  if [[ -z "$src" ]]; then
    echo "Missing weight: $filename (under $WEIGHT_ROOT)"
    missing=1
    continue
  fi
  link_weight "$src" "$dest_rel"
done

if [[ "$missing" -ne 0 ]]; then
  exit 1
fi
