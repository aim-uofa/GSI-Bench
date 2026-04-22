#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./prepare_datasets.sh <zip_dir> [--force]
bash prepare_datasets.sh <DATASET_ROOT>
Purpose:
  - Unzip dataset archives (if needed)
  - Create symlinks in repo root:
      fine_dataset
      mesatask_dataset
      bathroom_dataset
      robothor_dataset

Arguments:
  <zip_dir>  Directory that contains the dataset .zip files and/or extracted folders.
  --force   Replace existing symlinks in repo root.

Notes:
  - Default zip names are <dataset_key>.zip.
  - Default extracted folder names are <dataset_key>.
  - If your zip or folder names differ, edit ZIP_NAMES / EXTRACT_DIRS below.
EOF
}

ZIP_DIR=""
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
      if [[ -z "$ZIP_DIR" ]]; then
        ZIP_DIR="$1"
        shift
      else
        echo "Unknown argument: $1"
        usage
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$ZIP_DIR" ]]; then
  usage
  exit 1
fi

if [[ ! -d "$ZIP_DIR" ]]; then
  echo "Zip dir not found: $ZIP_DIR"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
ZIP_DIR="$(cd "$ZIP_DIR" && pwd)"

DATASETS=(fine_dataset mesatask_dataset bathroom_dataset robothor_dataset)

# Default names (edit as needed). For your current layout:
#   fine_dataset -> ScanNet++/fine_dataset_zmz
#   mesatask_dataset -> mesatask_simulation_dataset
#   bathroom_dataset -> bathroom_simulation_dataset
#   robothor_dataset -> robothor_dataset
declare -A ZIP_NAMES=(
  ["fine_dataset"]="fine_dataset.zip"
  ["mesatask_dataset"]="mesatask_dataset.zip"
  ["bathroom_dataset"]="bathroom_dataset.zip"
  ["robothor_dataset"]="robothor_dataset.zip"
)

declare -A EXTRACT_DIRS=(
  ["fine_dataset"]="ScanNet++/fine_dataset_zmz"
  ["mesatask_dataset"]="mesatask_simulation_dataset"
  ["bathroom_dataset"]="bathroom_simulation_dataset"
  ["robothor_dataset"]="robothor_dataset"
)

extract_zip() {
  local zip_path="$1"
  local dest_dir="$2"
  mkdir -p "$dest_dir"
  if command -v unzip >/dev/null 2>&1; then
    unzip -q "$zip_path" -d "$dest_dir"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 -m zipfile -e "$zip_path" "$dest_dir"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    python -m zipfile -e "$zip_path" "$dest_dir"
    return
  fi
  echo "Neither unzip nor python is available to extract: $zip_path"
  exit 1
}

pick_real_dir() {
  local dir="$1"
  local entries=()
  local item base

  for item in "$dir"/*; do
    [[ -e "$item" ]] || break
    base="$(basename "$item")"
    if [[ "$base" == "__MACOSX" ]]; then
      continue
    fi
    entries+=("$item")
  done

  if [[ ${#entries[@]} -eq 1 && -d "${entries[0]}" ]]; then
    echo "${entries[0]}"
    return
  fi
  echo "$dir"
}

link_dataset() {
  local key="$1"
  local target="$2"
  local link_path="$ROOT_DIR/$key"

  if [[ -L "$link_path" ]]; then
    local cur_target
    cur_target="$(readlink "$link_path")"
    if [[ "$cur_target" == "$target" ]]; then
      echo "Link exists: $link_path -> $target"
      return
    fi
    if [[ "$FORCE" -eq 1 ]]; then
      rm "$link_path"
    else
      echo "Link exists (use --force to replace): $link_path"
      return
    fi
  elif [[ -e "$link_path" ]]; then
    echo "Path exists and is not a symlink (skip): $link_path"
    return
  fi

  ln -s "$target" "$link_path"
  echo "Linked: $link_path -> $target"
}

missing=0

for key in "${DATASETS[@]}"; do
  zip_name="${ZIP_NAMES[$key]}"
  extract_name="${EXTRACT_DIRS[$key]}"
  zip_path="$ZIP_DIR/$zip_name"
  extract_dir="$ZIP_DIR/$extract_name"

  if [[ ! -f "$zip_path" ]]; then
    zip_path_alt="${zip_path%.zip}.ZIP"
    if [[ -f "$zip_path_alt" ]]; then
      zip_path="$zip_path_alt"
    else
      zip_path=""
    fi
  fi

  if [[ -d "$extract_dir" ]]; then
    if [[ -z "$(ls -A "$extract_dir" 2>/dev/null)" && -n "$zip_path" ]]; then
      echo "Extracting (empty dir found): $zip_path"
      extract_zip "$zip_path" "$extract_dir"
    else
      echo "Found extracted dir: $extract_dir"
    fi
  elif [[ -n "$zip_path" ]]; then
    echo "Extracting: $zip_path"
    extract_zip "$zip_path" "$extract_dir"
  else
    echo "Missing zip and extracted dir for: $key"
    missing=1
    continue
  fi

  real_dir="$(pick_real_dir "$extract_dir")"
  if [[ ! -d "$real_dir" ]]; then
    echo "Resolved dataset dir missing for $key: $real_dir"
    missing=1
    continue
  fi
  link_dataset "$key" "$real_dir"
done

if [[ "$missing" -ne 0 ]]; then
  exit 1
fi
