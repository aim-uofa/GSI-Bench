#!/usr/bin/env python3
"""
整理图像编辑数据集
从 transformed_layouts5 和 vis_final 中提取图片对和指令
"""

import argparse
import json
import shutil
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TRANSFORMED_DIR = REPO_ROOT / "transformed_layouts5"
DEFAULT_VIS_DIR = REPO_ROOT / "dataset" / "vis_final"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dataset" / "image_editing_dataset"
DEFAULT_MIN_IMAGE_SIZE = 10 * 1024  # 10KB，小于此大小认为是全黑图片

def parse_args():
    parser = argparse.ArgumentParser(description="Organize image editing dataset")
    parser.add_argument("--transformed-dir", default=str(DEFAULT_TRANSFORMED_DIR),
                        help="Transformed layouts directory")
    parser.add_argument("--vis-dir", default=str(DEFAULT_VIS_DIR),
                        help="Rendered visualization directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Output dataset directory")
    parser.add_argument("--min-image-size", type=int, default=DEFAULT_MIN_IMAGE_SIZE,
                        help="Minimum image size in bytes (filter black images)")
    parser.add_argument("--scene-types", nargs="*", default=None,
                        help="Scene types to process (default: all)")
    return parser.parse_args()

def get_origin_image_path(vis_scene_dir):
    """获取 origin layout 的 front.jpg 路径"""
    origin_layout_dir = vis_scene_dir / "origin_layout"
    if not origin_layout_dir.exists():
        return None

    # 查找 layout_enlarged_*_* 目录
    for layout_dir in origin_layout_dir.iterdir():
        if layout_dir.is_dir() and layout_dir.name.startswith("layout_enlarged"):
            front_jpg = layout_dir / "origin_layout_output" / "rendered_views" / "front.jpg"
            if front_jpg.exists():
                return front_jpg
    return None

def get_variant_image_path(vis_scene_dir, variant_name):
    """获取 variant 的 front.jpg 路径"""
    variant_dir = vis_scene_dir / "variants" / variant_name
    if not variant_dir.exists():
        return None

    front_jpg = variant_dir / "variants_output" / "rendered_views" / "front.jpg"
    if front_jpg.exists():
        return front_jpg
    return None

def process_scene(scene_type, scene_name, transformed_dir, vis_dir, images_dir, min_image_size):
    """处理单个场景的所有 variants"""
    transformed_scene_dir = transformed_dir / scene_type / scene_name
    vis_scene_dir = vis_dir / scene_type / scene_name

    # 检查目录是否存在
    if not transformed_scene_dir.exists() or not vis_scene_dir.exists():
        return []

    # 获取 origin 图片路径
    origin_img_src = get_origin_image_path(vis_scene_dir)
    if origin_img_src is None:
        print(f"  ⚠️  跳过 {scene_name}: 找不到 origin front.jpg")
        return []

    # 检查 origin 图片是否全黑
    if origin_img_src.stat().st_size < min_image_size:
        print(f"  ⚠️  跳过 {scene_name}: origin 图片太小 ({origin_img_src.stat().st_size} bytes)，可能是全黑")
        return []

    # 复制 origin 图片到输出目录
    origin_img_name = f"{scene_name}_origin.jpg"
    origin_img_dst = images_dir / origin_img_name
    if not origin_img_dst.exists():
        shutil.copy2(origin_img_src, origin_img_dst)

    # 处理所有 variants
    variants_dir = transformed_scene_dir / "variants"
    if not variants_dir.exists():
        return []

    metadata_entries = []
    variant_jsons = sorted(variants_dir.glob("*.json"))

    for variant_json in variant_jsons:
        variant_name = variant_json.stem  # 例如: 001_10_book_1_move

        # 读取 JSON 获取指令
        try:
            with open(variant_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ⚠️  读取 {variant_json.name} 失败: {e}")
            continue

        instruction_zh = data.get("instruction_zh", "")
        instruction_en = data.get("instruction_en", "")
        operation_meta = data.get("operation_meta", {})

        # 获取 variant 图片路径
        variant_img_src = get_variant_image_path(vis_scene_dir, variant_name)
        if variant_img_src is None:
            print(f"  ⚠️  跳过 {variant_name}: 找不到 front.jpg")
            continue

        # 检查 variant 图片是否全黑
        if variant_img_src.stat().st_size < min_image_size:
            print(f"  ⚠️  跳过 {variant_name}: 图片太小 ({variant_img_src.stat().st_size} bytes)，可能是全黑")
            continue

        # 复制 variant 图片到输出目录
        variant_img_name = f"{scene_name}_{variant_name}.jpg"
        variant_img_dst = images_dir / variant_img_name
        if not variant_img_dst.exists():
            shutil.copy2(variant_img_src, variant_img_dst)

        # 添加到 metadata
        entry = {
            "source_image": f"images/{origin_img_name}",
            "target_image": f"images/{variant_img_name}",
            "instruction_zh": instruction_zh,
            "instruction_en": instruction_en,
            "scene_type": scene_type,
            "scene_id": scene_name,
            "variant_id": variant_name,
            "operation_meta": operation_meta
        }
        metadata_entries.append(entry)

        # 保存单独的 JSON 文件
        individual_json_name = f"{scene_name}_{variant_name}.json"
        individual_json_path = images_dir / individual_json_name
        with open(individual_json_path, 'w', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

    return metadata_entries

def main():
    """主函数"""
    args = parse_args()

    transformed_dir = Path(args.transformed_dir)
    vis_dir = Path(args.vis_dir)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("开始整理图像编辑数据集")
    print(f"全黑图片检测阈值: {args.min_image_size / 1024:.0f} KB")
    print("=" * 60)

    all_metadata = []

    # 获取所有场景类型
    all_scene_types = [d.name for d in transformed_dir.iterdir() if d.is_dir()]

    # 根据配置过滤场景类型
    if args.scene_types is None:
        scene_types = all_scene_types
        print(f"处理所有场景类型: {', '.join(scene_types)}")
    else:
        scene_types = [st for st in args.scene_types if st in all_scene_types]
        print(f"只处理指定场景类型: {', '.join(scene_types)}")
        if len(scene_types) < len(args.scene_types):
            missing = set(args.scene_types) - set(scene_types)
            print(f"⚠️  未找到的场景类型: {', '.join(missing)}")

    for scene_type in scene_types:
        print(f"\n处理场景类型: {scene_type}")

        # 获取该类型下的所有场景
        scene_dirs = sorted((transformed_dir / scene_type).iterdir())
        scene_dirs = [d for d in scene_dirs if d.is_dir()]

        for scene_dir in tqdm(scene_dirs, desc=f"  {scene_type}"):
            scene_name = scene_dir.name
            entries = process_scene(scene_type, scene_name, transformed_dir, vis_dir, images_dir, args.min_image_size)
            all_metadata.extend(entries)

    # 保存 metadata JSON
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("整理完成!")
    print(f"总共处理: {len(all_metadata)} 个图片对")
    print(f"输出目录: {output_dir}")
    print(f"  - 图片目录: {images_dir}")
    print(f"    * {len(all_metadata)} 个图片对 (source + target)")
    print(f"    * {len(all_metadata)} 个单独 JSON 文件")
    print(f"  - 总元数据文件: {metadata_file}")
    print("=" * 60)

    # 输出统计信息
    scene_type_counts = {}
    operation_counts = {}
    for entry in all_metadata:
        scene_type = entry["scene_type"]
        scene_type_counts[scene_type] = scene_type_counts.get(scene_type, 0) + 1

        operation = entry["operation_meta"].get("operation", "unknown")
        operation_counts[operation] = operation_counts.get(operation, 0) + 1

    print("\n场景类型统计:")
    for scene_type, count in sorted(scene_type_counts.items()):
        print(f"  {scene_type}: {count}")

    print("\n操作类型统计:")
    for operation, count in sorted(operation_counts.items()):
        print(f"  {operation}: {count}")

if __name__ == "__main__":
    main()
