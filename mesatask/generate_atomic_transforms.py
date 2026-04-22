#!/usr/bin/env python3
import argparse
import json
import math
import random
from copy import deepcopy
from pathlib import Path
from instruction_templates import TEMPLATES

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = REPO_ROOT / "MesaTask-10K" / "Layout_info"
DEFAULT_ASSET_ANNOTATION = REPO_ROOT / "MesaTask-10K" / "Asset_annotation.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "transformed_layouts5"
DEFAULT_NUM_VARIANTS = 10

DEFAULT_ROTATE_DEGREES = [90, 45, -45, -90, 180, -180]
DEFAULT_SCALE_FACTORS = [0.5, 0.75, 1.5, 2.0]

def parse_args():
    parser = argparse.ArgumentParser(description="Generate atomic transforms for MesaTask layouts")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Layout_info root directory")
    parser.add_argument("--asset-annotation", default=str(DEFAULT_ASSET_ANNOTATION), help="Asset_annotation.json path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--num-variants", type=int, default=DEFAULT_NUM_VARIANTS, help="Variants per scene")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--max-attempts", type=int, default=50, help="Max collision-avoidance attempts")
    parser.add_argument("--iou-threshold", type=float, default=0.0, help="Collision IOU threshold")
    parser.add_argument("--enlarge-min", type=float, default=1.0, help="Min table enlarge factor")
    parser.add_argument("--enlarge-max", type=float, default=2.0, help="Max table enlarge factor")
    parser.add_argument("--move-min-cm", type=float, default=5.0, help="Min move distance (cm)")
    parser.add_argument("--move-max-cm", type=float, default=20.0, help="Max move distance (cm)")
    parser.add_argument("--rotate-degrees", nargs="+", type=int, default=DEFAULT_ROTATE_DEGREES,
                        help="Rotation degrees list")
    parser.add_argument("--scale-factors", nargs="+", type=float, default=DEFAULT_SCALE_FACTORS,
                        help="Scale factors list")
    return parser.parse_args()

def validate_args(args):
    if args.enlarge_min <= 0 or args.enlarge_max <= 0:
        raise ValueError("enlarge-min and enlarge-max must be > 0")
    if args.enlarge_min > args.enlarge_max:
        raise ValueError("enlarge-min must be <= enlarge-max")
    if args.move_min_cm <= 0 or args.move_max_cm <= 0:
        raise ValueError("move-min-cm and move-max-cm must be > 0")
    if args.move_min_cm > args.move_max_cm:
        raise ValueError("move-min-cm must be <= move-max-cm")
    if not args.rotate_degrees:
        raise ValueError("rotate-degrees must not be empty")
    if not args.scale_factors:
        raise ValueError("scale-factors must not be empty")

### -------- 原子操作定义（每次只调用一个） -------- ###

def move_object(obj, table_bounds, asset_info, cfg):
    """随机向一个方向移动 5-20cm"""
    new_obj = deepcopy(obj)
    x, y, z = new_obj["position"]
    xmin, xmax, ymin, ymax = table_bounds

    direction = random.choice(["left", "right", "forward", "backward"])
    delta = random.uniform(cfg["move_min_cm"], cfg["move_max_cm"])  # cm

    caption = asset_info.get("detailed_caption", obj['instance'])
    category = asset_info.get("category", "物体")
    obj_desc = f"{category} ({caption})"

    if direction == "left":
        x = max(x - delta, xmin)
        template_key = "move_left"
    elif direction == "right":
        x = min(x + delta, xmax)
        template_key = "move_right"
    elif direction == "forward":
        y = min(y + delta, ymax)
        template_key = "move_forward"
    else:  # "backward"
        y = max(y - delta, ymin)
        template_key = "move_backward"

    # 生成中英文指令
    template_zh = random.choice(TEMPLATES[template_key]["zh"])
    template_en = random.choice(TEMPLATES[template_key]["en"])
    instr_zh = template_zh.format(obj=obj_desc, value=round(delta))
    instr_en = template_en.format(obj=obj_desc, value=round(delta))

    new_obj["position"] = [x, y, z]
    new_obj["category"] = category
    new_obj["detailed_caption"] = caption

    # 返回操作元数据
    meta = {
        "operation": "move",
        "direction": direction,
        "delta_cm": round(delta, 2),
        "old_position": [x - (x - new_obj["position"][0]), y - (y - new_obj["position"][1]), z]
    }

    return new_obj, instr_zh, instr_en, meta

def rotate_object(obj, _, asset_info, cfg):
    """绕 Z 轴旋转"""
    new_obj = deepcopy(obj)

    deg = random.choice(cfg["rotate_degrees"])
    rad = deg * math.pi / 180.0

    caption = asset_info.get("detailed_caption", obj['instance'])
    category = asset_info.get("category", "物体")
    obj_desc = f"{category} ({caption})"

    if deg < 0:
        template_key = "rotate_clockwise"
        direction = "clockwise"
    else:
        template_key = "rotate_counterclockwise"
        direction = "counterclockwise"

    # 生成中英文指令
    template_zh = random.choice(TEMPLATES[template_key]["zh"])
    template_en = random.choice(TEMPLATES[template_key]["en"])
    instr_zh = template_zh.format(obj=obj_desc, value=abs(deg))
    instr_en = template_en.format(obj=obj_desc, value=abs(deg))

    # 原始 rotation（四元数），默认单位四元数 [0, 0, 0, 1]
    old_rotation = new_obj.get("rotation", [0.0, 0.0, 0.0, 1.0])
    qx, qy, qz, qw = old_rotation

    # 构造绕 Z 轴旋转的四元数
    sin_half = math.sin(rad / 2)
    cos_half = math.cos(rad / 2)
    rz = [0.0, 0.0, sin_half, cos_half]

    # 四元数乘法：q' = rz * q
    nx = rz[3]*qx + rz[0]*qw + rz[1]*qz - rz[2]*qy
    ny = rz[3]*qy - rz[0]*qz + rz[1]*qw + rz[2]*qx
    nz = rz[3]*qz + rz[0]*qy - rz[1]*qx + rz[2]*qw
    nw = rz[3]*qw - rz[0]*qx - rz[1]*qy - rz[2]*qz

    # 更新 rotation 字段
    new_obj["rotation"] = [nx, ny, nz, nw]
    old_z_rotation = new_obj.get("z_rotation", 0.0)
    new_obj["z_rotation"] = old_z_rotation + rad
    new_obj["category"] = category
    new_obj["detailed_caption"] = caption

    # 返回操作元数据
    meta = {
        "operation": "rotate",
        "direction": direction,
        "degrees": deg,
        "radians": round(rad, 4),
        "old_rotation": old_rotation,
        "old_z_rotation": round(old_z_rotation, 4)
    }

    return new_obj, instr_zh, instr_en, meta
def scale_object(obj, _, asset_info, cfg):
    """整体缩放"""
    new_obj = deepcopy(obj)
    factor = random.choice(cfg["scale_factors"])

    old_scale_factor = obj["scale_factor"]
    old_size = obj["size"]

    new_obj["scale_factor"] = [s * factor for s in obj["scale_factor"]]
    new_obj["size"] = [s * factor for s in obj["size"]]

    caption = asset_info.get("detailed_caption", obj['instance'])
    category = asset_info.get("category", "物体")
    obj_desc = f"{category} ({caption})"

    if factor > 1:
        template_key = "scale_up"
        percentage = round((factor - 1) * 100)
        scale_type = "up"
    else:
        template_key = "scale_down"
        percentage = round((1 - factor) * 100)
        scale_type = "down"

    # 生成中英文指令
    template_zh = random.choice(TEMPLATES[template_key]["zh"])
    template_en = random.choice(TEMPLATES[template_key]["en"])
    instr_zh = template_zh.format(obj=obj_desc, value=percentage, scale=factor)
    instr_en = template_en.format(obj=obj_desc, value=percentage, scale=factor)

    new_obj["category"] = category
    new_obj["detailed_caption"] = caption

    # 返回操作元数据
    meta = {
        "operation": "scale",
        "scale_type": scale_type,
        "factor": factor,
        "percentage": percentage,
        "old_scale_factor": old_scale_factor,
        "old_size": old_size
    }

    return new_obj, instr_zh, instr_en, meta

### -------- 执行主流程 -------- ###

def get_3d_bbox(obj):
    """获取物体的3D边界框 [xmin, xmax, ymin, ymax, zmin, zmax]"""
    x, y, z = obj["position"]
    size_x, size_y, size_z = obj["size"]

    return [
        x - size_x / 2,  # xmin
        x + size_x / 2,  # xmax
        y - size_y / 2,  # ymin
        y + size_y / 2,  # ymax
        z - size_z / 2,  # zmin
        z + size_z / 2   # zmax
    ]

def calculate_3d_iou(bbox1, bbox2):
    """计算两个3D边界框的IOU"""
    # bbox格式: [xmin, xmax, ymin, ymax, zmin, zmax]

    # 计算3D交集区域
    inter_xmin = max(bbox1[0], bbox2[0])
    inter_xmax = min(bbox1[1], bbox2[1])
    inter_ymin = max(bbox1[2], bbox2[2])
    inter_ymax = min(bbox1[3], bbox2[3])
    inter_zmin = max(bbox1[4], bbox2[4])
    inter_zmax = min(bbox1[5], bbox2[5])

    # 如果没有交集
    if inter_xmin >= inter_xmax or inter_ymin >= inter_ymax or inter_zmin >= inter_zmax:
        return 0.0

    # 计算交集体积
    inter_volume = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin) * (inter_zmax - inter_zmin)

    # 计算各自体积
    volume1 = (bbox1[1] - bbox1[0]) * (bbox1[3] - bbox1[2]) * (bbox1[5] - bbox1[4])
    volume2 = (bbox2[1] - bbox2[0]) * (bbox2[3] - bbox2[2]) * (bbox2[5] - bbox2[4])

    # 计算并集体积
    union_volume = volume1 + volume2 - inter_volume

    # 返回IOU
    return inter_volume / union_volume if union_volume > 0 else 0.0

def check_collision(new_obj, all_objects, target_idx, iou_threshold=0.0):
    """检查新物体是否与其他物体发生碰撞"""
    new_bbox = get_3d_bbox(new_obj)
    max_iou = 0.0

    for i, obj in enumerate(all_objects):
        if i == target_idx:
            continue
        obj_bbox = get_3d_bbox(obj)
        iou = calculate_3d_iou(new_bbox, obj_bbox)
        max_iou = max(max_iou, iou)
        if iou > iou_threshold:
            return True, max_iou

    return False, max_iou

def get_unique_category_objects(objects):
    """获取类别唯一的物体（避免歧义）"""
    category_count = {}
    for obj in objects:
        category = obj.get("category", "unknown")
        category_count[category] = category_count.get(category, 0) + 1

    # 只返回类别唯一的物体
    unique_objects = []
    for i, obj in enumerate(objects):
        category = obj.get("category", "unknown")
        if category_count[category] == 1:
            unique_objects.append((i, obj))

    return unique_objects

def generate_variant(layout, table_bounds, variant_id, cfg):
    new_layout = deepcopy(layout)
    objects = new_layout["objects"]

    # 只选择类别唯一的物体
    unique_objects = get_unique_category_objects(objects)

    if not unique_objects:
        # 如果没有类别唯一的物体，fallback到随机选择
        idx = random.randint(0, len(objects) - 1)
        target_obj = objects[idx]
    else:
        # 从类别唯一的物体中随机选择
        idx, target_obj = random.choice(unique_objects)

    # 物体已经包含 category 和 detailed_caption 了
    asset_info = {
        "category": target_obj.get("category", "物体"),
        "detailed_caption": target_obj.get("detailed_caption", target_obj['instance'])
    }

    # 尝试多次生成无碰撞的变换
    new_obj = None
    for attempt in range(cfg["max_attempts"]):
        operation = random.choice([move_object, rotate_object, scale_object])
        new_obj, instr_zh, instr_en, meta = operation(target_obj, table_bounds, asset_info, cfg)

        # 检查碰撞
        has_collision, max_iou = check_collision(new_obj, objects, idx, cfg["iou_threshold"])

        if not has_collision:
            # 无碰撞，接受这次变换
            meta["collision_check"] = {
                "passed": True,
                "attempts": attempt + 1,
                "max_iou": round(max_iou, 4)
            }
            break
    else:
        # 达到最大尝试次数，使用最后一次结果但标记碰撞
        meta["collision_check"] = {
            "passed": False,
            "attempts": cfg["max_attempts"],
            "max_iou": round(max_iou, 4),
            "warning": "Max attempts reached, collision may exist"
        }

    new_layout["objects"][idx] = new_obj

    # 完整的操作元数据
    operation_meta = {
        "target_object": {
            "instance": target_obj['instance'],
            "name": target_obj['name'],
            "category": asset_info['category'],
            "caption": asset_info['detailed_caption'],
            "object_index": idx
        },
        **meta
    }

    # 保存到 layout JSON
    new_layout['instruction_zh'] = instr_zh
    new_layout['instruction_en'] = instr_en
    new_layout['operation_meta'] = operation_meta

    return new_layout, instr_zh, instr_en, operation_meta

def enrich_layout_with_annotations(layout, asset_annotations):
    """为 layout 中的所有物体添加 category 和 detailed_caption"""
    category_count = {}

    for obj in layout["objects"]:
        uid = obj.get("retrieved_uid", "")
        asset_info = asset_annotations.get(uid, {})

        category = asset_info.get("category", "unknown")
        caption = asset_info.get("detailed_caption", "No description")

        obj["category"] = category
        obj["detailed_caption"] = caption

        # 统计类别数量
        category_count[category] = category_count.get(category, 0) + 1

    return category_count

def enlarge_table_and_scale_objects(layout, cfg):
    """随机扩大桌面并统一平移所有物体位置"""
    # 随机生成长宽扩大倍数 (1.0 - 2倍)
    scale_x = random.uniform(cfg["enlarge_min"], cfg["enlarge_max"])
    scale_y = random.uniform(cfg["enlarge_min"], cfg["enlarge_max"])

    # 获取原始 placement zone
    xmin, xmax, ymin, ymax = layout["item_placement_zone"]

    # 计算原始尺寸
    orig_width = xmax - xmin
    orig_height = ymax - ymin

    # 扩大 placement zone
    new_width = orig_width * scale_x
    new_height = orig_height * scale_y
    new_xmax = xmin + new_width
    new_ymax = ymin + new_height
    layout["item_placement_zone"] = [xmin, new_xmax, ymin, new_ymax]

    # 计算平移量（将物体整体平移到新区域的中心）
    # 保持物体间相对位置不变
    offset_x = (new_width - orig_width) / 2
    offset_y = (new_height - orig_height) / 2

    # 统一平移所有物体位置
    for obj in layout["objects"]:
        x, y, z = obj["position"]
        obj["position"] = [x + offset_x, y + offset_y, z]

    return scale_x, scale_y

def process_single_scene(layout_path, asset_annotations, scene_output_dir, cfg):
    """处理单个场景"""
    scene_name = layout_path.parent.name
    scene_type = layout_path.parent.parent.name

    print(f"\n{'='*60}")
    print(f"处理场景: {scene_type}/{scene_name}")
    print(f"{'='*60}")

    with open(layout_path, "r", encoding="utf-8") as f:
        layout = json.load(f)

    # 为 layout 添加标注信息并统计类别
    category_count = enrich_layout_with_annotations(layout, asset_annotations)

    print(f"物体统计: {sum(category_count.values())} 个物体")

    # 扩大桌面并平移物体位置
    scale_x, scale_y = enlarge_table_and_scale_objects(layout, cfg)

    # 创建场景专属的输出目录
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    origin_layout_dir = scene_output_dir / "origin_layout"
    origin_layout_dir.mkdir(exist_ok=True)

    # 保存扩大后的原始 layout
    origin_layout_filename = f"layout_enlarged_{scale_x:.2f}x_{scale_y:.2f}y.json"
    origin_layout_path = origin_layout_dir / origin_layout_filename
    with open(origin_layout_path, "w", encoding="utf-8") as f:
        json.dump(layout, f, ensure_ascii=False, indent=2)

    table_bounds = layout["item_placement_zone"]

    # 生成变换
    variants_dir = scene_output_dir / "variants"
    variants_dir.mkdir(exist_ok=True)

    log_path = scene_output_dir / "transform_descriptions.jsonl"
    success_count = 0

    with open(log_path, "w", encoding="utf-8") as log_file:
        for i in range(cfg["num_variants"]):
            try:
                new_layout, instr_zh, instr_en, meta = generate_variant(layout, table_bounds, i+1, cfg)

                # 保存变换后的layout
                variant_filename = f"{i+1:03d}_{meta['target_object']['instance']}_{meta['operation']}.json"
                variant_path = variants_dir / variant_filename

                with open(variant_path, "w", encoding="utf-8") as f:
                    json.dump(new_layout, f, ensure_ascii=False, indent=2)

                # 写入 JSONL
                log_entry = {
                    "file": variant_filename,
                    "instruction_zh": instr_zh,
                    "instruction_en": instr_en,
                    "operation_meta": meta
                }
                log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                success_count += 1

            except Exception as e:
                print(f"  [✗] 变换 {i+1} 失败: {e}")

    print(f"[✓] 成功生成 {success_count}/{cfg['num_variants']} 个变换")
    return success_count

def main():
    args = parse_args()
    validate_args(args)

    if args.seed is not None:
        random.seed(args.seed)

    input_dir = Path(args.input_dir)
    asset_annotation = Path(args.asset_annotation)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not asset_annotation.exists():
        raise FileNotFoundError(f"Asset annotation not found: {asset_annotation}")

    cfg = {
        "num_variants": args.num_variants,
        "max_attempts": args.max_attempts,
        "iou_threshold": args.iou_threshold,
        "enlarge_min": args.enlarge_min,
        "enlarge_max": args.enlarge_max,
        "move_min_cm": args.move_min_cm,
        "move_max_cm": args.move_max_cm,
        "rotate_degrees": args.rotate_degrees,
        "scale_factors": args.scale_factors,
    }

    with open(asset_annotation, "r", encoding="utf-8") as f:
        asset_annotations = json.load(f)

    # 遍历所有场景类型
    scene_types = [d for d in input_dir.iterdir() if d.is_dir()]

    total_scenes = 0
    total_variants = 0

    for scene_type_dir in sorted(scene_types):
        scene_type = scene_type_dir.name
        print(f"\n{'#'*60}")
        print(f"场景类型: {scene_type}")
        print(f"{'#'*60}")

        # 遍历该类型下的所有场景
        scene_dirs = [d for d in scene_type_dir.iterdir() if d.is_dir()]

        for scene_dir in sorted(scene_dirs):
            layout_path = scene_dir / "layout.json"

            if not layout_path.exists():
                print(f"[!] 跳过 {scene_dir.name}: 未找到 layout.json")
                continue

            # 创建输出目录结构: OUTPUT_DIR / scene_type / scene_name
            scene_output_dir = output_dir / scene_type / scene_dir.name

            try:
                variants_count = process_single_scene(layout_path, asset_annotations, scene_output_dir, cfg)
                total_scenes += 1
                total_variants += variants_count
            except Exception as e:
                print(f"[✗] 处理场景失败 {scene_dir.name}: {e}")

    print(f"\n{'='*60}")
    print(f"批量处理完成!")
    print(f"总共处理: {total_scenes} 个场景")
    print(f"总共生成: {total_variants} 个变换")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
