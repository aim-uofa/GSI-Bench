#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RoboTHOR environment test script.
Author: zmz @ Zhejiang University
"""
import os
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

# Import utility functions from robothor_utils
from robothor_utils import (
    set_render_quality,
    generate_room_views,
    count_visible_interactables,
    _pos_dict,
    _dist3,
    save_current_view,
    )


from action_utils.camera_relative import generate_camera_relative_commands, execute_camera_relative_command, get_object_by_id
from action_utils.object_relative import generate_object_relative_commands, execute_object_relative_command
from action_utils.rotate import generate_rotate_commands, execute_rotate_command
from action_utils.receptacle_placement import generate_receptacle_placement_commands, execute_receptacle_placement_command
from action_utils.agent_camera import generate_agent_camera_commands, execute_agent_camera_command
from action_utils.spatial_remove import generate_spatial_remove_commands, execute_spatial_remove_command
from action_utils.utils_record import save_current_view, append_jsonl, get_top_down_frame, get_camera_info
from action_utils.spawn_utils import save_all_receptacle_spawn_plot
from action_utils.resume_utils import is_scene_completed
from action_utils.object_utils import get_visible_interactables, reset_object_position
from action_utils.environment_utils import restore_view, apply_physics_settings
from action_utils.config_utils import (
    create_default_config,
    save_config,
    parse_arguments,
    merge_config
)

import json, math, random, time


# def get_visible_interactables(controller, pickup_only=False, max_objects=6):
#     objs = controller.last_event.metadata.get("objects", [])
#     res = []
#     for o in objs:
#         if not o.get("visible"): 
#             continue
#         if pickup_only and not o.get("pickupable", False):
#             continue
#         if (o.get("pickupable", False) or o.get("moveable", False)):
#             res.append(o)
#     # 适度限制数量，避免一次生成过多指令
#     random.shuffle(res)
#     return res[:max_objects]



# =========================
# Image & logging helpers
# =========================
# def save_current_view(controller, out_dir, prefix):
#     os.makedirs(out_dir, exist_ok=True)
#     event = controller.last_event
#     Image.fromarray(event.frame).save(f"{out_dir}/{prefix}_rgb.png")
#     depth = event.depth_frame
#     if depth is not None:
#         Image.fromarray((depth / (np.max(depth)+1e-6) * 255).astype(np.uint8)).save(f"{out_dir}/{prefix}_depth.png")
#     seg = event.instance_segmentation_frame
#     if seg is not None:
#         Image.fromarray(seg).save(f"{out_dir}/{prefix}_seg.png")
#     return {
#         "rgb": f"{out_dir}/{prefix}_rgb.png",
#         "depth": f"{out_dir}/{prefix}_depth.png",
#         "seg": f"{out_dir}/{prefix}_seg.png",
#     }










# =========================
# Per-view integration
# =========================
def process_view_with_commands(
        controller,
        view_idx,
        out_dir,
        jsonl_path,
        view_info=None,
        max_objects=5,
        n_commands_range=(3, 6),
        min_success=2,
        max_fail=8,
        pickup_only=False,
        command_types=("camera", "object"),
        receptacle_placement_types=("center", "nearest_to_camera", "farthest_from_camera", "leftmost", "rightmost"),
        agent_camera_params={},
        cam_test_directions=("left","right","forward","back"),
        cam_test_distances=(0.05, 0.10, 0.15, 0.20, 0.25),
        cam_sample_distances=(0.05, 0.10, 0.15, 0.20),
        obj_relations=("to the right of", "to the left of", "in front of", "behind"),
        obj_sample_distances=(0.05, 0.10, 0.20, 0.30),
        disable_physics=False
):
    os.makedirs(out_dir, exist_ok=True)
    results = []

    objects = get_visible_interactables(controller, pickup_only=pickup_only, max_objects=max_objects)
    print(f"[Info] View {view_idx:02d}: Found {len(objects)} objects to process")

    # 记录每个物体的处理统计
    processing_summary = {
        "view_index": view_idx,
        "total_objects": len(objects),
        "objects_list": [{"type": o["objectType"], "id": o["objectId"]} for o in objects],
        "object_details": []
    }

    def pick_anchor(exclude_id):
        pool = [o for o in objects if o["objectId"] != exclude_id]
        return random.choice(pool) if pool else None

    for obj_index, obj in enumerate(objects):
        print(f"\n[Info] Processing object {obj_index + 1}/{len(objects)}: {obj['objectType']} ({obj['objectId']})")
        obj_id = obj["objectId"]
        obj_type = obj["objectType"]

        # 记录当前物体的统计
        obj_stat = {
            "object_index": obj_index,
            "object_type": obj_type,
            "object_id": obj_id,
            "camera_commands_generated": 0,
            "object_commands_generated": 0,
            "rotate_commands_generated": 0,
            "receptacle_commands_generated": 0,
            "agent_camera_commands_generated": 0,
            "anchor_used": None,
            "receptacle_used": None,
            "total_commands": 0,
            "success_count": 0,
            "fail_count": 0,
            "skipped": False,
            "skip_reason": None
        }

        # 根据 command_types 参数生成对应类型的命令
        action_pool = []

        if "camera" in command_types:
            cam_cmds = generate_camera_relative_commands(
                obj, controller,
                sample_distances=cam_sample_distances,
                test_directions=cam_test_directions,
                test_distances=cam_test_distances
            )
            obj_stat["camera_commands_generated"] = len(cam_cmds)
            print(f"  Generated {len(cam_cmds)} camera commands")
            action_pool.extend([("camera", c) for c in cam_cmds])

        if "object" in command_types:
            anchor = pick_anchor(obj_id)
            if anchor:
                obj_stat["anchor_used"] = {"type": anchor["objectType"], "id": anchor["objectId"]}
                print(f"  Picked anchor: {anchor['objectType']}")
                obj_cmds = generate_object_relative_commands(
                    obj,
                    anchor,
                    controller,
                    relations=obj_relations,
                    sample_distances=obj_sample_distances
                )
                obj_stat["object_commands_generated"] = len(obj_cmds)
                print(f"  Generated {len(obj_cmds)} object-relative commands")
                action_pool.extend([("object", c) for c in obj_cmds])
            else:
                print(f"  No anchor available for object-relative commands")

        if "rotate" in command_types:
            rot_cmds = generate_rotate_commands(obj, controller)
            obj_stat["rotate_commands_generated"] = len(rot_cmds)
            print(f"  Generated {len(rot_cmds)} rotate commands")
            action_pool.extend([("rotate", c) for c in rot_cmds])

        if "receptacle" in command_types:
            # 排除不应该被移动到容器上的大型物体
            LARGE_UNMOVABLE_TO_RECEPTACLE = {
                "Sofa", "ArmChair", "Chair", "Bed", "Stool", "Ottoman",
                "Bathtub", "BathtubBasin", "Toilet", "LazyChair",
                "DiningTable", "CoffeeTable", "SideTable", "Desk", "Dresser", "Shelf",
            }

            # 如果当前物体是大型物体，跳过 receptacle 命令生成
            if obj_type in LARGE_UNMOVABLE_TO_RECEPTACLE:
                print(f"  Skipping receptacle commands for large object type: {obj_type}")
                obj_stat["receptacle_commands_generated"] = 0
            else:
                # 获取可见的 receptacle 物体作为目标
                all_objs = controller.last_event.metadata.get("objects", [])
                VALID_RECEPTACLE_TYPES = {
                    "Table", "CoffeeTable", "CounterTop", "Desk", "Dresser",
                    "CabinetShelf", "Shelf", "SideTable", "DiningTable"
                }
                INVALID_RECEPTACLE_TYPES = {
                    "Floor", "Drawer"
                }
                receptacles = [
                    o for o in all_objs
                    if (
                        o.get("visible") and
                        o.get("receptacle") and
                        o["objectId"] != obj_id and
                        o["objectType"] not in INVALID_RECEPTACLE_TYPES
                    )
                ]
                # receptacles = [o for o in all_objs
                #               if o.get("visible") and o.get("receptacle") and o["objectId"] != obj_id]

                if receptacles:
                    # 随机选择一个 receptacle
                    target_receptacle = random.choice(receptacles)
                    obj_stat["receptacle_used"] = {
                        "type": target_receptacle["objectType"],
                        "id": target_receptacle["objectId"]
                    }
                    print(f"  Picked target receptacle: {target_receptacle['objectType']}")

                    rec_cmds = generate_receptacle_placement_commands(
                        obj,
                        target_receptacle,
                        controller,
                        placement_types=receptacle_placement_types
                    )
                    obj_stat["receptacle_commands_generated"] = len(rec_cmds)
                    print(f"  Generated {len(rec_cmds)} receptacle placement commands")
                    action_pool.extend([("receptacle", c) for c in rec_cmds])
                else:
                    print(f"  No visible receptacles available for receptacle placement")

        if "agent_camera" in command_types:
            agent_cam_cmds = generate_agent_camera_commands(controller, **agent_camera_params)
            obj_stat["agent_camera_commands_generated"] = len(agent_cam_cmds)
            print(f"  Generated {len(agent_cam_cmds)} agent camera commands")
            action_pool.extend([("agent_camera", c) for c in agent_cam_cmds])

        if "spatial_remove" in command_types:
            # 空间移除命令在视角级别生成，而不是物体级别
            # 我们只在第一个物体处理时生成一次
            if obj_index == 0:
                # moveable_only=False 允许移除所有可见物体（包括椅子、桌子等）
                spatial_cmds = generate_spatial_remove_commands(controller, min_count=2, moveable_only=False)
                obj_stat["spatial_remove_commands_generated"] = len(spatial_cmds)
                print(f"  Generated {len(spatial_cmds)} spatial remove commands (view-level)")
                action_pool.extend([("spatial_remove", c) for c in spatial_cmds])
            else:
                obj_stat["spatial_remove_commands_generated"] = 0

        if not action_pool:
            obj_stat["skipped"] = True
            obj_stat["skip_reason"] = "No commands generated"
            processing_summary["object_details"].append(obj_stat)
            print(f"  No commands generated for this object, skipping...")
            continue

        obj_stat["total_commands"] = len(action_pool)

        # 记录所有 action_pool 的详细信息
        action_pool_details = []
        for ctype, cmd in action_pool:
            cmd_detail = {
                "command_type": ctype,
                "instruction": cmd.get("instruction", ""),
            }
            # 根据命令类型添加特定信息
            if ctype == "camera":
                cmd_detail["direction"] = cmd.get("direction")
                cmd_detail["distance"] = cmd.get("dist_m")
            elif ctype == "object":
                cmd_detail["relation"] = cmd.get("relation")
                cmd_detail["anchor_id"] = cmd.get("anchor_id")
                cmd_detail["distance"] = cmd.get("dist_m")
            elif ctype == "rotate":
                cmd_detail["angle"] = cmd.get("angle")
                cmd_detail["direction"] = cmd.get("direction")
            elif ctype == "receptacle":
                cmd_detail["placement_type"] = cmd.get("placement_type")
                cmd_detail["receptacle_id"] = cmd.get("receptacle_id")
            elif ctype == "agent_camera":
                cmd_detail["action_type"] = cmd.get("action_type")
                cmd_detail["direction"] = cmd.get("direction")
                cmd_detail["magnitude"] = cmd.get("magnitude")
                cmd_detail["degrees"] = cmd.get("degrees")
            elif ctype == "spatial_remove":
                cmd_detail["object_type"] = cmd.get("object_type")
                cmd_detail["object_id"] = cmd.get("object_id")
                cmd_detail["spatial_relation"] = cmd.get("spatial_relation")

            action_pool_details.append(cmd_detail)

        obj_stat["action_pool"] = action_pool_details
        print(f"  Total action pool size: {len(action_pool)}")

        random.shuffle(action_pool)

        success_count = 0
        fail_count = 0
        target_cmd_num = random.randint(*n_commands_range)

        for idx, (ctype, cmd) in enumerate(action_pool):
            if success_count >= min_success and (success_count + fail_count) >= target_cmd_num:
                break
            if fail_count >= max_fail:
                break

            # 对于 spatial_remove，使用命令中的物体类型；其他命令使用当前物体类型
            if ctype == "spatial_remove":
                prefix_obj_type = cmd.get("object_type", obj_type)
                prefix_obj_id = cmd.get("object_id", obj_id)
            else:
                prefix_obj_type = obj_type
                prefix_obj_id = obj_id

            before_prefix = f"view{view_idx:02d}_{prefix_obj_type}_{idx}_before"
            before_imgs = save_current_view(controller, out_dir, before_prefix)

            # 获取执行前的相机信息
            before_camera = get_camera_info(controller)

            # 获取执行前的物体边界框信息
            # 对于 spatial_remove，获取目标物体的边界框
            before_obj = get_object_by_id(controller, prefix_obj_id)
            before_bbox = {
                "axisAlignedBoundingBox": before_obj.get("axisAlignedBoundingBox") if before_obj else None,
                "objectOrientedBoundingBox": before_obj.get("objectOrientedBoundingBox") if before_obj else None
            }

            if ctype == "camera":
                ok, result_dict = execute_camera_relative_command(
                    controller,
                    obj_id=obj_id,
                    direction=cmd["direction"],
                    dist_m=cmd["dist_m"]
                )
                # 从result_dict中提取信息
                reason = result_dict["reason"]
                used_coord = result_dict.get("final_pos", result_dict.get("target_pos"))
            elif ctype == "object":
                ok, result_dict = execute_object_relative_command(
                    controller,
                    obj_id=obj_id,
                    anchor_id=cmd["anchor_id"],
                    relation=cmd["relation"],
                    dist_m=cmd["dist_m"]
                )
                # 从result_dict中提取信息
                reason = result_dict["reason"]
                used_coord = result_dict.get("final_pos", result_dict.get("target_pos"))
            elif ctype == "rotate":
                ok, result_dict = execute_rotate_command(
                    controller,
                    obj_id=obj_id,
                    angle=cmd["angle"],
                    direction=cmd["direction"]
                )
                # 从result_dict中提取信息
                reason = result_dict["reason"]
                used_coord = result_dict.get("final_pos", result_dict.get("target_pos"))
            elif ctype == "receptacle":
                ok, result_dict = execute_receptacle_placement_command(
                    controller,
                    obj_id=obj_id,
                    receptacle_id=cmd["receptacle_id"],
                    placement_type=cmd["placement_type"]
                )
                # 从result_dict中提取信息
                reason = result_dict["reason"]
                used_coord = result_dict.get("final_pos", result_dict.get("target_pos"))
            elif ctype == "agent_camera":
                ok, result_dict = execute_agent_camera_command(
                    controller,
                    action_type=cmd["action_type"],
                    direction=cmd["direction"],
                    magnitude=cmd.get("magnitude"),
                    degrees=cmd.get("degrees")
                )
                # 从result_dict中提取信息
                reason = result_dict["reason"]
                used_coord = result_dict.get("final_pos", result_dict.get("target_pos"))
            elif ctype == "spatial_remove":
                ok, result_dict = execute_spatial_remove_command(
                    controller,
                    object_id=cmd["object_id"]
                )
                # 从result_dict中提取信息
                reason = result_dict["reason"]
                used_coord = None  # 移除操作没有坐标信息
            else:
                print(f"[Warning] Unknown command type: {ctype}")
                continue
            if ok:
                after_prefix = f"view{view_idx:02d}_{prefix_obj_type}_{idx}_after"
                after_imgs = save_current_view(controller, out_dir, after_prefix)

                # 获取执行后的相机信息
                after_camera = get_camera_info(controller)

                # 获取执行后的物体边界框信息
                # 对于 spatial_remove，获取目标物体的边界框（如果还存在的话）
                after_obj = get_object_by_id(controller, prefix_obj_id)
                after_bbox = {
                    "axisAlignedBoundingBox": after_obj.get("axisAlignedBoundingBox") if after_obj else None,
                    "objectOrientedBoundingBox": after_obj.get("objectOrientedBoundingBox") if after_obj else None
                }

                # ✅ 只在成功时复原物体或相机
                # spatial_remove命令是不可逆的，需要完全重置环境
                if ctype == "spatial_remove":
                    # spatial_remove命令：完全重置环境
                    start_time = time.time()
                    controller.reset()
                    apply_physics_settings(controller, disable_physics)
                    restore_view(controller, view_info)
                    end_time = time.time()
                    print(f"[Info] Full environment reset after spatial_remove took {end_time - start_time:.4f} seconds.")
                elif ctype == "agent_camera":
                    # agent_camera命令：重置相机到原始位置
                    if view_info:
                        start_time = time.time()
                        restore_view(controller, view_info)
                        end_time = time.time()
                        print(f"[Info] Camera reset took {end_time - start_time:.4f} seconds.")
                    else:
                        print(f"[Warning] No view_info available to reset camera.")
                else:
                    # 其他命令：使用优化的物体重置方法
                    original_pos = result_dict.get("original_pos")
                    original_rot = result_dict.get("original_rot")

                    if original_pos and original_rot:
                        start_time = time.time()
                        success, method = reset_object_position(
                            controller, prefix_obj_id, original_pos, original_rot, view_info, disable_physics
                        )
                        end_time = time.time()
                        print(f"[Info] Object reset using {method} took {end_time - start_time:.4f} seconds.")
                    else:
                        # 如果没有原始位置信息（如object relative命令），使用完整重置
                        start_time = time.time()
                        controller.reset()
                        apply_physics_settings(controller, disable_physics)
                        restore_view(controller, view_info)
                        end_time = time.time()
                        print(f"[Info] Full environment reset took {end_time - start_time:.4f} seconds.")

                # 记录结果

                # 根据命令类型提取 meta_direction
                if ctype == "camera":
                    meta_direction = cmd.get("direction")
                elif ctype == "object":
                    meta_direction = cmd.get("relation")
                elif ctype == "rotate":
                    meta_direction = f"{cmd.get('angle')}° {cmd.get('direction')}"
                elif ctype == "receptacle":
                    meta_direction = cmd.get("placement_type")
                elif ctype == "agent_camera":
                    action_type = cmd.get("action_type")
                    if action_type == "move":
                        meta_direction = f"{cmd.get('direction')} {cmd.get('magnitude')}m"
                    elif action_type in ["rotate", "pitch"]:
                        meta_direction = f"{cmd.get('direction')} {cmd.get('degrees')}°"
                    else:
                        meta_direction = cmd.get("direction")
                elif ctype == "spatial_remove":
                    meta_direction = cmd.get("spatial_relation")
                else:
                    meta_direction = None

                # 对于 spatial_remove，使用命令中的 object_id 和 object_type
                # 对于其他命令，使用当前循环的物体信息
                if ctype == "spatial_remove":
                    record_obj_id = cmd.get("object_id", obj_id)
                    record_obj_type = cmd.get("object_type", obj_type)
                else:
                    record_obj_id = obj_id
                    record_obj_type = obj_type

                rec = {
                    "view_index": view_idx,
                    "object_id": record_obj_id,
                    "object_type": record_obj_type,
                    "command_type": ctype,
                    "instruction_en": cmd["instruction"],
                    "meta_direction": meta_direction,
                    "before_images": before_imgs,
                    "after_images": after_imgs,
                    "success": bool(ok),
                    "reason": reason,
                    "used_coord": used_coord,
                    # 添加详细的位置和旋转信息
                    "original_pos": result_dict.get("original_pos"),
                    "original_rot": result_dict.get("original_rot"),
                    "target_pos": result_dict.get("target_pos"),
                    "target_rot": result_dict.get("target_rot"),
                    "final_pos": result_dict.get("final_pos"),
                    "final_rot": result_dict.get("final_rot"),
                    "used_force": result_dict.get("used_force", False),
                    # 添加边界框信息
                    "original_bbox": before_bbox,
                    "final_bbox": after_bbox,
                    # 添加相机pose和内参信息
                    "before_camera": before_camera,
                    "after_camera": after_camera
                }

                # 保存同名 JSON 文件
                json_path = os.path.join(out_dir, f"{after_prefix}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)

                append_jsonl(rec, jsonl_path)
                results.append(rec)
            else:
                print(f"[Info] Command failed: {cmd['instruction']} | Reason: {reason}")
                # 记录失败原因到json
                # 根据命令类型提取 meta_direction
                if ctype == "camera":
                    meta_direction = cmd.get("direction")
                elif ctype == "object":
                    meta_direction = cmd.get("relation")
                elif ctype == "rotate":
                    meta_direction = f"{cmd.get('angle')}° {cmd.get('direction')}"
                elif ctype == "receptacle":
                    meta_direction = cmd.get("placement_type")
                elif ctype == "agent_camera":
                    action_type = cmd.get("action_type")
                    if action_type == "move":
                        meta_direction = f"{cmd.get('direction')} {cmd.get('magnitude')}m"
                    elif action_type in ["rotate", "pitch"]:
                        meta_direction = f"{cmd.get('direction')} {cmd.get('degrees')}°"
                    else:
                        meta_direction = cmd.get("direction")
                elif ctype == "spatial_remove":
                    meta_direction = cmd.get("spatial_relation")
                else:
                    meta_direction = None

                # 对于 spatial_remove，使用命令中的 object_id 和 object_type
                # 对于其他命令，使用当前循环的物体信息
                if ctype == "spatial_remove":
                    record_obj_id = cmd.get("object_id", obj_id)
                    record_obj_type = cmd.get("object_type", obj_type)
                else:
                    record_obj_id = obj_id
                    record_obj_type = obj_type

                rec = {
                    "view_index": view_idx,
                    "object_id": record_obj_id,
                    "object_type": record_obj_type,
                    "command_type": ctype,
                    "instruction_en": cmd["instruction"],
                    "meta_direction": meta_direction,
                    "before_images": before_imgs,
                    "after_images": None,
                    "success": bool(ok),
                    "reason": reason,
                    "used_coord": used_coord,
                    # 添加详细的位置和旋转信息
                    "original_pos": result_dict.get("original_pos"),
                    "original_rot": result_dict.get("original_rot"),
                    "target_pos": result_dict.get("target_pos"),
                    "target_rot": result_dict.get("target_rot"),
                    "final_pos": result_dict.get("final_pos"),
                    "final_rot": result_dict.get("final_rot"),
                    "used_force": result_dict.get("used_force", False),
                    # 添加边界框信息（失败情况下只有原始边界框）
                    "original_bbox": before_bbox,
                    "final_bbox": None,
                    # 添加相机pose和内参信息（失败情况下只有before_camera）
                    "before_camera": before_camera,
                    "after_camera": None
                }
                if True:
                    # 失败情况下的重置逻辑
                    if ctype == "agent_camera":
                        # agent_camera命令失败：重置相机到原始位置
                        if view_info:
                            restore_view(controller, view_info)
                    else:
                        # 其他命令失败：需要reset环境或物体
                        original_pos = result_dict.get("original_pos")
                        original_rot = result_dict.get("original_rot")

                        if original_pos and original_rot:
                            # 尝试快速重置物体
                            reset_object_position(controller, prefix_obj_id, original_pos, original_rot, view_info, disable_physics)
                        else:
                            # 完整环境重置
                            controller.reset()
                            apply_physics_settings(controller, disable_physics)
                            restore_view(controller, view_info)
                json_path = os.path.join(out_dir, f"{before_prefix}_fail.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)

            if ok: success_count += 1
            else: fail_count += 1

        # 更新当前物体的统计
        obj_stat["success_count"] = success_count
        obj_stat["fail_count"] = fail_count
        processing_summary["object_details"].append(obj_stat)
        print(f"  Object processed: {success_count} successes, {fail_count} failures")

        # 每个物体处理完后进行一次完整重置，确保环境干净
        controller.reset()
        apply_physics_settings(controller, disable_physics)
        restore_view(controller, view_info)

    # 计算总体统计
    total_success = sum(d["success_count"] for d in processing_summary["object_details"])
    total_fail = sum(d["fail_count"] for d in processing_summary["object_details"])
    processing_summary["total_success"] = total_success
    processing_summary["total_fail"] = total_fail
    processing_summary["total_results"] = len(results)

    # 保存处理摘要
    summary_path = os.path.join(out_dir, "processing_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(processing_summary, f, ensure_ascii=False, indent=2)

    print(f"\n[Info] View {view_idx:02d} completed: {total_success} successes, {total_fail} failures")
    print(f"[Info] Processing summary saved to: {summary_path}")
    return results


def run_commands_for_all_views(controller, views, output_root="./output_instructions", jsonl="output_instructions/records.jsonl",
                                max_objects=3, pickup_only=False, command_types=("camera",),
                                receptacle_placement_types=("center", "nearest_to_camera", "farthest_from_camera", "leftmost", "rightmost"),
                                disable_physics=False):
    os.makedirs(output_root, exist_ok=True)

    for idx, v in enumerate(views):
        # 切换到该视角
        controller.step(
            action="TeleportFull",
            position=v["pos"],
            rotation={"x":0,"y":int(v["rot"]),"z":0},
            horizon=int(v.get("horizon", 0)),
            standing=True,
            forceAction=True
        )
        # 为该视角生成并执行指令（英文，每个物体1~2条；前后图片都会保存）
        out_dir = os.path.join(output_root, f"view_{idx:02d}_room{v.get('room_id','x')}")
        process_view_with_commands(
            controller,
            view_idx=idx,
            out_dir=out_dir,
            jsonl_path=jsonl,
            view_info=v,              # 传入视角信息用于reset后恢复
            max_objects=max_objects,
            pickup_only=pickup_only,
            command_types=command_types,
            receptacle_placement_types=receptacle_placement_types,
            disable_physics=disable_physics
        )




# =====================
# Main test logic
# =====================

def process_single_scene(scene_name, config, output_dir, resume=False):
    """
    处理单个场景

    Args:
        scene_name: 场景名称
        config: 配置字典
        output_dir: 输出目录
        resume: 是否启用 resume 模式
    """
    print(f"\n{'='*80}")
    print(f"[Info] Processing scene: {scene_name}")
    print(f"[Info] Output directory: {output_dir}")
    print(f"{'='*80}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Resume: 检查场景是否已完成（通过 records.jsonl 文件）
    if resume and is_scene_completed(output_dir):
        print(f"[Resume] Scene {scene_name} already completed (records.jsonl exists), skipping")
        return True

    # 使用配置初始化Controller
    ctrl_config = config['controller']

    # 如果是 spatial_remove 模式，使用更大的可见距离
    cmd_config = config['command_processing']
    visibility_distance = ctrl_config['visibilityDistance']
    if 'spatial_remove' in cmd_config.get('command_types', []):
        visibility_distance = 6.0
        print(f"[Info] Using extended visibilityDistance={visibility_distance} for spatial_remove mode")

    controller = Controller(
        scene=scene_name,
        agentMode=ctrl_config['agentMode'],
        platform=CloudRendering,
        gridSize=ctrl_config['gridSize'],
        rotateStepDegrees=ctrl_config['rotateStepDegrees'],
        snapToGrid=ctrl_config['snapToGrid'],
        visibilityDistance=visibility_distance,
        renderDepthImage=ctrl_config['renderDepthImage'],
        renderInstanceSegmentation=ctrl_config['renderInstanceSegmentation'],
        fieldOfView=ctrl_config['fieldOfView'],
        width=ctrl_config['width'],
        height=ctrl_config['height'],
        quality=ctrl_config['quality'],
    )
    print("[Info] RoboTHOR Controller initialized successfully.")

    # 应用物理设置
    apply_physics_settings(controller, config.get('disable_physics', False))
    # 先保存 俯视图以及 receptacle 分布图
    topdown_img = get_top_down_frame(controller)
    topdown_img.save(f"{output_dir}/top_down_view.png")
    save_all_receptacle_spawn_plot(controller, out_path=f"{output_dir}/rec_spawn_points.png", anywhere=True)

    # 使用配置参数生成或加载视角
    room_config = config['room_views']
    cmd_config = config['command_processing']

    # 1. 首先尝试从当前输出目录加载
    views = None
    if os.path.exists(f"{output_dir}/selected_views.json"):
        with open(f"{output_dir}/selected_views.json", 'r', encoding='utf-8') as f:
            views = json.load(f)
        print(f"[Info] Loaded {len(views)} views from current output directory.")

    # 2. 如果当前目录没有，尝试从预先生成的视角路径加载
    elif room_config.get('pregenerated_views_path'):
        pregenerated_path = os.path.join(
            room_config['pregenerated_views_path'],
            scene_name,
            'selected_views.json'
        )
        if os.path.exists(pregenerated_path):
            with open(pregenerated_path, 'r', encoding='utf-8') as f:
                views = json.load(f)
            print(f"[Info] Loaded {len(views)} views from pregenerated path: {pregenerated_path}")
            # 将加载的视角保存到当前输出目录，便于下次复用
            with open(f"{output_dir}/selected_views.json", 'w', encoding='utf-8') as f:
                json.dump(views, f, ensure_ascii=False, indent=2)
            print(f"[Info] Saved views to current output directory for future use.")
        else:
            print(f"[Warning] Pregenerated views file not found at: {pregenerated_path}")

    # 3. 如果都没有找到，则生成新的视角
    if views is None:
        print(f"[Info] Generating new views for scene {scene_name}...")
        views = generate_room_views(
            controller,
            k_per_room=room_config['k_per_room'],
            eps=room_config['eps'],
            min_members=room_config['min_members'],
            sample_positions_per_room=room_config['sample_positions_per_room'],
            require_pickupable=room_config['require_pickupable']
        )
        # 保存views信息便于下次复用
        with open(f"{output_dir}/selected_views.json", 'w', encoding='utf-8') as f:
            json.dump(views, f, ensure_ascii=False, indent=2)
        print(f"[Info] Generated and saved {len(views)} views.")

    # 使用配置处理视角
    run_commands_for_all_views(
        controller,
        views,
        output_root=output_dir,
        jsonl=f"{output_dir}/records.jsonl",
        max_objects=cmd_config['max_objects'],
        pickup_only=cmd_config['pickup_only'],
        command_types=tuple(cmd_config['command_types']),
        disable_physics=config.get('disable_physics', False)
    )
    print("\n[Result] Final selected views:")
    for v in views:
        print(f"  - Room {v['room_id']}: rot={v['rot']}, score={v['score']}, pos={v['pos']}")

    # 可选：把这些视角逐一保存帧图
    os.makedirs(f"{output_dir}/room_views", exist_ok=True)

    for i, v in enumerate(views):
        evt = controller.step(action="Teleport", position=v["pos"], rotation=int(v["rot"]), forceAction=True)
        if not evt.metadata.get("lastActionSuccess", False):
            continue
        # 统计并保存
        score, vis_ids = count_visible_interactables(controller, require_pickupable=False)
        Image.fromarray(evt.frame).save(f"{output_dir}/room_views/view_{i:02d}_room{v['room_id']}_rot{v['rot']}_sc{score}.png")

    # 重置环境后再保存一次
    controller.reset()
    apply_physics_settings(controller, config.get('disable_physics', False))
    os.makedirs(f"{output_dir}/room_views_reset", exist_ok=True)
    for i, v in enumerate(views):
        evt = controller.step(action="Teleport", position=v["pos"], rotation=int(v["rot"]), forceAction=True)
        if not evt.metadata.get("lastActionSuccess", False):
            continue
        # 统计并保存
        score, vis_ids = count_visible_interactables(controller, require_pickupable=False)
        Image.fromarray(evt.frame).save(f"{output_dir}/room_views_reset/view_{i:02d}_room{v['room_id']}_rot{v['rot']}_sc{score}.png")

    controller.stop()

    print(f"\n[Info] Scene {scene_name} processing completed!")
    return True


def main(config=None):
    """
    主函数 - 支持批量处理多个场景
    """
    # 如果没有提供配置，使用默认配置
    if config is None:
        config = create_default_config()

    # 设置随机种子
    random.seed(config['random_seed'])

    # 获取场景列表
    scenes = config.get('scenes', ['FloorPlan_Train1_3'])
    if isinstance(scenes, str):
        scenes = [scenes]

    base_output_dir = config['output_dir']
    separate_folders = config.get('separate_scene_folders', True)
    resume = config.get('resume', False)

    print(f"\n[Info] Total scenes to process: {len(scenes)}")
    print(f"[Info] Scenes: {', '.join(scenes)}")
    print(f"[Info] Base output directory: {base_output_dir}")
    print(f"[Info] Separate scene folders: {separate_folders}")
    print(f"[Info] Resume mode: {'Enabled' if resume else 'Disabled'}\n")

    # 保存总体配置到基础输出目录
    os.makedirs(base_output_dir, exist_ok=True)
    save_config(config, base_output_dir)

    # 处理每个场景
    successful_scenes = []
    failed_scenes = []

    for idx, scene_name in enumerate(scenes, 1):
        print(f"\n{'#'*80}")
        print(f"# Processing scene {idx}/{len(scenes)}: {scene_name}")
        print(f"{'#'*80}")

        try:
            # 确定输出目录
            if separate_folders:
                scene_output_dir = os.path.join(base_output_dir, scene_name)
            else:
                scene_output_dir = base_output_dir

            # 处理场景
            success = process_single_scene(scene_name, config, scene_output_dir, resume=resume)

            if success:
                successful_scenes.append(scene_name)
            else:
                failed_scenes.append(scene_name)

        except Exception as e:
            print(f"\n[Error] Failed to process scene {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            failed_scenes.append(scene_name)
            continue

    # 打印总结
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total scenes: {len(scenes)}")
    print(f"Successful: {len(successful_scenes)}")
    print(f"Failed: {len(failed_scenes)}")

    if successful_scenes:
        print(f"\nSuccessful scenes:")
        for scene in successful_scenes:
            print(f"  ✓ {scene}")

    if failed_scenes:
        print(f"\nFailed scenes:")
        for scene in failed_scenes:
            print(f"  ✗ {scene}")

    print(f"\n[Info] All processing completed!")
    print(f"[Info] Results saved to: {base_output_dir}")


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    args.debug = False  # 默认关闭debug模式
    if args.debug:
        print("[Debug] Debug mode enabled.")
        #         --scenes "val:all" \
        # --output-dir ./data/outputs/val/spatial_remove_debug \
        # --command-types spatial_remove \
        # --pregenerated-views ./data/pregenerated_views/val
        args.scenes = "val:all"
        args.output_dir = "./data/outputs/val/spatial_remove_debug"
        args.command_types = ["spatial_remove"]
        args.pregenerated_views = "./data/pregenerated_views/val"



    # 创建默认配置
    default_config = create_default_config()

    # 合并配置（配置文件 + 命令行参数）
    config = merge_config(default_config, args)

    # 打印使用的配置
    print("[Info] Running with configuration:")
    print(json.dumps(config, indent=2, ensure_ascii=False))

    # 运行主程序
    main(config)
