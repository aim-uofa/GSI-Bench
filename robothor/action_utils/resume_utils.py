#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Resume functionality for RoboTHOR experiments.
Author: zmz @ Zhejiang University
"""

import os


def is_scene_completed(scene_output_dir):
    """
    检查场景是否已完成（通过检查 records.jsonl 文件是否存在）

    Args:
        scene_output_dir: 场景输出目录

    Returns:
        True if 场景已完成 (records.jsonl 存在), False otherwise
    """
    records_file = os.path.join(scene_output_dir, "records.jsonl")
    return os.path.exists(records_file)
