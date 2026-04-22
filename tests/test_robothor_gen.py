"""
Unit tests for GSI-Bench-robothor-GEN pure functions.
No AI2-THOR or GPU required.
"""
import sys
import os
import json
import math
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robothor'))

from action_utils.scene_utils import generate_scene_list, parse_scene_specification
from action_utils.config_utils import create_default_config, save_config, load_config
from action_utils.resume_utils import is_scene_completed
from action_utils.spawn_utils import _dist3, _dist2, _norm2, _dot2


# ---------- scene_utils ----------

def test_generate_scene_list_train():
    scenes = generate_scene_list("train", 1, 2, 1, 3)
    assert scenes == [
        "FloorPlan_Train1_1", "FloorPlan_Train1_2", "FloorPlan_Train1_3",
        "FloorPlan_Train2_1", "FloorPlan_Train2_2", "FloorPlan_Train2_3",
    ]


def test_generate_scene_list_val():
    scenes = generate_scene_list("val", 1, 1, 1, 2)
    assert scenes == ["FloorPlan_Val1_1", "FloorPlan_Val1_2"]


def test_parse_scene_specification_all_train():
    scenes = parse_scene_specification("train:all")
    assert len(scenes) == 60          # 12 major × 5 minor
    assert scenes[0] == "FloorPlan_Train1_1"
    assert scenes[-1] == "FloorPlan_Train12_5"


def test_parse_scene_specification_all_val():
    scenes = parse_scene_specification("val:all")
    assert len(scenes) == 15          # 3 major × 5 minor


def test_parse_scene_specification_range():
    scenes = parse_scene_specification("train:1-2:1-3")
    assert len(scenes) == 6


def test_parse_scene_specification_single():
    scenes = parse_scene_specification("FloorPlan_Train1_3")
    assert scenes == ["FloorPlan_Train1_3"]


def test_parse_scene_specification_comma():
    scenes = parse_scene_specification("FloorPlan_Train1_1,FloorPlan_Train1_2")
    assert len(scenes) == 2


# ---------- spawn_utils (geometry) ----------

def test_dist3():
    a = {"x": 0.0, "y": 0.0, "z": 0.0}
    b = {"x": 3.0, "y": 4.0, "z": 0.0}
    assert abs(_dist3(a, b) - 5.0) < 1e-9


def test_dist2_ignores_y():
    a = {"x": 0.0, "y": 0.0, "z": 0.0}
    b = {"x": 3.0, "y": 99.0, "z": 4.0}   # y does not affect dist2
    assert abs(_dist2(a, b) - 5.0) < 1e-9


def test_norm2():
    v = {"x": 3.0, "y": 0.0, "z": 4.0}
    assert abs(_norm2(v) - 5.0) < 1e-9


def test_dot2():
    a = {"x": 1.0, "y": 0.0, "z": 2.0}
    b = {"x": 3.0, "y": 0.0, "z": 4.0}
    assert abs(_dot2(a, b) - 11.0) < 1e-9


# ---------- config_utils ----------

def test_create_default_config_keys():
    cfg = create_default_config()
    assert "controller" in cfg
    assert "room_views" in cfg
    assert "command_processing" in cfg
    assert cfg["controller"]["fieldOfView"] == 90
    assert cfg["controller"]["width"] == 1024


def test_save_load_config_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = create_default_config()
        cfg["random_seed"] = 9999
        save_config(cfg, tmp)
        loaded = load_config(os.path.join(tmp, "config.json"))
        assert loaded["random_seed"] == 9999
        assert loaded["controller"] == cfg["controller"]


# ---------- resume_utils ----------

def test_is_scene_completed_false():
    with tempfile.TemporaryDirectory() as tmp:
        assert not is_scene_completed(tmp)


def test_is_scene_completed_true():
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "records.jsonl"), "w").close()
        assert is_scene_completed(tmp)
