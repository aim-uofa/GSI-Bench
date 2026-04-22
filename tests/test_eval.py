"""
Unit tests for Generative-Spatial-Intelligence evaluation pure functions.
No GPU or model inference required.
"""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'evaluation'))

from eval.statistics_utils import compute_statistics
from eval.aggregate import (
    normalize_dataset_name,
    normalize_model_name,
    parse_ac_scores,
    load_json,
    write_json,
)


# ---------- statistics_utils ----------

def test_compute_statistics_ic_overall():
    result = {
        "a_edit_q0": {"operation": "move",   "compliance": 1},
        "b_edit_q0": {"operation": "move",   "compliance": 0},
        "c_edit_q0": {"operation": "rotate", "compliance": 1},
    }
    stats = compute_statistics(result, "instruction-compliance", None)
    assert abs(stats["overall_success_rate"] - 2 / 3) < 1e-9
    assert stats["overall_total_count"] == 3


def test_compute_statistics_ic_per_op():
    result = {
        "a_edit_q0": {"operation": "move",   "compliance": 1},
        "b_edit_q0": {"operation": "move",   "compliance": 0},
        "c_edit_q0": {"operation": "rotate", "compliance": 1},
    }
    stats = compute_statistics(result, "instruction-compliance", None)
    assert abs(stats["operation_success_rate"]["move"]["rate"] - 0.5) < 1e-9
    assert stats["operation_success_rate"]["move"]["count"] == 2
    assert abs(stats["operation_success_rate"]["rotate"]["rate"] - 1.0) < 1e-9


def test_compute_statistics_sa():
    result = {
        "a_edit_q0": {"operation": "move",   "edit_score": 0.8},
        "b_edit_q0": {"operation": "rotate", "edit_score": 0.6},
    }
    stats = compute_statistics(result, "spatial-accuracy", None)
    assert abs(stats["overall_mean_score"] - 0.7) < 1e-9
    assert abs(stats["operation_mean_score"]["move"] - 0.8) < 1e-9


def test_compute_statistics_el():
    result = {
        "a_edit_q0": {"operation": "move", "ssim": 0.9, "lpips": 0.1, "mse": 0.01},
        "b_edit_q0": {"operation": "move", "ssim": 0.7, "lpips": 0.3, "mse": 0.03},
    }
    stats = compute_statistics(result, "edit-locality", None)
    assert abs(stats["overall_mean_ssim"] - 0.8) < 1e-9
    assert abs(stats["overall_mean_lpips"] - 0.2) < 1e-9


def test_compute_statistics_empty():
    stats = compute_statistics({}, "instruction-compliance", None)
    assert stats["overall_success_rate"] == 0.0


# ---------- aggregate ----------

def test_normalize_dataset_name_with_suffix():
    assert normalize_dataset_name("robothor_eval") == "robothor"


def test_normalize_dataset_name_without_suffix():
    assert normalize_dataset_name("robothor") == "robothor"


def test_normalize_model_name():
    assert normalize_model_name("my_model_v2") == "mymodelv2"


def test_normalize_model_name_no_underscore():
    assert normalize_model_name("bagel") == "bagel"


def test_parse_ac_scores():
    ac_data = [
        {"meta": {"index": "img1_edit_q1"},
         "prediction": 'text||V^=^V||{"score": 0.9}'},
        {"meta": {"index": "img2_edit_q1"},
         "prediction": 'text||V^=^V||{"score": 0.3}'},
    ]
    result = parse_ac_scores(ac_data)
    assert abs(result["img1_edit_q1"] - 0.9) < 1e-9
    assert abs(result["img2_edit_q1"] - 0.3) < 1e-9


def test_parse_ac_scores_malformed():
    ac_data = [
        {"meta": {"index": "img1_edit_q1"}, "prediction": "no separator here"},
    ]
    result = parse_ac_scores(ac_data)
    assert result["img1_edit_q1"] is None


def test_load_write_json_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sub", "data.json")
        data = {"key": [1, 2, 3], "nested": {"a": "b"}}
        write_json(path, data)
        loaded = load_json(path)
        assert loaded == data
