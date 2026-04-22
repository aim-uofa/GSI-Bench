"""
Integration tests — verify the open-source release is usable end-to-end.
No GPU, AI2-THOR, or Blender required.
"""
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo(*parts):
    return os.path.join(REPO_ROOT, *parts)


def _run_python(script_path, args, cwd=None, timeout=30):
    """Run a Python script, return (returncode, stdout, stderr)."""
    cmd = [sys.executable, script_path] + args
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd or os.path.dirname(script_path),
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


# ===========================================================================
# 1. Repo structure: documented files actually exist
# ===========================================================================

class TestRepoStructure:

    REQUIRED_FILES = [
        "README.md",
        ".gitignore",
        "paper/main.pdf",
        # robothor
        "robothor/README.md",
        "robothor/README.en.md",
        "robothor/requirements.txt",
        "robothor/test_robothor_simple_cluster_move.py",
        "robothor/robothor_utils.py",
        "robothor/action_utils/__init__.py",
        "robothor/action_utils/config_utils.py",
        "robothor/action_utils/scene_utils.py",
        "robothor/action_utils/spawn_utils.py",
        "robothor/action_utils/resume_utils.py",
        "robothor/action_utils/camera_relative.py",
        "robothor/action_utils/object_relative.py",
        "robothor/action_utils/rotate.py",
        "robothor/action_utils/receptacle_placement.py",
        "robothor/action_utils/agent_camera.py",
        "robothor/action_utils/spatial_remove.py",
        "robothor/action_utils/object_utils.py",
        "robothor/action_utils/environment_utils.py",
        "robothor/action_utils/utils_record.py",
        "robothor/scripts/generate_train.sh",
        "robothor/scripts/generate_train_object.sh",
        "robothor/scripts/generate_train_rotate.sh",
        "robothor/scripts/generate_train_receptacle.sh",
        "robothor/scripts/generate_train_spatial_remove.sh",
        "robothor/scripts/generate_train_agent_camera.sh",
        "robothor/scripts/generate_val_agent_camera.sh",
        # mesatask
        "mesatask/README.md",
        "mesatask/requirement.txt",
        "mesatask/generate_atomic_transforms.py",
        "mesatask/organize_image_editing_dataset.py",
        "mesatask/instruction_templates.py",
        "mesatask/config.yaml",
        "mesatask/inference.py",
        "mesatask/get_task_info.py",
        # evaluation
        "evaluation/README.md",
        "evaluation/requirements.txt",
        "evaluation/eval.sh",
        "evaluation/eval/eval.py",
        "evaluation/eval/eval_utils.py",
        "evaluation/eval/aggregate.py",
        "evaluation/eval/statistics_utils.py",
        "evaluation/eval/README.md",
        "evaluation/mllm_eval/mllm_eval.py",
        "evaluation/mllm_eval/eval_infer.sh",
        "evaluation/examples/inference.py",
        "evaluation/prepare_datasets.sh",
    ]

    def test_all_documented_files_exist(self):
        missing = []
        for f in self.REQUIRED_FILES:
            if not os.path.exists(_repo(f)):
                missing.append(f)
        assert not missing, f"Missing files:\n" + "\n".join(f"  - {f}" for f in missing)

    def test_shell_scripts_are_executable_or_bashable(self):
        """All .sh files should be valid shell scripts (at least parseable)."""
        sh_files = []
        for root, _, files in os.walk(REPO_ROOT):
            for f in files:
                if f.endswith(".sh"):
                    sh_files.append(os.path.join(root, f))
        assert len(sh_files) >= 5, f"Expected at least 5 .sh files, found {len(sh_files)}"

        errors = []
        for sh in sh_files:
            proc = subprocess.run(
                ["bash", "-n", sh], capture_output=True, text=True
            )
            if proc.returncode != 0:
                errors.append(f"{os.path.relpath(sh, REPO_ROOT)}: {proc.stderr.strip()}")
        assert not errors, "Shell syntax errors:\n" + "\n".join(errors)


# ===========================================================================
# 2. No hardcoded internal paths
# ===========================================================================

class TestNoHardcodedPaths:

    FORBIDDEN_PATTERNS = [
        r"/home/zmz/",
        r"/code/zmz/",
        r"/dllab/",
        r"/dllab_nas/",
        r"/heyuan2_12/",
    ]

    SCAN_EXTENSIONS = {".py", ".sh", ".md", ".yaml", ".yml", ".txt", ".json"}
    # Skip test files themselves (they legitimately contain the patterns as strings)
    SKIP_DIRS = {"__pycache__", ".git", "tests"}

    def test_no_internal_paths_in_source_files(self):
        violations = []
        for root, dirs, files in os.walk(REPO_ROOT):
            # Prune skipped directories
            dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]
            for fname in files:
                ext = os.path.splitext(fname)[1]
                if ext not in self.SCAN_EXTENSIONS:
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            for pat in self.FORBIDDEN_PATTERNS:
                                if re.search(pat, line):
                                    rel = os.path.relpath(fpath, REPO_ROOT)
                                    violations.append(f"{rel}:{i} matches '{pat}'")
                except Exception:
                    pass
        assert not violations, (
            f"Found {len(violations)} hardcoded path(s):\n"
            + "\n".join(f"  {v}" for v in violations[:20])
        )


# ===========================================================================
# 3. CLI help works (scripts are at least parseable)
# ===========================================================================

class TestCLIHelp:

    def test_robothor_config_parse_arguments(self):
        """Verify config_utils module is importable with all expected functions."""
        sys.path.insert(0, _repo("robothor"))
        try:
            from action_utils.config_utils import (
                parse_arguments,
                create_default_config,
                merge_config,
                save_config,
                load_config,
            )
            assert callable(parse_arguments)
            assert callable(create_default_config)
            assert callable(merge_config)
        finally:
            sys.path.pop(0)

    def test_mesatask_generate_help(self):
        """generate_atomic_transforms.py --help should exit 0."""
        rc, out, err = _run_python(
            _repo("mesatask", "generate_atomic_transforms.py"),
            ["--help"],
            cwd=_repo("mesatask"),
        )
        assert rc == 0, f"Exit code {rc}.\nstdout: {out}\nstderr: {err}"
        assert "--input-dir" in out
        assert "--num-variants" in out

    def test_eval_aggregate_help(self):
        """aggregate.py --help should exit 0."""
        rc, out, err = _run_python(
            _repo("evaluation", "eval", "aggregate.py"),
            ["--help"],
            cwd=_repo("evaluation"),
        )
        assert rc == 0, f"Exit code {rc}.\nstdout: {out}\nstderr: {err}"
        assert "--root-dir" in out
        assert "--mllm-eval-dir" in out


# ===========================================================================
# 4. Robothor config full round-trip pipeline
# ===========================================================================

class TestRobothorConfigPipeline:

    def test_config_create_save_load_parse_merge(self):
        """Full config lifecycle: create → save → load → parse scenes → merge."""
        sys.path.insert(0, _repo("robothor"))
        try:
            from action_utils.config_utils import (
                create_default_config,
                save_config,
                load_config,
                merge_config,
            )
            from action_utils.scene_utils import parse_scene_specification

            cfg = create_default_config()
            assert isinstance(cfg, dict)

            with tempfile.TemporaryDirectory() as tmp:
                save_config(cfg, tmp)
                path = os.path.join(tmp, "config.json")
                assert os.path.exists(path)
                loaded = load_config(path)
                assert loaded["controller"]["width"] == cfg["controller"]["width"]

            scenes = parse_scene_specification("train:1:1-3")
            assert len(scenes) == 3

            scenes_all = parse_scene_specification("val:all")
            assert len(scenes_all) == 15
        finally:
            sys.path.pop(0)


# ===========================================================================
# 5. MesaTask instruction templates are usable
# ===========================================================================

class TestMesataskInstructionGeneration:

    def test_format_instruction_en(self):
        sys.path.insert(0, _repo("mesatask"))
        try:
            from instruction_templates import TEMPLATES

            for key in ["move_right", "move_left", "move_forward", "move_backward"]:
                templates = TEMPLATES[key]["en"]
                result = templates[0].format(obj="red cup", value="10")
                assert "red cup" in result
                assert "10" in result
        finally:
            sys.path.pop(0)

    def test_format_instruction_zh(self):
        sys.path.insert(0, _repo("mesatask"))
        try:
            from instruction_templates import TEMPLATES

            tpl = TEMPLATES["move_right"]["zh"][0]
            result = tpl.format(obj="红色杯子", value="15")
            assert "红色杯子" in result
            assert "15" in result
        finally:
            sys.path.pop(0)


# ===========================================================================
# 6. Evaluation pipeline on mock data
# ===========================================================================

class TestEvalPipelineMock:

    def test_statistics_all_modes(self):
        """compute_statistics works for all three modes on synthetic data."""
        sys.path.insert(0, _repo("evaluation"))
        try:
            from eval.statistics_utils import compute_statistics

            # IC
            ic_data = {
                f"img{i}_edit_q0": {"operation": op, "compliance": c}
                for i, (op, c) in enumerate([
                    ("move", 1), ("move", 0), ("rotate", 1),
                    ("remove", 1), ("scale", 0), ("view", 1),
                ])
            }
            ic_stats = compute_statistics(ic_data, "instruction-compliance", None)
            assert 0 < ic_stats["overall_success_rate"] < 1
            assert ic_stats["overall_total_count"] == 6
            assert "move" in ic_stats["operation_success_rate"]

            # SA
            sa_data = {
                f"img{i}_edit_q0": {"operation": "move", "edit_score": 0.1 * i}
                for i in range(1, 6)
            }
            sa_stats = compute_statistics(sa_data, "spatial-accuracy", None)
            assert 0 < sa_stats["overall_mean_score"] < 1

            # EL
            el_data = {
                f"img{i}_edit_q0": {
                    "operation": "move",
                    "ssim": 0.8 + 0.02 * i,
                    "lpips": 0.1 - 0.01 * i,
                    "mse": 0.01 * i,
                }
                for i in range(5)
            }
            el_stats = compute_statistics(el_data, "edit-locality", None)
            assert "overall_mean_ssim" in el_stats
            assert "overall_mean_lpips" in el_stats
        finally:
            sys.path.pop(0)

    def test_aggregate_write_and_load(self):
        """write_json → load_json round-trip, normalize helpers."""
        sys.path.insert(0, _repo("evaluation"))
        try:
            from eval.aggregate import (
                write_json,
                load_json,
                normalize_dataset_name,
                normalize_model_name,
            )

            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "d1", "d2", "results.json")
                data = {
                    "model": "test_model",
                    "scores": {"ic": 0.8, "sa": 0.7, "el": 0.9},
                }
                write_json(path, data)
                loaded = load_json(path)
                assert loaded == data

            assert normalize_dataset_name("fine_eval") == "fine"
            assert normalize_dataset_name("robothor") == "robothor"
            assert normalize_model_name("my_best_model") == "mybestmodel"
        finally:
            sys.path.pop(0)

    def test_eval_full_mock_pipeline(self):
        """Simulate: generate IC/SA/EL stats → write JSON → load & verify."""
        sys.path.insert(0, _repo("evaluation"))
        try:
            from eval.statistics_utils import compute_statistics
            from eval.aggregate import write_json, load_json

            mock_results = {}
            for i in range(20):
                op = ["move", "rotate", "remove", "scale"][i % 4]
                mock_results[f"scene{i:03d}_edit_q{i}"] = {
                    "operation": op,
                    "compliance": 1 if i % 3 != 0 else 0,
                    "edit_score": 0.5 + 0.02 * i,
                    "ssim": 0.85 + 0.005 * i,
                    "lpips": 0.15 - 0.005 * i,
                    "mse": 0.02 + 0.001 * i,
                }

            with tempfile.TemporaryDirectory() as tmp:
                all_stats = {}
                for mode in ["instruction-compliance", "spatial-accuracy", "edit-locality"]:
                    stats = compute_statistics(mock_results, mode, None)
                    assert isinstance(stats, dict)
                    all_stats[mode] = stats

                out_path = os.path.join(tmp, "mock_eval_results.json")
                write_json(out_path, all_stats)
                loaded = load_json(out_path)

                assert loaded["instruction-compliance"]["overall_total_count"] == 20
                assert 0 < loaded["spatial-accuracy"]["overall_mean_score"] < 1
                assert 0 < loaded["edit-locality"]["overall_mean_ssim"] <= 1
        finally:
            sys.path.pop(0)


# ===========================================================================
# Runner (for environments without pytest)
# ===========================================================================

def _run_all():
    """Run all test methods, return (passed, failed)."""
    test_classes = [
        TestRepoStructure,
        TestNoHardcodedPaths,
        TestCLIHelp,
        TestRobothorConfigPipeline,
        TestMesataskInstructionGeneration,
        TestEvalPipelineMock,
    ]
    passed = failed = 0
    for cls in test_classes:
        inst = cls()
        methods = [m for m in dir(inst) if m.startswith("test_")]
        for m in methods:
            try:
                getattr(inst, m)()
                print(f"  PASS  {cls.__name__}::{m}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {cls.__name__}::{m}: {e}")
                failed += 1
    return passed, failed


if __name__ == "__main__":
    p, f = _run_all()
    print(f"\nTotal: {p} passed, {f} failed")
    sys.exit(1 if f else 0)
