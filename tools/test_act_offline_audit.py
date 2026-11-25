#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline unit tests for tools/act_offline_audit.py.

No robot, CAN, camera, GPU, checkpoint weights or pyarrow are required. The
inference-loop test uses a synthetic in-memory dataset and a Fake Policy.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import act_offline_audit as audit  # noqa: E402


CAMERAS_3 = [
    "observation.images.top",
    "observation.images.wrist",
    "observation.images.wrist_depth",
]


def make_info(cameras=None, state_dim=7, action_dim=7, total_episodes=2, total_frames=4):
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": [f"joint_{i + 1}.deg" for i in range(state_dim)],
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": [f"joint_{i + 1}.deg" for i in range(action_dim)],
        },
    }
    for key in cameras if cameras is not None else CAMERAS_3:
        features[key] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
        }
    return {
        "codebase_version": "v3.0",
        "robot_type": "piper_pika",
        "fps": 30,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "features": features,
    }


def make_config(cameras=None, state_dim=7, action_dim=7):
    inputs = {
        "observation.state": {"type": "STATE", "shape": [state_dim]},
    }
    for key in cameras if cameras is not None else CAMERAS_3:
        inputs[key] = {"type": "VISUAL", "shape": [3, 480, 640]}
    return {
        "type": "act",
        "input_features": inputs,
        "output_features": {"action": {"type": "ACTION", "shape": [action_dim]}},
        "normalization_mapping": {"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
    }


def make_stats(info):
    stats = {}
    for key, feature in info["features"].items():
        dim = 3 if feature["dtype"] == "video" else int(feature["shape"][0])
        stats[key] = {
            "mean": [0.0] * dim,
            "std": [1.0] * dim,
            "min": [0.0] * dim,
            "max": [1.0] * dim,
        }
    return stats


def make_records(preds, states=None, actions=None, episodes=None):
    preds = np.asarray(preds, dtype=np.float32)
    n = preds.shape[0]
    states = np.zeros_like(preds) if states is None else np.asarray(states, dtype=np.float32)
    actions = preds + 1.0 if actions is None else np.asarray(actions, dtype=np.float32)
    episodes = [0] * n if episodes is None else list(episodes)
    return [
        {
            "episode": int(episodes[i]),
            "frame": i,
            "dataset_frame": i,
            "state": states[i].copy(),
            "action": actions[i].copy(),
            "pred": preds[i].copy(),
        }
        for i in range(n)
    ]


class TestParsingAndNames(unittest.TestCase):
    def test_parse_episodes_sorted_unique(self):
        self.assertEqual(audit.parse_episodes("2, 0,2,1"), [0, 1, 2])

    def test_parse_episodes_rejects_bad_input(self):
        for bad in ("", "a", "-1", "1,,x"):
            with self.assertRaises(ValueError):
                audit.parse_episodes(bad)

    def test_find_joint_indices_by_name(self):
        names = ["joint_1.deg", "joint_2.deg", "joint_3.deg", "gripper.mm"]
        self.assertEqual(audit.find_joint_indices(names, 4), (1, 2))
        self.assertEqual(audit.find_joint_indices(["J1", "J2", "J3"], 3), (1, 2))

    def test_find_joint_indices_fallback_and_error(self):
        self.assertEqual(audit.find_joint_indices(None, 7), (1, 2))
        with self.assertRaises(ValueError):
            audit.find_joint_indices(None, 2)


class TestBoundaryStats(unittest.TestCase):
    def test_counts_ratios_first_frame_and_extremes(self):
        values = np.zeros((4, 7), dtype=np.float32)
        values[:, 1] = [0.4, -1.0, 0.6, 2.0]  # J2
        values[:, 2] = [-0.4, 0.5, -0.6, -2.0]  # J3
        stats = audit.boundary_stats(values, 1, 2)
        self.assertEqual(stats["frames"], 4)
        self.assertEqual(stats["pinned"]["count"], 2)
        self.assertAlmostEqual(stats["pinned"]["ratio"], 0.5)
        self.assertEqual(stats["pinned"]["first_frame"], 0)
        self.assertEqual(stats["zero_cross"]["count"], 1)
        self.assertEqual(stats["zero_cross"]["first_frame"], 1)
        self.assertEqual(stats["j2_negative"]["count"], 1)
        self.assertAlmostEqual(stats["j2_negative"]["extreme"], -1.0)
        self.assertEqual(stats["j3_positive"]["count"], 1)
        self.assertAlmostEqual(stats["j3_positive"]["extreme"], 0.5)
        self.assertAlmostEqual(stats["j2_min"], -1.0)
        self.assertAlmostEqual(stats["j3_max"], 0.5)

    def test_per_episode_breakdown(self):
        values = np.zeros((4, 7), dtype=np.float32)
        values[:, 1] = [0.1, 1.0, 0.2, 2.0]
        values[:, 2] = [0.0, -1.0, 0.0, -2.0]
        stats = audit.boundary_stats(
            values, 1, 2, episodes=[0, 0, 1, 1], frames=[0, 1, 0, 1]
        )
        self.assertEqual(stats["per_episode"]["0"]["frames"], 2)
        self.assertEqual(stats["per_episode"]["1"]["frames"], 2)
        self.assertEqual(stats["per_episode"]["0"]["pinned"]["count"], 1)
        self.assertEqual(stats["per_episode"]["1"]["pinned"]["count"], 1)


class TestDatasetSummary(unittest.TestCase):
    def setUp(self):
        self.info = make_info(total_episodes=2, total_frames=4)
        self.state = np.array(
            [
                [0, 0.4, -0.4, 0, 0, 0, 10],
                [0, 1.0, -1.0, 0, 0, 0, 20],
                [0, 2.0, 2.0, 0, 0, 0, 30],
                [0, -1.0, -2.0, 0, 0, 0, 40],
            ],
            dtype=np.float32,
        )
        self.action = self.state + 0.5
        self.episodes = np.array([0, 0, 1, 1], dtype=np.int64)
        self.frames = np.array([0, 1, 0, 1], dtype=np.int64)

    def test_summary_integrity_first_frames_and_extrema(self):
        summary = audit.summarize_arrays(
            self.state, self.action, self.episodes, self.frames, info=self.info
        )
        self.assertEqual(summary["total_episodes"], 2)
        self.assertEqual(summary["total_frames"], 4)
        self.assertTrue(summary["integrity"]["episodes_match"])
        self.assertTrue(summary["integrity"]["frames_match"])
        self.assertAlmostEqual(summary["first_frames"]["0"]["state"]["joint_2.deg"], 0.4)
        self.assertAlmostEqual(summary["first_frames"]["1"]["state"]["joint_2.deg"], 2.0)
        self.assertAlmostEqual(summary["state_extrema"][1]["min"], -1.0)
        self.assertAlmostEqual(summary["state_extrema"][1]["max"], 2.0)
        self.assertEqual(summary["boundary"]["state"]["pinned"]["count"], 3)
        self.assertIn("per_episode", summary["boundary"]["state"])

    def test_summary_detects_info_mismatch(self):
        info = make_info(total_episodes=9, total_frames=99)
        summary = audit.summarize_arrays(self.state, self.action, self.episodes, self.frames, info=info)
        self.assertFalse(summary["integrity"]["episodes_match"])
        self.assertFalse(summary["integrity"]["frames_match"])

    def test_summary_rejects_nonfinite(self):
        bad = self.state.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            audit.summarize_arrays(bad, self.action, self.episodes, self.frames, info=self.info)


class TestContract(unittest.TestCase):
    def test_full_match_is_strict(self):
        info = make_info()
        contract = audit.check_contract(info, make_config(), make_stats(info))
        self.assertTrue(contract["ok"])
        self.assertTrue(contract["strict_match"])
        self.assertEqual(contract["missing_cameras"], [])
        self.assertEqual(contract["extra_cameras"], [])
        self.assertTrue(all(cam["status"] == "match" for cam in contract["cameras"]))

    def test_dataset_has_extra_third_camera(self):
        info = make_info()  # 3 cameras
        config = make_config(cameras=CAMERAS_3[:2])  # 2 cameras
        contract = audit.check_contract(info, config, make_stats(info))
        self.assertTrue(contract["ok"])
        self.assertFalse(contract["strict_match"])
        self.assertEqual(contract["extra_cameras"], ["observation.images.wrist_depth"])
        self.assertTrue(any("2-camera vs 3-camera" in w for w in contract["warnings"]))

    def test_checkpoint_requires_missing_camera(self):
        info = make_info(cameras=CAMERAS_3[:2])
        config = make_config(cameras=CAMERAS_3)
        contract = audit.check_contract(info, config, make_stats(info))
        self.assertFalse(contract["ok"])
        self.assertFalse(contract["strict_match"])
        self.assertEqual(contract["missing_cameras"], ["observation.images.wrist_depth"])

    def test_camera_shape_mismatch(self):
        info = make_info()
        config = make_config()
        config["input_features"]["observation.images.top"]["shape"] = [3, 240, 320]
        contract = audit.check_contract(info, config, make_stats(info))
        self.assertFalse(contract["ok"])
        top = next(cam for cam in contract["cameras"] if cam["key"] == "observation.images.top")
        self.assertEqual(top["status"], "shape_mismatch")
        self.assertEqual(top["checkpoint_hwc"], (240, 320, 3))

    def test_state_and_action_dim_mismatch(self):
        info = make_info()
        contract = audit.check_contract(info, make_config(state_dim=6, action_dim=8), make_stats(info))
        self.assertFalse(contract["ok"])
        self.assertEqual(contract["state"]["status"], "shape_mismatch")
        self.assertEqual(contract["action"]["status"], "shape_mismatch")

    def test_dataset_without_stats_is_warning_only(self):
        info = make_info()
        contract = audit.check_contract(info, make_config(), stats={})
        self.assertTrue(contract["ok"])
        self.assertTrue(any("stats.json" in w for w in contract["warnings"]))


class FakePolicy:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class TestOpenLoopInference(unittest.TestCase):
    def make_dataset(self, episode=0, frames=3):
        samples = []
        for i in range(frames):
            state = np.arange(7, dtype=np.float32) + i
            samples.append(
                {
                    "episode_index": np.array([episode]),
                    "frame_index": np.array([i]),
                    "observation.state": state.copy(),
                    "action": state + 1.0,
                    "observation.images.top": np.full((3, 4, 5), 0.5, dtype=np.float32),
                    "observation.images.wrist": np.full((4, 5, 3), 128, dtype=np.uint8),
                }
            )
        return samples

    def test_reset_per_episode_and_image_conversion(self):
        policy = FakePolicy()
        seen = []

        def predict(obs):
            seen.append(obs)
            self.assertEqual(obs["observation.images.top"].dtype, np.uint8)
            self.assertEqual(obs["observation.images.top"].shape, (4, 5, 3))
            self.assertLessEqual(int(obs["observation.images.top"].max()), 255)
            return np.ones(7, dtype=np.float32)

        audit_obj = {}
        records = audit.run_open_loop_episode(
            self.make_dataset(episode=0),
            episode=0,
            max_frames=2,
            expected_inputs=["observation.state", "observation.images.top", "observation.images.wrist"],
            predict_fn=predict,
            reset_fn=policy.reset,
            image_audit=audit_obj,
        )
        self.assertEqual(policy.reset_count, 1)
        self.assertEqual(len(records), 2)
        self.assertEqual(len(seen), 2)
        self.assertEqual(audit_obj["observation.images.top"]["dataset_dtype"], "float32")
        self.assertEqual(audit_obj["observation.images.top"]["deploy_dtype"], "uint8")

        audit.run_open_loop_episode(
            self.make_dataset(episode=1),
            episode=1,
            max_frames=1,
            expected_inputs=["observation.state"],
            predict_fn=lambda obs: np.ones(7, dtype=np.float32),
            reset_fn=policy.reset,
        )
        self.assertEqual(policy.reset_count, 2)

    def test_episode_mismatch_is_rejected(self):
        with self.assertRaises(RuntimeError):
            audit.run_open_loop_episode(
                self.make_dataset(episode=0, frames=1),
                episode=7,
                max_frames=1,
                expected_inputs=["observation.state"],
                predict_fn=lambda obs: np.ones(7, dtype=np.float32),
            )

    def test_missing_camera_is_rejected(self):
        with self.assertRaises(ValueError):
            audit.run_open_loop_episode(
                self.make_dataset(frames=1),
                episode=0,
                max_frames=1,
                expected_inputs=["observation.state", "observation.images.missing"],
                predict_fn=lambda obs: np.ones(7, dtype=np.float32),
            )


class TestActionMetrics(unittest.TestCase):
    def setUp(self):
        self.preds = np.array(
            [
                [0, -1.0, 1.0, 0, 0, 0, 10],
                [0, 0.3, -0.2, 0, 0, 0, 20],
                [0, 2.0, 2.0, 0, 0, 0, 30],
            ],
            dtype=np.float32,
        )
        self.states = np.zeros_like(self.preds)
        self.records = make_records(self.preds, states=self.states, actions=self.preds + 1.0)

    def test_mae_max_error_and_bounds(self):
        metrics = audit.compute_action_metrics(self.records)
        self.assertAlmostEqual(metrics["per_joint"]["J2"]["mae"], 1.0)
        self.assertAlmostEqual(metrics["per_joint"]["J2"]["max_abs_err"], 1.0)
        self.assertEqual(metrics["bounds"]["J2 < 0.0"]["count"], 1)
        self.assertEqual(metrics["bounds"]["J2 < 0.0"]["first"], {"episode": 0, "frame": 0})
        self.assertAlmostEqual(metrics["bounds"]["J2 < 0.0"]["extreme"], -1.0)
        self.assertEqual(metrics["bounds"]["J2 < 0.5"]["count"], 2)
        self.assertEqual(metrics["bounds"]["J3 > 0.0"]["count"], 2)
        self.assertEqual(metrics["bounds"]["J3 > -0.5"]["count"], 3)
        self.assertAlmostEqual(metrics["bounds"]["J3 > 0.0"]["extreme"], 2.0)

    def test_first_jump_and_adjacent_step(self):
        metrics = audit.compute_action_metrics(self.records)
        self.assertAlmostEqual(metrics["first_frame_jump_max"]["J2"], 1.0)
        self.assertAlmostEqual(metrics["first_frame_jump_max"]["J3"], 1.0)
        self.assertAlmostEqual(metrics["adjacent_max_step"]["J2"]["max_step"], 1.7)
        self.assertAlmostEqual(metrics["adjacent_max_step"]["J3"]["max_step"], 2.2)
        self.assertEqual(metrics["episodes"]["0"]["frames"], 3)

    def test_adjacent_step_does_not_cross_episode_boundary(self):
        preds = np.array(
            [
                [0, 0.0, 0.0, 0, 0, 0, 0],
                [0, 30.0, 30.0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        records = make_records(preds, episodes=[0, 1])
        metrics = audit.compute_action_metrics(records)
        self.assertAlmostEqual(metrics["adjacent_max_step"]["J2"]["max_step"], 0.0)
        self.assertEqual(metrics["episodes"]["1"]["frames"], 1)


class TestImageAndGrid(unittest.TestCase):
    def test_rgb_conversion_chw_float_and_hwc_uint8(self):
        chw = np.full((3, 4, 5), 0.5, dtype=np.float32)
        converted = audit._rgb_hwc_uint8(chw, "camera")
        self.assertEqual(converted.shape, (4, 5, 3))
        self.assertEqual(converted.dtype, np.uint8)
        self.assertTrue(126 <= int(converted[0, 0, 0]) <= 128)

        hwc = np.full((4, 5, 3), 200, dtype=np.uint8)
        np.testing.assert_array_equal(audit._rgb_hwc_uint8(hwc, "camera"), hwc)

    def test_rgb_conversion_rejects_bad_range(self):
        with self.assertRaises(ValueError):
            audit._rgb_hwc_uint8(np.full((3, 4, 5), 999.0, dtype=np.float32), "camera")

    def test_build_and_save_visual_grid(self):
        try:
            from PIL import Image  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("Pillow not installed")
        images = [
            [np.zeros((12, 16, 3), dtype=np.uint8), np.full((12, 16, 3), 80, dtype=np.uint8)],
            [np.full((12, 16, 3), 160, dtype=np.uint8), np.full((12, 16, 3), 255, dtype=np.uint8)],
        ]
        grid = audit.build_visual_grid(
            images,
            row_labels=["episode 0 frame 0", "episode 0 frame 1"],
            column_labels=["observation.images.top", "observation.images.wrist"],
        )
        expected_w = 2 * 16 + 3 * 8
        expected_h = 2 * (12 + 26) + 3 * 8
        self.assertEqual((grid.width, grid.height), (expected_w, expected_h))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grid.png"
            grid.save(path)
            self.assertTrue(path.is_file())


class TestPolicyPathResolution(unittest.TestCase):
    def _write_config(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"type": "act"}), encoding="utf-8")

    def test_nested_pretrained_model_is_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoints" / "last"
            self._write_config(checkpoint / "pretrained_model" / "config.json")
            policy_dir, config, config_path = audit.resolve_policy_dir(checkpoint)
            self.assertEqual(policy_dir, (checkpoint / "pretrained_model").resolve())
            self.assertEqual(config["type"], "act")
            self.assertEqual(config_path, (checkpoint / "pretrained_model" / "config.json").resolve())

    def test_direct_pretrained_model_and_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            pretrained = Path(tmp) / "pretrained_model"
            self._write_config(pretrained / "config.json")
            self.assertEqual(audit.resolve_policy_dir(pretrained)[0], pretrained.resolve())
            self.assertEqual(audit.resolve_policy_dir(pretrained / "config.json")[0], pretrained.resolve())

    def test_missing_config_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                audit.resolve_policy_dir(Path(tmp) / "empty")


class TestReportsAndSafety(unittest.TestCase):
    def test_write_json_and_csv(self):
        records = make_records(np.zeros((2, 7), dtype=np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "report.json"
            csv_path = Path(tmp) / "pred.csv"
            audit.write_json_report(json_path, {"ok": True, "frames": np.int64(2)})
            audit.write_predictions_csv(csv_path, records)
            report = json.loads(json_path.read_text())
            self.assertEqual(report["frames"], 2)
            with csv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("pred_J2", rows[0])
            self.assertIn("abs_err_J2", rows[0])

    def test_output_dir_cannot_be_inside_dataset_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            policy = root / "policy"
            dataset.mkdir()
            policy.mkdir()
            with self.assertRaises(ValueError):
                audit.ensure_safe_output_dir(dataset / "out", dataset, policy)
            with self.assertRaises(ValueError):
                audit.ensure_safe_output_dir(policy / "out", dataset, policy)
            with self.assertRaises(ValueError):
                audit.ensure_safe_output_dir(
                    policy / "audit_out", dataset, policy / "pretrained_model", protected_paths=[policy]
                )
            out = audit.ensure_safe_output_dir(root / "audit_out", dataset, policy)
            self.assertTrue(out.is_dir())


if __name__ == "__main__":
    unittest.main(verbosity=2)
