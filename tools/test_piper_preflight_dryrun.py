#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fully offline tests for tools/piper_can_codec.py and piper_preflight_dryrun.py.

No real CAN bus, camera, checkpoint, GPU, pyarrow, or LeRobot import is needed.
"""
from __future__ import annotations

import ast
import csv
import io
import json
import os
import socket
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import piper_can_codec as codec  # noqa: E402
import piper_preflight_dryrun as preflight  # noqa: E402


def make_spec(name, path="/dev/video0", camera_type="opencv", width=640, height=480):
    return preflight.CameraSpec(
        name=name,
        camera_type=camera_type,
        index_or_path=path,
        width=width,
        height=height,
        fps=30.0,
        fourcc="MJPG",
        backend="v4l2",
        reader="direct",
        warmup_s=0.0,
        raw={},
    )


def write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class StepClock:
    def __init__(self, step=0.05, start=0.0):
        self.value = start
        self.step = step

    def __call__(self):
        current = self.value
        self.value += self.step
        return current


class SequenceClock:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        if not self.values:
            raise AssertionError("clock sequence exhausted")
        return self.values.pop(0)


class FakeSocket:
    """Minimal recv-only CAN socket double; deliberately has no TX methods."""

    def __init__(self, raw_frames):
        self.raw_frames = list(raw_frames)
        self.bound = None
        self.filter = None
        self.timeout = None
        self.closed = False

    def bind(self, address):
        self.bound = address

    def setsockopt(self, level, option, value):
        self.filter = value

    def settimeout(self, timeout):
        self.timeout = timeout

    def recv(self, size):
        if self.raw_frames:
            return self.raw_frames.pop(0)
        raise socket.timeout("no frame")

    def close(self):
        self.closed = True


class FakeCamera:
    instances = []

    def __init__(self, spec):
        self.spec = spec
        self.opened = False
        self.released = False
        self.frame = np.zeros((4, 5, 3), dtype=np.uint8)
        FakeCamera.instances.append(self)

    def open(self):
        self.opened = True

    def read(self):
        return self.frame

    def release(self):
        self.released = True


class FakePolicy:
    def __init__(self, predictions=None):
        self.reset_count = 0
        self.predictions = list(predictions or [])
        self.index = 0

    def reset(self):
        self.reset_count += 1

    def predict(self, observation):
        value = self.predictions[self.index]
        self.index += 1
        return value


class TestCanCodec(unittest.TestCase):
    def test_joint_pair_roundtrip(self):
        payload = codec.encode_joint_pair(1_234, -5_678)
        self.assertEqual(codec.decode_joint_pair(payload), (1_234, -5_678))
        self.assertEqual(len(payload), 8)

    def test_gripper_roundtrip(self):
        payload = codec.encode_gripper_frame(12_345, 678, 0x01, 0x00)
        frame = codec.decode_gripper_frame(payload)
        self.assertEqual(frame.target_um, 12_345)
        self.assertEqual(frame.effort_mnm, 678)
        self.assertEqual(frame.status_code, 0x01)
        self.assertEqual(frame.set_zero, 0x00)
        self.assertEqual(codec.decode_gripper(payload), 12_345)

    def test_status_decode_and_fault_bits(self):
        data = bytes([0x01, 0x00, 0x01, 0x02, 0x00, 0x07, 0b00100101, 0b01000001])
        status = codec.decode_status(data)
        self.assertEqual(status.ctrl_mode, 0x01)
        self.assertEqual(status.ctrl_mode_name, "CAN_CTRL")
        self.assertEqual(status.arm_status_name, "NORMAL")
        self.assertEqual(status.mode_name, "MOVE_J")
        self.assertEqual(status.trajectory_num, 7)
        self.assertEqual(status.joint_limit_error_bits, 0b00100101)
        self.assertEqual(status.joint_comm_error_bits, 0b00000001)
        self.assertEqual(status.error_code, 0x4125)
        self.assertEqual(status.as_dict()["error_code_hex"], "0x4125")
        self.assertIn("joint_limit_error_bits", status.as_dict())

    def test_feedback_accumulation(self):
        frames = [
            codec.CanFrame(0.0, codec.ID_JOINT_FEEDBACK_12, codec.encode_joint_pair(1_000, 2_000)),
            codec.CanFrame(0.0, codec.ID_JOINT_FEEDBACK_34, codec.encode_joint_pair(3_000, 4_000)),
            codec.CanFrame(0.0, codec.ID_JOINT_FEEDBACK_56, codec.encode_joint_pair(5_000, 6_000)),
            codec.CanFrame(0.0, codec.ID_GRIPPER_FEEDBACK, codec.encode_gripper_frame(20_000, 0)),
            codec.CanFrame(0.0, codec.ID_STATUS_FEEDBACK, bytes(8)),
        ]
        feedback = codec.decode_feedback_frames(frames)
        self.assertTrue(feedback.complete)
        self.assertEqual(feedback.joints_deg, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertAlmostEqual(feedback.gripper_mm, 20.0)
        self.assertEqual(feedback.state_vector(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 20.0])
        self.assertEqual(sum(feedback.counts.values()), 5)

    def test_malformed_feedback_is_counted(self):
        frames = [codec.CanFrame(0.0, codec.ID_JOINT_FEEDBACK_12, b"\x00")]
        feedback = codec.decode_feedback_frames(frames)
        self.assertEqual(feedback.malformed, 1)
        self.assertFalse(feedback.complete)

    def test_nominal_limits_match_teleop(self):
        self.assertEqual(codec.JOINT_LIMITS_DEG[0], (-150.0, 150.0))
        self.assertEqual(codec.JOINT_LIMITS_DEG[1], (0.0, 180.0))
        self.assertEqual(codec.JOINT_LIMITS_DEG[2], (-170.0, 0.0))
        self.assertEqual(codec.JOINT_LIMITS_DEG[5], (-120.0, 120.0))

    def test_can_frame_roundtrip(self):
        raw = codec.encode_can_frame(codec.ID_JOINT_FEEDBACK_12, b"\x01\x02\x03")
        can_id, payload = codec.parse_can_frame(raw)
        self.assertEqual(can_id, codec.ID_JOINT_FEEDBACK_12)
        self.assertEqual(payload, b"\x01\x02\x03")

    def test_bad_lengths_raise(self):
        for function, data in (
            (codec.decode_joint_pair, b"\x00"),
            (codec.decode_gripper, b"\x00"),
            (codec.decode_status, b"\x00"),
            (codec.parse_can_frame, b"\x00"),
        ):
            with self.assertRaises(ValueError):
                function(data)

    def test_codec_imports_only_stdlib_protocol_modules(self):
        source = Path(codec.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue(imported <= {"__future__", "struct", "dataclasses", "typing"})
        for forbidden in ("import socket", "import rclpy", "import piper_pika_teleop"):
            self.assertNotIn(forbidden, source)


class TestZeroTxConstraint(unittest.TestCase):
    def test_preflight_has_no_tx_tokens(self):
        source = Path(preflight.__file__).read_text(encoding="utf-8")
        self.assertNotIn(".send(", source)
        for token in ("0x151", "0x155", "0x156", "0x157", "0x159", "0x470", "0x471"):
            self.assertNotIn(token, source)

    def test_preflight_ast_has_no_send_calls(self):
        tree = ast.parse(Path(preflight.__file__).read_text(encoding="utf-8"))
        forbidden = {"send", "sendto", "sendmsg", "sendall"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, forbidden)
        self.assertFalse(hasattr(preflight.CanSocketReader, "send"))

    def test_reader_binds_and_only_receives(self):
        raw = [
            codec.encode_can_frame(codec.ID_JOINT_FEEDBACK_12, codec.encode_joint_pair(1_000, 2_000)),
            codec.encode_can_frame(codec.ID_JOINT_FEEDBACK_34, codec.encode_joint_pair(3_000, 4_000)),
        ]
        fake = FakeSocket(raw)
        reader = preflight.CanSocketReader(
            "can_test", socket_factory=lambda: fake, recv_timeout_s=0.01
        )
        frames = None
        with reader:
            frames = reader.read_until(0.2, clock=StepClock(step=0.05))
        self.assertEqual(fake.bound, ("can_test",))
        self.assertIsNotNone(fake.filter)
        self.assertEqual(len(fake.filter), len(codec.FEEDBACK_IDS) * 8)
        self.assertTrue(fake.closed)
        self.assertEqual([frame.arbitration_id for frame in frames],
                         [codec.ID_JOINT_FEEDBACK_12, codec.ID_JOINT_FEEDBACK_34])

    def test_summarize_frames_reports_zero_tx_mode(self):
        frames = [
            codec.CanFrame(10.0, codec.ID_JOINT_FEEDBACK_12, codec.encode_joint_pair(1_000, 2_000)),
            codec.CanFrame(10.5, codec.ID_JOINT_FEEDBACK_34, codec.encode_joint_pair(3_000, 4_000)),
        ]
        summary = preflight.summarize_can_frames(frames, 2.0)
        self.assertEqual(summary["mode"], "READ_ONLY_RX_ONLY")
        self.assertEqual(summary["frames"], 2)
        self.assertFalse(summary["complete"])
        self.assertIn("0x2A1", summary["missing_feedback_ids"])


class TestCameraConfig(unittest.TestCase):
    def test_load_two_cameras(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_yaml(
                Path(tmp) / "cameras.yaml",
                """
top:
  type: opencv
  index_or_path: /dev/video0
  width: 640
  height: 480
  fps: 30
wrist:
  type: opencv
  index_or_path: /dev/v4l/by-id/usb-wrist-video-index0
  width: 640
  height: 480
  fps: 30
""",
            )
            specs = preflight.load_camera_config(path)
            self.assertEqual(sorted(specs), ["top", "wrist"])
            self.assertEqual(specs["top"].width, 640)
            self.assertEqual(specs["wrist"].index_or_path, "/dev/v4l/by-id/usb-wrist-video-index0")

    def test_volatile_path_warning(self):
        specs = {
            "top": make_spec("top", "/dev/video0"),
            "wrist": make_spec("wrist", "/dev/v4l/by-id/usb-wrist-video-index0"),
        }
        result = preflight.validate_camera_config(specs)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["volatile_paths"], [{"camera": "top", "path": "/dev/video0"}])
        self.assertTrue(any("volatile" in warning for warning in result["warnings"]))
        self.assertFalse(preflight.is_volatile_video_path("/dev/v4l/by-id/cam"))

    def test_missing_required_camera(self):
        result = preflight.validate_camera_config({"top": make_spec("top")})
        self.assertTrue(result["errors"])
        self.assertTrue(any("wrist" in error for error in result["errors"]))

    def test_duplicate_device_is_error(self):
        specs = {
            "top": make_spec("top", "/dev/video0"),
            "wrist": make_spec("wrist", "/dev/video0"),
        }
        result = preflight.validate_camera_config(specs)
        self.assertTrue(any("same device" in error for error in result["errors"]))

    def test_wrong_resolution_is_warning(self):
        specs = {
            "top": make_spec("top", "/dev/video0", width=1280, height=720),
            "wrist": make_spec("wrist", "/dev/video6"),
        }
        result = preflight.validate_camera_config(specs)
        self.assertTrue(any("640x480" in warning for warning in result["warnings"]))

    def test_enumerate_v4l_by_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev = root / "dev"
            dev.mkdir()
            (dev / "video0").touch()
            links = root / "by-id"
            links.mkdir()
            os.symlink("../dev/video0", links / "usb-cam-video-index0")
            rows = preflight.enumerate_v4l_by_id(links)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["name"], "usb-cam-video-index0")
            self.assertEqual(rows[0]["video_device"], "/dev/video0")
            self.assertEqual(preflight.enumerate_v4l_by_id(root / "missing"), [])

    def test_list_mode_uses_injected_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev = root / "dev"
            dev.mkdir()
            (dev / "video2").touch()
            links = root / "by-id"
            links.mkdir()
            os.symlink("../dev/video2", links / "cam-video-index0")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = preflight.main(["--list-v4l-by-id", "--v4l-base", str(links)])
            self.assertEqual(code, 0)
            self.assertIn("/dev/video2", buffer.getvalue())


class TestCameraCapture(unittest.TestCase):
    def test_capture_converts_bgr_and_releases(self):
        FakeCamera.instances = []
        specs = {
            "top": make_spec("top", width=5, height=4),
            "wrist": make_spec("wrist", width=5, height=4),
        }
        frames = {
            "top": np.full((4, 5, 3), 0, dtype=np.uint8),
            "wrist": np.full((4, 5, 3), 0, dtype=np.uint8),
        }
        frames["top"][..., 0] = 10
        frames["top"][..., 1] = 20
        frames["top"][..., 2] = 30

        def factory(spec):
            camera = FakeCamera(spec)
            camera.frame = frames[spec.name]
            return camera

        images, stats = preflight.capture_live_images(
            specs, camera_factory=factory, expected_size=(5, 4)
        )
        self.assertEqual(images["top"][0, 0].tolist(), [30, 20, 10])
        self.assertEqual(stats["top"]["shape"], [4, 5, 3])
        self.assertEqual(len(FakeCamera.instances), 2)
        self.assertTrue(all(camera.released for camera in FakeCamera.instances))

    def test_wrong_live_resolution_fails(self):
        specs = {"top": make_spec("top", width=5, height=4)}
        with self.assertRaises(preflight.CameraFrameError):
            preflight.capture_live_images(
                specs, camera_factory=FakeCamera, expected=("top",), expected_size=(6, 4)
            )

    def test_non_uint8_frame_fails(self):
        specs = {"top": make_spec("top", width=5, height=4)}

        def factory(spec):
            camera = FakeCamera(spec)
            camera.frame = np.zeros((4, 5, 3), dtype=np.float32)
            return camera

        with self.assertRaises(preflight.CameraFrameError):
            preflight.capture_live_images(
                specs, camera_factory=factory, expected=("top",), expected_size=(5, 4)
            )

    def test_frame_stats(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[..., 0] = 10
        image[..., 1] = 20
        image[..., 2] = 30
        stats = preflight.frame_stats(image)
        self.assertEqual(stats["mean_rgb"], [10.0, 20.0, 30.0])
        self.assertEqual(stats["std_rgb"], [0.0, 0.0, 0.0])


class TestComparisonGrid(unittest.TestCase):
    def test_grid_layout_and_labels(self):
        try:
            from PIL import Image  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("Pillow not installed")
        train = {
            "top": np.full((12, 16, 3), 40, dtype=np.uint8),
            "wrist": np.full((12, 16, 3), 80, dtype=np.uint8),
        }
        live = {
            "top": np.full((12, 16, 3), 120, dtype=np.uint8),
            "wrist": np.full((12, 16, 3), 200, dtype=np.uint8),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live_vs_train_f0.png"
            result = preflight.make_comparison_grid(train, live, path)
            self.assertTrue(path.is_file())
            self.assertEqual(result["width"], 2 * 16 + 3 * 6)
            self.assertEqual(result["height"], 2 * (12 + 48) + 3 * 6)
            self.assertEqual(result["stats"]["train"]["top"]["mean_rgb"], [40.0, 40.0, 40.0])
            self.assertEqual(result["stats"]["live"]["wrist"]["mean_rgb"], [200.0, 200.0, 200.0])

    def test_missing_camera_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                preflight.make_comparison_grid({}, {}, Path(tmp) / "x.png")

    def test_image_to_rgb_from_chw_float(self):
        image = np.full((3, 4, 5), 0.5, dtype=np.float32)
        result = preflight.image_to_rgb_uint8(image)
        self.assertEqual(result.shape, (4, 5, 3))
        self.assertEqual(result.dtype, np.uint8)


class TestDatasetStatistics(unittest.TestCase):
    def test_first_frame_medians(self):
        state = np.array(
            [
                [10, 20],
                [100, 200],
                [20, 40],
                [200, 400],
                [30, 60],
                [300, 600],
            ],
            dtype=np.float64,
        )
        episodes = np.array([0, 0, 1, 1, 2, 2])
        frames = np.array([0, 1, 0, 1, 0, 1])
        result = preflight.first_frame_medians(state, episodes, frames, ["a", "b"])
        self.assertEqual(result["episodes"], 3)
        self.assertEqual(result["median"], [20.0, 40.0])
        self.assertEqual(result["median_by_name"], {"a": 20.0, "b": 40.0})
        self.assertEqual(result["first_frame_indices"]["2"], 0)

    def test_joint_deviation(self):
        result = preflight.compute_joint_deviation([1.0, 2.0], [0.5, 3.0], ["a", "b"])
        self.assertEqual(result["a"]["delta"], 0.5)
        self.assertEqual(result["b"]["delta"], -1.0)

    def test_zero_offset_j2_j3(self):
        joints = [0.0, 0.2, -0.1, 0.0, 0.0, 0.0]
        result = preflight.zero_offset_report(joints)
        self.assertAlmostEqual(result["J2"]["offset_from_zero_deg"], 0.2)
        self.assertTrue(result["J2"]["inside_nominal_range"])
        self.assertTrue(result["J2"]["near_zero_pinned"])
        self.assertAlmostEqual(result["J2"]["limit_margin_deg"], 0.2)
        self.assertAlmostEqual(result["J3"]["offset_from_zero_deg"], -0.1)
        self.assertTrue(result["J3"]["near_zero_pinned"])
        self.assertAlmostEqual(result["J3"]["limit_margin_deg"], 0.1)

    def test_zero_offset_out_of_range(self):
        joints = [0.0, -1.0, 1.0, 0.0, 0.0, 0.0]
        result = preflight.zero_offset_report(joints)
        self.assertFalse(result["J2"]["inside_nominal_range"])
        self.assertFalse(result["J3"]["inside_nominal_range"])


class TestDryRun(unittest.TestCase):
    def make_inputs(self):
        state = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0], dtype=np.float64)
        images = {
            "top": np.full((4, 5, 3), 10, dtype=np.uint8),
            "wrist": np.full((4, 5, 3), 20, dtype=np.uint8),
        }
        expected = ["observation.state", "observation.images.top", "observation.images.wrist"]
        return state, images, expected

    def test_latency_and_trajectory_stats(self):
        state, images, expected = self.make_inputs()
        predictions = [
            np.array([0.0, -1.0, 1.0, 3.0, 4.0, 5.0, 10.0]),
            np.array([0.0, 0.5, -0.2, 3.0, 4.0, 5.0, 10.0]),
            np.array([0.0, 2.0, 2.0, 3.0, 4.0, 5.0, 10.0]),
        ]
        calls = {"count": 0}
        policy = FakePolicy(predictions)

        def predict(observation):
            self.assertEqual(observation["observation.state"].shape, (7,))
            self.assertEqual(observation["observation.images.top"].dtype, np.uint8)
            value = policy.predict(observation)
            calls["count"] += 1
            return value

        clock = SequenceClock([0.0, 0.010, 0.010, 0.030, 0.030, 0.090])
        metrics = preflight.run_dry_run(
            state, images, expected, predict, steps=3, reset_fn=policy.reset, clock=clock
        )
        self.assertEqual(policy.reset_count, 1)
        self.assertEqual(metrics["steps"], 3)
        self.assertAlmostEqual(metrics["latency_ms"]["p50"], 20.0)
        self.assertAlmostEqual(metrics["latency_ms"]["p95"], 56.0)
        self.assertAlmostEqual(metrics["latency_ms"]["max"], 60.0)
        self.assertAlmostEqual(metrics["latency_ms"]["mean"], 30.0)
        self.assertAlmostEqual(metrics["fps"]["closed_loop_mean"], 1000.0 / 30.0)
        self.assertAlmostEqual(metrics["first_frame_jump"]["J2"], 2.0)
        self.assertAlmostEqual(metrics["j2_max_deg"], 2.0)
        self.assertEqual(metrics["j2_max_step"], 2)
        self.assertAlmostEqual(metrics["j2_delta_from_state_deg"], 1.0)
        self.assertAlmostEqual(metrics["j2_reach_deg"], 1.0)
        self.assertAlmostEqual(metrics["j3_min_deg"], -0.2)
        self.assertAlmostEqual(metrics["j3_delta_from_state_deg"], -2.2)
        self.assertAlmostEqual(metrics["j3_reach_deg"], 2.2)
        self.assertTrue(metrics["trajectory_motion_detected"])
        self.assertEqual(len(metrics["trace"]), 3)

    def test_forward_grasp_counts_j2_up_and_j3_down(self):
        state = np.array([0.0, 0.7, -0.4, 0.0, 0.0, 0.0, 10.0], dtype=np.float64)
        images = {"top": np.zeros((4, 5, 3), dtype=np.uint8)}
        expected = ["observation.state", "observation.images.top"]
        predictions = [
            np.array([0.0, 0.7, -0.4, 0.0, 0.0, 0.0, 10.0]),
            np.array([0.0, 45.0, -30.0, 0.0, 0.0, 0.0, 10.0]),
            np.array([0.0, 44.0, -29.0, 0.0, 0.0, 0.0, 10.0]),
        ]
        policy = FakePolicy(predictions)
        metrics = preflight.run_dry_run(
            state, images, expected, policy.predict, steps=3
        )
        self.assertAlmostEqual(metrics["j2_max_deg"], 45.0)
        self.assertEqual(metrics["j2_max_step"], 1)
        self.assertAlmostEqual(metrics["j2_reach_deg"], 44.3)
        self.assertAlmostEqual(metrics["j3_min_deg"], -30.0)
        self.assertEqual(metrics["j3_min_step"], 1)
        self.assertAlmostEqual(metrics["j3_reach_deg"], 29.6)
        self.assertTrue(metrics["trajectory_motion_detected"])

    def test_reverse_drift_is_not_counted_as_reach(self):
        state = np.array([0.0, 0.7, -0.4, 0.0, 0.0, 0.0, 10.0], dtype=np.float64)
        images = {"top": np.zeros((4, 5, 3), dtype=np.uint8)}
        expected = ["observation.state", "observation.images.top"]
        predictions = [
            np.array([0.0, 0.7, -0.4, 0.0, 0.0, 0.0, 10.0]),
            np.array([0.0, 0.2, -0.1, 0.0, 0.0, 0.0, 10.0]),
        ]
        policy = FakePolicy(predictions)
        metrics = preflight.run_dry_run(
            state, images, expected, policy.predict, steps=2
        )
        self.assertAlmostEqual(metrics["j2_reach_deg"], 0.0)
        self.assertAlmostEqual(metrics["j3_reach_deg"], 0.0)
        self.assertFalse(metrics["trajectory_motion_detected"])

    def test_wrong_prediction_dimension_fails(self):
        state, images, expected = self.make_inputs()
        with self.assertRaises(ValueError):
            preflight.run_dry_run(state, images, expected, lambda obs: np.zeros(3), steps=1)

    def test_missing_camera_fails(self):
        state, images, _ = self.make_inputs()
        with self.assertRaises(ValueError):
            preflight.run_dry_run(
                state,
                images,
                ["observation.state", "observation.images.wrist_depth"],
                lambda obs: state,
                steps=1,
            )

    def test_trace_and_report_are_written(self):
        state, images, expected = self.make_inputs()
        metrics = preflight.run_dry_run(
            state,
            images,
            expected,
            lambda obs: state,
            steps=2,
            clock=SequenceClock([0.0, 0.005, 0.005, 0.015]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "dryrun_trace.csv"
            preflight.write_dry_run_trace(trace_path, metrics["trace"])
            with trace_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("latency_ms", rows[0])
            self.assertIn("pred_J2", rows[0])

            report_path = Path(tmp) / "preflight_report.json"
            preflight.write_preflight_report(
                report_path, {"tool": "piper_preflight_dryrun", "dry_run": metrics}
            )
            report = json.loads(report_path.read_text())
            self.assertEqual(report["dry_run"]["steps"], 2)
            self.assertEqual(report["dry_run"]["j2_max_deg"], 1.0)


class TestOutputSafety(unittest.TestCase):
    def test_refuses_output_inside_protected_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            with self.assertRaises(ValueError):
                preflight.ensure_safe_output_dir(dataset / "out", [dataset])
            output = preflight.ensure_safe_output_dir(root / "audit_out", [dataset])
            self.assertTrue(output.is_dir())


if __name__ == "__main__":
    unittest.main(verbosity=2)
