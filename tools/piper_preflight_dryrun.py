#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PiPER read-only preflight, camera pinning check, and zero-motion ACT dry-run.

P1A is strictly TX-free:

* CAN access is receive-only: the socket is opened, filtered to feedback IDs,
  and read with ``recv``; there is no transmit API call anywhere in this file.
* No motor enable, mode, joint, gripper, or master/slave configuration is sent.
* Dataset, checkpoint, and camera YAML files are opened read-only.
* Writes are limited to ``--output-dir``.

What it does:

1. Validate a two-camera YAML (``top``/``wrist``), warn when a camera is pinned
   to a volatile ``/dev/videoN`` path, and print ``/dev/v4l/by-id`` mappings
   with ``--list-v4l-by-id`` so the operator can pin stable by-id paths.
2. Capture one live RGB frame per camera and build ``live_vs_train_f0.png``:
   training top|wrist on the first row, live top|wrist on the second, with
   per-channel mean/std labels.
3. Listen to PiPER feedback for ``--can-duration`` seconds (default 2 s),
   decode J1..J6 (millidegrees), gripper (micrometres), and the 0x2A1 status
   fault bits, then compare against the dataset's per-episode first-frame
   median and report J2/J3 offsets relative to nominal zero.
4. With ``--policy-path``, run a static-observation ACT dry-run for
   ``--steps`` steps and report latency P50/P95/max, achievable FPS, first-step
   jump, J2 upward reach, and J3 downward reach of the predicted trajectory.

Examples::

    python3 tools/piper_preflight_dryrun.py --list-v4l-by-id

Use the LeRobot environment for the real run (pyarrow + LeRobot are needed):

    /home/jetson/miniconda3/envs/lerobot/bin/python tools/piper_preflight_dryrun.py \
      --camera-config /home/jetson/lerobot/src/lerobot/scripts/piper_cameras_act_2cam.yaml \
      --dataset-root /home/jetson/lerobot/outputs/piper_pika_test_1013_2cam \
      --policy-path /home/jetson/lerobot/outputs/train/piper_pika_test_1013_2cam/checkpoints/last/pretrained_model \
      --can-interface can0 --steps 60 --device cpu \
      --output-dir /tmp/piper_preflight_out
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:  # direct script execution / unittest discovery
    import piper_can_codec as codec
except ImportError:  # package-style import from the repository root
    from tools import piper_can_codec as codec

EXPECTED_CAMERAS = ("top", "wrist")
VOLATILE_VIDEO_RE = re.compile(r"^/dev/video(\d+)$")
DEFAULT_OUTPUT_DIR = Path("/tmp/piper_preflight_out")
DEFAULT_CAN_DURATION_S = 2.0
DEFAULT_STEPS = 60
NEAR_ZERO_EPS_DEG = 0.55


# ---------------------------------------------------------------------------
# Camera configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraSpec:
    name: str
    camera_type: str
    index_or_path: str | None
    width: int
    height: int
    fps: float
    fourcc: str | None
    backend: str | None
    reader: str | None
    warmup_s: float
    raw: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.camera_type,
            "index_or_path": self.index_or_path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "fourcc": self.fourcc,
            "backend": self.backend,
            "reader": self.reader,
            "warmup_s": self.warmup_s,
        }


def load_camera_config(path: Path) -> dict[str, CameraSpec]:
    """Load a LeRobot-style camera YAML (top/wrist/...)."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is present in the ops env
        raise RuntimeError("PyYAML is required to read the camera YAML") from exc

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"camera config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or not data:
        raise ValueError(f"camera config {config_path} must be a non-empty mapping")

    specs: dict[str, CameraSpec] = {}
    for name, raw in data.items():
        if not isinstance(raw, Mapping):
            continue
        value = raw.get("index_or_path", raw.get("index", raw.get("device")))
        specs[str(name)] = CameraSpec(
            name=str(name),
            camera_type=str(raw.get("type", "opencv")).lower(),
            index_or_path=None if value is None else str(value),
            width=int(raw.get("width", 0) or 0),
            height=int(raw.get("height", 0) or 0),
            fps=float(raw.get("fps", 0.0) or 0.0),
            fourcc=None if raw.get("fourcc") is None else str(raw.get("fourcc")),
            backend=None if raw.get("backend") is None else str(raw.get("backend")),
            reader=None if raw.get("reader") is None else str(raw.get("reader")),
            warmup_s=float(raw.get("warmup_s", 0.0) or 0.0),
            raw=dict(raw),
        )
    if not specs:
        raise ValueError(f"camera config {config_path} contains no camera entries")
    return specs


def is_volatile_video_path(value: Any) -> bool:
    """True for a literal /dev/videoN index, which can move across reconnects."""
    if not isinstance(value, str):
        return False
    return VOLATILE_VIDEO_RE.match(value.strip()) is not None


def validate_camera_config(
    specs: Mapping[str, CameraSpec], expected: Sequence[str] = EXPECTED_CAMERAS
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    volatile: list[dict[str, str]] = []

    for name in expected:
        if name not in specs:
            errors.append(f"required camera {name!r} is missing from the YAML")
    for name in sorted(set(specs) - set(expected)):
        warnings.append(f"extra camera {name!r} is present; P1A compares only {list(expected)}")

    seen_paths: dict[str, str] = {}
    for name, spec in specs.items():
        if spec.camera_type in ("opencv", "v4l2"):
            if not spec.index_or_path:
                errors.append(f"camera {name!r} has no index_or_path")
            elif is_volatile_video_path(spec.index_or_path):
                volatile.append({"camera": name, "path": spec.index_or_path})
                warnings.append(
                    f"camera {name!r} uses volatile {spec.index_or_path}; "
                    "pin /dev/v4l/by-id/... instead (run --list-v4l-by-id)"
                )
            if (spec.width, spec.height) != (640, 480):
                warnings.append(
                    f"camera {name!r} config is {spec.width}x{spec.height}, expected 640x480"
                )
            if spec.index_or_path:
                other = seen_paths.get(spec.index_or_path)
                if other is not None:
                    errors.append(
                        f"cameras {other!r} and {name!r} share the same device {spec.index_or_path}"
                    )
                else:
                    seen_paths[spec.index_or_path] = name
        elif spec.camera_type != "realsense":
            warnings.append(f"camera {name!r} has unknown type {spec.camera_type!r}")

    return {
        "warnings": warnings,
        "errors": errors,
        "volatile_paths": volatile,
        "expected_cameras": list(expected),
        "present_cameras": sorted(specs),
    }


def enumerate_v4l_by_id(base: Path | str = "/dev/v4l/by-id") -> list[dict[str, str]]:
    """List stable /dev/v4l/by-id symlinks and their video device targets."""
    directory = Path(base)
    if not directory.is_dir():
        return []
    rows: list[dict[str, str]] = []
    for entry in sorted(directory.iterdir()):
        if entry.name.startswith("."):
            continue
        try:
            target = os.readlink(entry)
        except OSError:
            target = ""
        resolved = entry.resolve()
        video = resolved.name if re.fullmatch(r"video\d+", resolved.name) else ""
        if not video and target:
            candidate = Path(target).name
            video = candidate if re.fullmatch(r"video\d+", candidate) else ""
        rows.append(
            {
                "by_id": str(entry),
                "name": entry.name,
                "target": target,
                "resolved": str(resolved),
                "video_device": f"/dev/{video}" if video else "",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Live camera capture
# ---------------------------------------------------------------------------


class CameraFrameError(ValueError):
    pass


def bgr_to_rgb(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise CameraFrameError(f"expected HxWx3/4 BGR frame, got {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    rgb = np.ascontiguousarray(arr[:, :, ::-1])
    return rgb


def validate_rgb_frame(image: Any, width: int, height: int, name: str = "camera") -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise CameraFrameError(f"{name}: expected uint8 RGB, got {arr.dtype}")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise CameraFrameError(f"{name}: expected HxWx3 RGB, got {arr.shape}")
    if arr.shape[1] != int(width) or arr.shape[0] != int(height):
        raise CameraFrameError(
            f"{name}: expected {width}x{height}, got {arr.shape[1]}x{arr.shape[0]}"
        )
    return arr


class OpenCVCamera:
    """Thin OpenCV V4L2 camera wrapper; instantiated only for a real capture."""

    def __init__(self, spec: CameraSpec):
        self.spec = spec
        self._capture = None

    def _source(self) -> Any:
        value = self.spec.index_or_path or ""
        return int(value) if value.isdigit() else value

    def open(self) -> None:
        import cv2

        source = self._source()
        if (self.spec.backend or "").lower() == "v4l2":
            capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        else:
            capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise CameraFrameError(f"camera {self.spec.name!r} could not open {source!r}")
        if self.spec.fourcc:
            fourcc = (self.spec.fourcc + "    ")[:4]
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if self.spec.width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec.width)
        if self.spec.height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.height)
        if self.spec.fps:
            capture.set(cv2.CAP_PROP_FPS, self.spec.fps)
        self._capture = capture
        warmup = int(max(0.0, self.spec.warmup_s) * max(1.0, self.spec.fps))
        for _ in range(min(warmup, 180)):
            capture.read()

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise RuntimeError("camera is not open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraFrameError(f"camera {self.spec.name!r} returned no frame")
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def capture_live_images(
    specs: Mapping[str, CameraSpec],
    camera_factory: Callable[[CameraSpec], Any] = OpenCVCamera,
    expected: Sequence[str] = EXPECTED_CAMERAS,
    expected_size: tuple[int, int] = (640, 480),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Capture one RGB frame per required camera and return RGB + stats.

    ``expected_size`` is enforced after BGR->RGB conversion; ACT training used
    640x480, so a different live resolution is a hard preflight failure.
    """
    images: dict[str, np.ndarray] = {}
    stats: dict[str, Any] = {}
    width, height = int(expected_size[0]), int(expected_size[1])
    for name in expected:
        if name not in specs:
            raise CameraFrameError(f"camera {name!r} is not configured")
        spec = specs[name]
        camera = camera_factory(spec)
        camera.open()
        try:
            bgr = camera.read()
        finally:
            camera.release()
        rgb = bgr_to_rgb(bgr)
        rgb = validate_rgb_frame(rgb, width, height, name)
        images[name] = rgb
        stats[name] = frame_stats(rgb)
    return images, stats


def frame_stats(image: Any) -> dict[str, Any]:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got {arr.shape}")
    mean = arr.reshape(-1, 3).mean(axis=0)
    variance = arr.reshape(-1, 3).var(axis=0)
    std = np.sqrt(variance)
    return {
        "shape": [int(arr.shape[0]), int(arr.shape[1]), 3],
        "mean_rgb": [float(value) for value in mean],
        "variance_rgb": [float(value) for value in variance],
        "std_rgb": [float(value) for value in std],
    }


# ---------------------------------------------------------------------------
# Training-frame comparison image
# ---------------------------------------------------------------------------


def image_to_rgb_uint8(image: Any, name: str = "image") -> np.ndarray:
    """Convert a LeRobot dataset image (CHW float [0,1] or HWC uint8) to HWC uint8."""
    arr = (
        np.asarray(image.detach().cpu().numpy())
        if hasattr(image, "detach")
        else np.asarray(image)
    )
    if arr.ndim != 3:
        raise ValueError(f"{name}: expected 3 dims, got {arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"{name}: expected 3 RGB channels, got {arr.shape}")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    arr = arr.astype(np.float32, copy=False)
    low, high = float(arr.min()), float(arr.max())
    if -0.01 <= low and high <= 1.01:
        out = np.rint(np.clip(arr, 0.0, 1.0) * 255.0)
    elif -0.01 <= low and high <= 255.01:
        out = np.rint(np.clip(arr, 0.0, 255.0))
    else:
        raise ValueError(f"{name}: unexpected range [{low}, {high}]")
    return np.ascontiguousarray(out.astype(np.uint8))


def make_comparison_grid(
    train_images: Mapping[str, np.ndarray],
    live_images: Mapping[str, np.ndarray],
    output_path: Path,
    camera_keys: Sequence[str] = EXPECTED_CAMERAS,
    train_label: str = "TRAIN ep0 f0",
    live_label: str = "LIVE",
    padding: int = 6,
    label_height: int = 48,
) -> dict[str, Any]:
    """Save training(top|wrist) over live(top|wrist) with per-channel stats."""
    from PIL import Image, ImageDraw

    for key in camera_keys:
        if key not in train_images or key not in live_images:
            raise ValueError(f"comparison image is missing {key!r}")

    cell_h, cell_w = np.asarray(train_images[camera_keys[0]]).shape[:2]
    for key in camera_keys:
        for source in (train_images, live_images):
            arr = np.asarray(source[key])
            if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
                raise ValueError(f"{key}: expected HxWx3 uint8, got {arr.shape}/{arr.dtype}")
            if arr.shape[:2] != (cell_h, cell_w):
                raise ValueError(f"{key}: comparison images must share one resolution")

    rows = 2
    cols = len(camera_keys)
    width = cols * cell_w + (cols + 1) * padding
    height = rows * (cell_h + label_height) + (rows + 1) * padding
    canvas = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)

    stats: dict[str, Any] = {"train": {}, "live": {}}
    for row, (source, row_label) in enumerate(
        ((train_images, train_label), (live_images, live_label))
    ):
        for col, key in enumerate(camera_keys):
            image = source[key]
            stat = frame_stats(image)
            stats["train" if row == 0 else "live"][key] = stat
            x = padding + col * (cell_w + padding)
            y = padding + row * (cell_h + label_height + padding)
            mean = stat["mean_rgb"]
            std = stat["std_rgb"]
            var = stat["variance_rgb"]
            draw.text((x + 2, y + 2), f"{row_label} | {key}", fill=(235, 235, 235))
            draw.text(
                (x + 2, y + 18),
                f"mean=({mean[0]:.1f},{mean[1]:.1f},{mean[2]:.1f}) "
                f"std=({std[0]:.1f},{std[1]:.1f},{std[2]:.1f}) "
                f"var=({var[0]:.1f},{var[1]:.1f},{var[2]:.1f})",
                fill=(190, 190, 190),
            )
            canvas.paste(Image.fromarray(image, mode="RGB"), (x, y + label_height))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {
        "path": str(output_path),
        "width": int(canvas.width),
        "height": int(canvas.height),
        "camera_keys": list(camera_keys),
        "train_label": train_label,
        "live_label": live_label,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Dataset loading: first-frame medians and training frame
# ---------------------------------------------------------------------------


def _dataset_state_names(info: Mapping[str, Any]) -> list[str]:
    names = (((info.get("features") or {}).get("observation.state") or {}).get("names"))
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    return list(codec.STATE_NAMES)


def first_frame_medians(
    state: Any,
    episode_index: Any,
    frame_index: Any,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Median of each episode's first recorded state, per dimension."""
    values = np.asarray(state, dtype=np.float64)
    episodes = np.asarray(episode_index).astype(np.int64).reshape(-1)
    frames = np.asarray(frame_index).astype(np.int64).reshape(-1)
    if values.ndim != 2:
        raise ValueError(f"state must be 2-D, got {values.shape}")
    if not (values.shape[0] == episodes.size == frames.size) or values.shape[0] == 0:
        raise ValueError("state/episode/frame row counts disagree or are empty")
    if not np.isfinite(values).all():
        raise ValueError("state contains NaN/Inf")
    labels = list(names) if names is not None else list(codec.STATE_NAMES)
    if len(labels) != values.shape[1]:
        labels = [f"dim_{index}" for index in range(values.shape[1])]

    first_rows: list[int] = []
    first_frames: dict[str, int] = {}
    for episode in sorted({int(value) for value in episodes}):
        indices = np.flatnonzero(episodes == episode)
        first_local = indices[int(np.argmin(frames[indices]))]
        first_rows.append(int(first_local))
        first_frames[str(episode)] = int(frames[first_local])

    first_matrix = values[first_rows, :]
    median = np.median(first_matrix, axis=0)
    std = first_matrix.std(axis=0)
    return {
        "episodes": len(first_rows),
        "episode_indices": sorted(int(value) for value in np.unique(episodes)),
        "first_frame_indices": first_frames,
        "median": [float(value) for value in median],
        "median_by_name": {label: float(median[index]) for index, label in enumerate(labels)},
        "std": [float(value) for value in std],
        "first_values": first_matrix.tolist(),
        "names": labels,
    }


def load_dataset_state_medians(root: Path) -> dict[str, Any]:
    """Read state/episode/frame parquet columns and compute first-frame medians."""
    try:
        import pyarrow.dataset as pa_dataset
    except ImportError as exc:  # pragma: no cover - ops env has pyarrow
        raise RuntimeError(
            "pyarrow is required; run with the LeRobot env, e.g. "
            "/home/jetson/miniconda3/envs/lerobot/bin/python"
        ) from exc

    root = Path(root).expanduser()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"dataset metadata missing: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    data_dir = root / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"dataset data directory missing: {data_dir}")
    columns = ["observation.state", "episode_index", "frame_index"]
    table = pa_dataset.dataset(str(data_dir), format="parquet").to_table(columns=columns)
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
    episodes = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64).reshape(-1)
    frames = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64).reshape(-1)
    medians = first_frame_medians(state, episodes, frames, _dataset_state_names(info))
    medians["info_total_episodes"] = info.get("total_episodes")
    medians["info_total_frames"] = info.get("total_frames")
    medians["actual_total_frames"] = int(state.shape[0])
    medians["episodes_match_info"] = info.get("total_episodes") in (None, medians["episodes"])
    medians["fps"] = info.get("fps")
    medians["robot_type"] = info.get("robot_type")
    return medians


def load_train_first_frame(
    root: Path,
    camera_keys: Sequence[str] = EXPECTED_CAMERAS,
    repo_id: str | None = None,
    video_backend: str = "pyav",
) -> dict[str, Any]:
    """Load episode 0 / frame 0 images from a local LeRobot dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root).expanduser()
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"dataset metadata missing: {root / 'meta' / 'info.json'}")
    dataset = LeRobotDataset(
        repo_id or f"local/{root.name}",
        root=root,
        episodes=[0],
        video_backend=video_backend,
        download_videos=False,
    )
    if len(dataset) == 0:
        raise RuntimeError("episode 0 returned zero frames")
    sample = dataset[0]
    images: dict[str, np.ndarray] = {}
    for key in camera_keys:
        feature = f"observation.images.{key}"
        if feature not in sample:
            raise ValueError(f"dataset sample is missing {feature!r}")
        images[key] = image_to_rgb_uint8(sample[feature], feature)
    state = np.asarray(
        sample["observation.state"].detach().cpu().numpy()
        if hasattr(sample["observation.state"], "detach")
        else sample["observation.state"],
        dtype=np.float64,
    ).reshape(-1)
    episode = sample.get("episode_index", 0)
    frame = sample.get("frame_index", 0)
    return {
        "episode_index": int(np.asarray(episode).reshape(-1)[0]),
        "frame_index": int(np.asarray(frame).reshape(-1)[0]),
        "state": [float(value) for value in state],
        "images": images,
        "image_stats": {key: frame_stats(image) for key, image in images.items()},
    }


# ---------------------------------------------------------------------------
# Read-only CAN feedback
# ---------------------------------------------------------------------------


class CanSocketReader:
    """Receive-only AF_CAN reader. This class intentionally has no TX method."""

    def __init__(
        self,
        interface: str,
        socket_factory: Callable[[], Any] | None = None,
        recv_timeout_s: float = 0.2,
    ):
        self.interface = str(interface)
        self._socket_factory = socket_factory
        self._recv_timeout_s = float(recv_timeout_s)
        self._socket_module: Any = None
        self._socket: Any = None

    def open(self) -> None:
        import socket as socket_module

        self._socket_module = socket_module
        sock = (
            self._socket_factory()
            if self._socket_factory is not None
            else socket_module.socket(
                socket_module.AF_CAN, socket_module.SOCK_RAW, socket_module.CAN_RAW
            )
        )
        sock.bind((self.interface,))
        try:
            filters = b"".join(
                struct.pack("=II", feedback_id, codec.CAN_SFF_MASK)
                for feedback_id in codec.FEEDBACK_IDS
            )
            sock.setsockopt(
                socket_module.SOL_CAN_RAW, socket_module.CAN_RAW_FILTER, filters
            )
        except (AttributeError, OSError):
            # Filtering is an optimisation; recv-only semantics do not depend on it.
            pass
        sock.settimeout(self._recv_timeout_s)
        self._socket = sock

    def read_until(
        self, duration_s: float, clock: Callable[[], float] = time.monotonic
    ) -> list[codec.CanFrame]:
        if self._socket is None:
            raise RuntimeError("CAN reader is not open")
        duration = float(duration_s)
        if duration <= 0:
            raise ValueError("CAN listen duration must be > 0")
        start = float(clock())
        deadline = start + duration
        frames: list[codec.CanFrame] = []
        while True:
            now = float(clock())
            if now >= deadline:
                break
            try:
                raw = self._socket.recv(16)
            except self._socket_module.timeout:
                continue
            try:
                arbitration_id, data = codec.parse_can_frame(raw)
            except ValueError:
                continue
            frames.append(codec.CanFrame(timestamp_s=now, arbitration_id=arbitration_id, data=data))
        return frames

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

    def __enter__(self) -> "CanSocketReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def summarize_can_frames(
    frames: Sequence[codec.CanFrame],
    duration_s: float,
    feedback: codec.PiperFeedback | None = None,
) -> dict[str, Any]:
    decoder = feedback if feedback is not None else codec.decode_feedback_frames(frames)
    counts: dict[str, int] = {}
    for frame in frames:
        key = f"0x{frame.arbitration_id:03X}"
        counts[key] = counts.get(key, 0) + 1
    span = 0.0
    if len(frames) >= 2:
        span = float(frames[-1].timestamp_s - frames[0].timestamp_s)
    missing = [
        f"0x{feedback_id:03X}" for feedback_id in codec.FEEDBACK_IDS if feedback_id not in decoder.counts
    ]
    return {
        "mode": "READ_ONLY_RX_ONLY",
        "duration_s": float(duration_s),
        "frames": len(frames),
        "frames_per_s": float(len(frames) / duration_s) if duration_s > 0 else 0.0,
        "observed_span_s": span,
        "observed_fps": float((len(frames) - 1) / span) if span > 0 else 0.0,
        "counts": counts,
        "missing_feedback_ids": missing,
        "malformed_frames": int(decoder.malformed),
        "complete": bool(decoder.complete),
        "joints_deg": [None if value is None else float(value) for value in decoder.joints_deg],
        "gripper_mm": None if decoder.gripper_mm is None else float(decoder.gripper_mm),
        "state_vector": decoder.state_vector(),
        "status": decoder.status.as_dict() if decoder.status is not None else None,
        "nominal_limits": codec.nominal_limit_report(decoder.joints_deg),
    }


def collect_can_feedback(
    reader: CanSocketReader,
    duration_s: float,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    with reader:
        frames = reader.read_until(duration_s, clock=clock)
    return summarize_can_frames(frames, duration_s)


# ---------------------------------------------------------------------------
# Deviation and zero-offset diagnostics
# ---------------------------------------------------------------------------


def compute_joint_deviation(
    current_deg: Sequence[float | None],
    median_deg: Sequence[float],
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    if len(current_deg) != len(median_deg):
        raise ValueError("current and median vectors must have the same length")
    labels = list(names) if names is not None else list(codec.STATE_NAMES)
    if len(labels) != len(median_deg):
        labels = [f"dim_{index}" for index in range(len(median_deg))]
    rows: dict[str, Any] = {}
    for index, label in enumerate(labels):
        current = current_deg[index]
        median = float(median_deg[index])
        rows[label] = {
            "current": None if current is None else float(current),
            "median": median,
            "delta": None if current is None else float(current - median),
        }
    return rows


def zero_offset_report(joints_deg: Sequence[float | None]) -> dict[str, Any]:
    """J2/J3 offset from nominal zero plus limit margins."""
    if len(joints_deg) < 6:
        raise ValueError("need at least 6 joints")
    result: dict[str, Any] = {}
    for index, name in ((1, "J2"), (2, "J3")):
        value = joints_deg[index]
        low, high = codec.JOINT_LIMITS_DEG[index]
        entry: dict[str, Any] = {
            "current_deg": None if value is None else float(value),
            "nominal_zero_deg": 0.0,
            "nominal_range_deg": [low, high],
            "offset_from_zero_deg": None,
            "inside_nominal_range": None,
            "limit_margin_deg": None,
            "near_zero_pinned": None,
        }
        if value is not None:
            entry.update(
                {
                    "offset_from_zero_deg": float(value),
                    "inside_nominal_range": bool(low <= value <= high),
                    "limit_margin_deg": float(min(value - low, high - value)),
                    "near_zero_pinned": bool(
                        value <= NEAR_ZERO_EPS_DEG if name == "J2" else value >= -NEAR_ZERO_EPS_DEG
                    ),
                }
            )
        result[name] = entry
    return result


# ---------------------------------------------------------------------------
# Zero-motion ACT dry-run
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size == 0:
        return 0.0
    if ordered.size == 1:
        return float(ordered[0])
    rank = (ordered.size - 1) * (float(percentile) / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    fraction = rank - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def build_dry_run_observation(
    state_vector: Sequence[float],
    images: Mapping[str, np.ndarray],
    expected_inputs: Sequence[str],
) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {}
    for key in expected_inputs:
        if key == "observation.state":
            observation[key] = np.asarray(state_vector, dtype=np.float32).reshape(-1)
            continue
        if key.startswith("observation.images."):
            camera = key.rsplit(".", 1)[-1]
            if camera not in images:
                raise ValueError(f"checkpoint needs camera {camera!r}, but live images only have {sorted(images)}")
            image = np.asarray(images[camera])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"{key}: expected HxWx3 uint8, got {image.shape}/{image.dtype}")
            observation[key] = image
            continue
        raise ValueError(f"unsupported checkpoint input {key!r} for live dry-run")
    if "observation.state" not in observation:
        raise ValueError("checkpoint input_features do not contain observation.state")
    return observation


def run_dry_run(
    state_vector: Sequence[float],
    images: Mapping[str, np.ndarray],
    expected_inputs: Sequence[str],
    predict_fn: Callable[[dict[str, np.ndarray]], Any],
    steps: int = DEFAULT_STEPS,
    reset_fn: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    names: Sequence[str] = codec.STATE_NAMES,
) -> dict[str, Any]:
    """Run N static-observation predictions and summarize latency/trajectory."""
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be >= 1")
    state = np.asarray(state_vector, dtype=np.float64).reshape(-1)
    if not np.isfinite(state).all():
        raise ValueError("state_vector contains NaN/Inf")
    labels = list(names)
    if len(labels) != state.size:
        labels = [f"dim_{index}" for index in range(state.size)]
    observation = build_dry_run_observation(state, images, expected_inputs)

    if reset_fn is not None:
        reset_fn()

    trace: list[dict[str, Any]] = []
    latencies: list[float] = []
    predictions: list[np.ndarray] = []
    for step in range(steps):
        start = float(clock())
        prediction = np.asarray(predict_fn(dict(observation)), dtype=np.float64).reshape(-1)
        end = float(clock())
        if prediction.size != state.size:
            raise ValueError(
                f"prediction size {prediction.size} does not match state size {state.size}"
            )
        if not np.isfinite(prediction).all():
            raise ValueError(f"non-finite prediction at step {step}")
        latency_ms = max(0.0, (end - start) * 1000.0)
        latencies.append(latency_ms)
        predictions.append(prediction.copy())
        trace.append(
            {
                "step": step,
                "latency_ms": latency_ms,
                "state": state.tolist(),
                "pred": prediction.tolist(),
            }
        )

    trajectory = np.stack(predictions)
    latency_array = np.asarray(latencies, dtype=np.float64)
    mean_latency = float(latency_array.mean())
    p50 = _percentile(latency_array, 50.0)
    p95 = _percentile(latency_array, 95.0)
    first_jump = np.abs(trajectory[0] - state)
    extrema: dict[str, Any] = {}
    for index, label in enumerate(labels):
        min_index = int(np.argmin(trajectory[:, index]))
        max_index = int(np.argmax(trajectory[:, index]))
        extrema[label] = {
            "min": float(trajectory[min_index, index]),
            "min_step": min_index,
            "max": float(trajectory[max_index, index]),
            "max_step": max_index,
        }

    j2_index = 1 if state.size > 1 else 0
    j3_index = 2 if state.size > 2 else j2_index
    j2_values = trajectory[:, j2_index]
    j3_values = trajectory[:, j3_index]
    # Piper6 geometry: J2 belongs to [0, 180] and reaches downward by increasing;
    # J3 belongs to [-170, 0] and reaches downward by decreasing.
    j2_max_step = int(np.argmax(j2_values))
    j3_min_step = int(np.argmin(j3_values))
    j2_reach = float(max(0.0, j2_values[j2_max_step] - state[j2_index]))
    j3_reach = float(max(0.0, state[j3_index] - j3_values[j3_min_step]))
    return {
        "steps": steps,
        "latency_ms": {
            "p50": p50,
            "p95": p95,
            "max": float(latency_array.max()),
            "mean": mean_latency,
        },
        "fps": {
            "closed_loop_mean": 1000.0 / mean_latency if mean_latency > 0 else 0.0,
            "closed_loop_p95": 1000.0 / p95 if p95 > 0 else 0.0,
        },
        "first_frame_jump": {
            label: float(first_jump[index]) for index, label in enumerate(labels)
        },
        "first_prediction": {label: float(trajectory[0, index]) for index, label in enumerate(labels)},
        "initial_state": {label: float(state[index]) for index, label in enumerate(labels)},
        "trajectory_extrema": extrema,
        "j2_max_deg": float(j2_values[j2_max_step]),
        "j2_max_step": j2_max_step,
        "j2_delta_from_state_deg": float(j2_values[j2_max_step] - state[j2_index]),
        "j2_reach_deg": j2_reach,
        "j3_min_deg": float(j3_values[j3_min_step]),
        "j3_min_step": j3_min_step,
        "j3_delta_from_state_deg": float(j3_values[j3_min_step] - state[j3_index]),
        "j3_reach_deg": j3_reach,
        "trajectory_motion_detected": bool(max(j2_reach, j3_reach) >= 0.1),
        "trace": trace,
    }


def write_dry_run_trace(
    path: Path, trace: Sequence[Mapping[str, Any]], names: Sequence[str] = codec.STATE_NAMES
) -> None:
    if not trace:
        raise ValueError("empty dry-run trace")
    labels = list(names)
    state_width = len(trace[0]["state"])
    if len(labels) != state_width:
        labels = [f"dim_{index}" for index in range(state_width)]
    fieldnames = ["step", "latency_ms"]
    fieldnames.extend(f"state_{label}" for label in labels)
    fieldnames.extend(f"pred_{label}" for label in labels)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in trace:
            flat: dict[str, Any] = {"step": row["step"], "latency_ms": row["latency_ms"]}
            for index, label in enumerate(labels):
                flat[f"state_{label}"] = float(row["state"][index])
                flat[f"pred_{label}"] = float(row["pred"][index])
            writer.writerow(flat)


# ---------------------------------------------------------------------------
# ACT loading (same LeRobot v0.4.4 chain as deployment)
# ---------------------------------------------------------------------------


def load_act_policy(
    policy_dir: Path,
    device: str,
    n_action_steps: int | None = None,
    temporal_ensemble_coeff: float | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.utils import get_safe_torch_device
    import torch

    path = str(Path(policy_dir).expanduser().resolve())
    config = PreTrainedConfig.from_pretrained(path)
    if config.type != "act":
        raise ValueError(f"expected ACT checkpoint, got type={config.type!r}")
    config.device = device
    config.pretrained_path = path
    if n_action_steps is not None:
        if int(n_action_steps) < 1:
            raise ValueError("--n-action-steps must be >= 1")
        config.n_action_steps = int(n_action_steps)
    if temporal_ensemble_coeff is not None:
        if float(temporal_ensemble_coeff) <= 0:
            raise ValueError("--temporal-ensemble-coeff must be > 0")
        config.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
        if n_action_steps is None:
            config.n_action_steps = 1
    queue_steps = int(config.n_action_steps)
    chunk_size = int(getattr(config, "chunk_size", queue_steps))
    if queue_steps < 1 or queue_steps > chunk_size:
        raise ValueError(f"n_action_steps={queue_steps} must be within 1..chunk_size({chunk_size})")
    if getattr(config, "temporal_ensemble_coeff", None) is not None and queue_steps != 1:
        raise ValueError("ACT temporal ensemble requires n_action_steps=1")

    torch_device = get_safe_torch_device(config.device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    policy = ACTPolicy.from_pretrained(path, config=config)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": config.device}},
    )
    return config, policy, preprocessor, postprocessor, torch_device


def run_act_dry_run(
    policy_dir: Path,
    state_vector: Sequence[float],
    images: Mapping[str, np.ndarray],
    steps: int,
    device: str,
    n_action_steps: int | None,
    temporal_ensemble_coeff: float | None,
    task: str = "",
    robot_type: str = "piper_pika",
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    from lerobot.utils.control_utils import predict_action

    config, policy, preprocessor, postprocessor, torch_device = load_act_policy(
        policy_dir, device, n_action_steps, temporal_ensemble_coeff
    )
    expected_inputs = list((config.input_features or {}).keys())
    state_feature = (config.input_features or {}).get("observation.state") or {}
    expected_state_dim = state_feature.get("shape")
    if isinstance(expected_state_dim, (list, tuple)) and len(expected_state_dim) == 1:
        if int(expected_state_dim[0]) != len(state_vector):
            raise ValueError(
                f"checkpoint state dim {expected_state_dim[0]} != live state dim {len(state_vector)}"
            )

    def predict(observation: dict[str, np.ndarray]) -> Any:
        return predict_action(
            observation=observation,
            policy=policy,
            device=torch_device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=bool(getattr(config, "use_amp", False)),
            task=task,
            robot_type=robot_type,
        )

    def reset() -> None:
        for item in (policy, preprocessor, postprocessor):
            if hasattr(item, "reset"):
                item.reset()

    metrics = run_dry_run(
        state_vector=state_vector,
        images=images,
        expected_inputs=expected_inputs,
        predict_fn=predict,
        steps=steps,
        reset_fn=reset,
        clock=clock,
    )
    metrics["checkpoint"] = {
        "path": str(Path(policy_dir).expanduser().resolve()),
        "chunk_size": int(getattr(config, "chunk_size", 0)),
        "n_action_steps": int(getattr(config, "n_action_steps", 0)),
        "temporal_ensemble_coeff": getattr(config, "temporal_ensemble_coeff", None),
        "input_features": expected_inputs,
    }
    return metrics


# ---------------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------------


def ensure_safe_output_dir(output_dir: Path, protected: Sequence[Path]) -> Path:
    out = Path(output_dir).expanduser().resolve()
    for item in protected:
        directory = Path(item).expanduser().resolve()
        if out == directory or directory in out.parents:
            raise ValueError(f"refusing to write output inside read-only path: {directory}")
    if out.exists() and not out.is_dir():
        raise ValueError(f"--output-dir is not a directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_preflight_report(path: Path, report: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def print_v4l_table(rows: Sequence[Mapping[str, str]]) -> None:
    print("=== /dev/v4l/by-id ===")
    if not rows:
        print("  (no entries; is /dev/v4l/by-id present?)")
        return
    print(f"  {'by-id':<78} {'target':<18} video")
    for row in rows:
        print(f"  {row['name']:<78} {row['target']:<18} {row['video_device'] or '-'}")


def print_preflight_summary(report: Mapping[str, Any]) -> None:
    print("=== 1. 相机配置 ===")
    for spec in report["camera_config"]["cameras"]:
        print(
            f"  {spec['name']}: {spec['type']} path={spec['index_or_path']} "
            f"{spec['width']}x{spec['height']}@{spec['fps']}fps backend={spec['backend']}"
        )
    for warning in report["camera_config"]["warnings"]:
        print(f"  [WARN] {warning}")
    for error in report["camera_config"]["errors"]:
        print(f"  [ERROR] {error}")
    if report["camera_config"]["volatile_paths"]:
        print(f"  [PIN] volatile paths: {report['camera_config']['volatile_paths']}")

    if report.get("live_capture"):
        print("=== 2. 实时画面 ===")
        for name, stat in report["live_capture"].items():
            print(
                f"  {name}: {stat['shape']} mean={stat['mean_rgb']} std={stat['std_rgb']}"
            )
    if report.get("comparison_image"):
        print(f"  comparison: {report['comparison_image']['path']}")

    can = report.get("can")
    if can:
        print("=== 3. CAN 只读反馈（零发送）===")
        print(
            f"  frames={can['frames']} rate={can['frames_per_s']:.2f}/s complete={can['complete']} "
            f"malformed={can['malformed_frames']}"
        )
        print(f"  joints_deg={can['joints_deg']} gripper_mm={can['gripper_mm']}")
        if can.get("status"):
            status = can["status"]
            print(
                f"  status arm={status['arm_status_name']} mode={status['mode_name']} "
                f"limit_bits=0x{status['joint_limit_error_bits']:02X} "
                f"comm_bits=0x{status['joint_comm_error_bits']:02X}"
            )
        if can.get("missing_feedback_ids"):
            print(f"  [WARN] missing feedback IDs: {can['missing_feedback_ids']}")

    if report.get("dataset_median"):
        print("=== 4. 数据集首帧中位数对照 ===")
        median = report["dataset_median"]
        print(f"  episodes={median['episodes']} median={median['median_by_name']}")
        for name, item in (report.get("joint_deviation") or {}).items():
            print(f"  {name}: current={item['current']} median={item['median']} delta={item['delta']}")
        for name, item in (report.get("zero_offset") or {}).items():
            print(
                f"  {name}: offset={item['offset_from_zero_deg']} "
                f"margin={item['limit_margin_deg']} pinned={item['near_zero_pinned']}"
            )

    dry = report.get("dry_run")
    if dry:
        print("=== 5. 零运动 ACT Dry-Run ===")
        latency = dry["latency_ms"]
        print(
            f"  steps={dry['steps']} P50={latency['p50']:.2f}ms P95={latency['p95']:.2f}ms "
            f"max={latency['max']:.2f}ms FPS(mean)={dry['fps']['closed_loop_mean']:.2f} "
            f"FPS(P95)={dry['fps']['closed_loop_p95']:.2f}"
        )
        print(
            f"  J2 max={dry['j2_max_deg']:.3f} (reach +{dry['j2_reach_deg']:.3f}) "
            f"J3 min={dry['j3_min_deg']:.3f} (reach -{dry['j3_reach_deg']:.3f}) "
            f"motion={dry['trajectory_motion_detected']}"
        )
        jump = ", ".join(f"{name}={value:.2f}" for name, value in dry["first_frame_jump"].items())
        print(f"  first-frame jump: {jump}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PiPER read-only preflight, camera pin check, and zero-motion ACT dry-run (NO TX)"
    )
    parser.add_argument("--camera-config", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument("--can-interface", default="can0")
    parser.add_argument("--can-duration", type=float, default=DEFAULT_CAN_DURATION_S)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=None)
    parser.add_argument("--task", default="")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--v4l-base", type=Path, default=Path("/dev/v4l/by-id"))
    parser.add_argument("--list-v4l-by-id", action="store_true")
    parser.add_argument("--skip-camera", action="store_true")
    parser.add_argument("--skip-can", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.can_duration <= 0:
        raise ValueError("--can-duration must be > 0")
    if args.camera_config is None:
        raise ValueError("--camera-config is required unless --list-v4l-by-id is used")

    protected = [Path(args.camera_config).expanduser()]
    if args.dataset_root is not None:
        protected.append(Path(args.dataset_root))
    if args.policy_path is not None:
        protected.append(Path(args.policy_path))
    output_dir = ensure_safe_output_dir(args.output_dir, protected)

    specs = load_camera_config(args.camera_config)
    validation = validate_camera_config(specs)
    v4l_rows = enumerate_v4l_by_id(args.v4l_base)
    report: dict[str, Any] = {
        "tool": "piper_preflight_dryrun",
        "mode": "READ_ONLY_ZERO_TX",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "camera_config": {
            "path": str(Path(args.camera_config).expanduser().resolve()),
            "cameras": [specs[name].as_dict() for name in sorted(specs)],
            **validation,
        },
        "v4l_by_id": v4l_rows,
        "live_capture": None,
        "comparison_image": None,
        "dataset_median": None,
        "joint_deviation": None,
        "zero_offset": None,
        "can": None,
        "dry_run": None,
        "dryrun_trace_csv": None,
        "warnings": [],
        "errors": [],
        "ok": True,
    }
    report["warnings"].extend(validation["warnings"])
    report["errors"].extend(validation["errors"])

    train_frame = None
    dataset_median = None
    if args.dataset_root is not None:
        try:
            dataset_median = load_dataset_state_medians(args.dataset_root)
            report["dataset_median"] = dataset_median
        except Exception as exc:  # keep the rest of the preflight useful
            report["errors"].append(f"dataset median failed: {exc}")
        try:
            train_frame = load_train_first_frame(
                args.dataset_root,
                camera_keys=EXPECTED_CAMERAS,
                repo_id=args.repo_id,
                video_backend=args.video_backend,
            )
            report["train_frame"] = {
                "episode_index": train_frame["episode_index"],
                "frame_index": train_frame["frame_index"],
                "state": train_frame["state"],
                "image_stats": train_frame["image_stats"],
            }
        except Exception as exc:
            report["errors"].append(f"training frame failed: {exc}")

    live_images: dict[str, np.ndarray] = {}
    if args.skip_camera:
        report["warnings"].append("live camera capture skipped by --skip-camera")
    elif validation["errors"]:
        report["errors"].append("live camera capture skipped because camera config has errors")
    else:
        try:
            live_images, live_stats = capture_live_images(specs)
            report["live_capture"] = live_stats
        except Exception as exc:
            report["errors"].append(f"live camera capture failed: {exc}")

    if live_images and train_frame is not None:
        try:
            comparison = make_comparison_grid(
                train_images=train_frame["images"],
                live_images=live_images,
                output_path=output_dir / "live_vs_train_f0.png",
                train_label=f"TRAIN ep{train_frame['episode_index']} f{train_frame['frame_index']}",
                live_label="LIVE",
            )
            report["comparison_image"] = comparison
        except Exception as exc:
            report["errors"].append(f"comparison image failed: {exc}")

    can_summary = None
    if args.skip_can:
        report["warnings"].append("CAN listen skipped by --skip-can")
    else:
        try:
            reader = CanSocketReader(args.can_interface)
            can_summary = collect_can_feedback(reader, args.can_duration)
            report["can"] = can_summary
            if can_summary["frames"] == 0:
                report["errors"].append("CAN listen received zero frames")
            if not can_summary["complete"]:
                report["warnings"].append("CAN feedback is incomplete; some joints/gripper are missing")
        except Exception as exc:
            report["errors"].append(f"CAN listen failed: {exc}")

    if dataset_median is not None and can_summary is not None:
        current = can_summary["state_vector"]
        if current is not None:
            names = dataset_median["names"]
            report["joint_deviation"] = compute_joint_deviation(current, dataset_median["median"], names)
            report["zero_offset"] = zero_offset_report(current)
        else:
            report["warnings"].append("current CAN state is incomplete; deviation not computed")

    if args.policy_path is not None:
        if not live_images:
            report["errors"].append("dry-run requested but no live camera frames are available")
        elif can_summary is None or can_summary["state_vector"] is None:
            report["errors"].append(
                "dry-run requested but complete read-only CAN state is unavailable"
            )
        else:
            try:
                dry_run = run_act_dry_run(
                    policy_dir=args.policy_path,
                    state_vector=can_summary["state_vector"],
                    images=live_images,
                    steps=args.steps,
                    device=args.device,
                    n_action_steps=args.n_action_steps,
                    temporal_ensemble_coeff=args.temporal_ensemble_coeff,
                    task=args.task,
                    robot_type=str((dataset_median or {}).get("robot_type") or "piper_pika"),
                )
                trace_path = output_dir / "dryrun_trace.csv"
                write_dry_run_trace(trace_path, dry_run["trace"])
                dry_run["dryrun_trace_csv"] = str(trace_path)
                report["dry_run"] = dry_run
                report["dryrun_trace_csv"] = str(trace_path)
            except Exception as exc:
                report["errors"].append(f"ACT dry-run failed: {exc}")

    report["ok"] = not report["errors"]
    report_path = output_dir / "preflight_report.json"
    report["report_path"] = str(report_path)
    write_preflight_report(report_path, report)
    print_preflight_summary(report)
    print(f"\nJSON report: {report_path}")
    if report["errors"]:
        print("\nERRORS:")
        for error in report["errors"]:
            print(f"  - {error}")
    return 0 if report["ok"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.list_v4l_by_id:
        print_v4l_table(enumerate_v4l_by_id(args.v4l_base))
        return 0
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
