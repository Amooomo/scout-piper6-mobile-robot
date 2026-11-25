#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline, READ-ONLY audit for a LeRobot ACT dataset + checkpoint.

This tool never imports hardware stacks and never writes to the dataset or
checkpoint directories:

* no CAN / socket / rclpy / pyrealsense2 / cv2.VideoCapture;
* dataset videos are decoded through LeRobotDataset (read-only);
* the only writes go to ``--output-dir`` (JSON report, CSV, PNG grid).

Sections:
1. Dataset integrity: episodes, frames, cameras, state/action extrema,
   first-frame joint values, J2/J3 boundary-pinning and zero-cross ratios.
2. Dataset <-> ACT checkpoint contract: input/output features, camera names,
   HWC/CHW shapes, state/action dims, normalization stats.
3. Open-loop inference: real recorded images + real recorded observations are
   fed through the same LeRobot v0.4.4 ``predict_action`` call chain used by
   deployment; predictions are compared with the recorded actions.
4. Visual samples: start/middle frames from the first audited episode, saved
   as a labelled PNG grid for RGB/view inspection.

Run with the LeRobot v0.4.4 environment, for example:

    /home/jetson/miniconda3/envs/lerobot/bin/python tools/act_offline_audit.py \
      --dataset-root /home/jetson/lerobot/outputs/piper_pika_test_7242 \
      --policy-path /home/jetson/lerobot/outputs/train/piper_pika_test_7242/checkpoints/last/pretrained_model \
      --episodes 0,1,2 --max-frames 450 --device cpu \
      --output-dir /tmp/act_offline_audit_out
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

LOG = logging.getLogger("act_offline_audit")

JOINT_FALLBACK_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6", "gripper")
BOUNDARY_EPS_DEG = 0.55
DEFAULT_EPISODES = (0, 1, 2)
DEFAULT_MAX_FRAMES = 450

# (label, joint index selector, comparison, threshold, extreme kind)
PRED_BOUND_CONDITIONS = (
    ("J2 < 0.0", "j2", "lt", 0.0, "min"),
    ("J2 < 0.5", "j2", "lt", 0.5, "min"),
    ("J3 > 0.0", "j3", "gt", 0.0, "max"),
    ("J3 > -0.5", "j3", "gt", -0.5, "max"),
)


# ---------------------------------------------------------------------------
# Small conversion helpers
# ---------------------------------------------------------------------------


def _as_numpy(value: Any) -> np.ndarray:
    """torch.Tensor / numpy / list -> numpy without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value: Any) -> int:
    arr = _as_numpy(value).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"expected a scalar, got shape {arr.shape}")
    return int(arr[0])


def _flat_vector(value: Any, key: str) -> np.ndarray:
    arr = np.asarray(_as_numpy(value), dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError(f"{key} must be a finite non-empty vector, got {arr.shape}")
    return arr


def _rgb_hwc_uint8(image: Any, name: str) -> np.ndarray:
    """Normalize a recorded image to HWC uint8 RGB for ``predict_action``.

    LeRobotDataset v0.4.4 returns CHW float32 in [0, 1]. The deployment helper
    expects HWC uint8 and divides by 255 internally. Feeding CHW float directly
    would silently double-normalize and mis-permute axes.
    """
    img = _as_numpy(image)
    if img.ndim != 3:
        raise ValueError(f"{name}: image must have 3 dims, got {img.shape}")
    if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] != 3:
        raise ValueError(f"{name}: expected 3 RGB channels, got shape {img.shape}")
    if img.dtype == np.uint8:
        return np.ascontiguousarray(img)
    img = img.astype(np.float32, copy=False)
    if not np.all(np.isfinite(img)):
        raise ValueError(f"{name}: image contains NaN/Inf")
    low, high = float(img.min()), float(img.max())
    if -0.01 <= low and high <= 1.01:
        out = np.rint(np.clip(img, 0.0, 1.0) * 255.0)
    elif -0.01 <= low and high <= 255.01:
        out = np.rint(np.clip(img, 0.0, 255.0))
    else:
        raise ValueError(f"{name}: unexpected image range [{low}, {high}]")
    return np.ascontiguousarray(out.astype(np.uint8))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


# ---------------------------------------------------------------------------
# CLI parsing / names / boundaries
# ---------------------------------------------------------------------------


def parse_episodes(text: str) -> list[int]:
    try:
        episodes = sorted({int(part.strip()) for part in str(text).split(",") if part.strip()})
    except ValueError as exc:
        raise ValueError("--episodes must be comma-separated integers, e.g. 0,1,2") from exc
    if not episodes:
        raise ValueError("--episodes must contain at least one episode index")
    if any(ep < 0 for ep in episodes):
        raise ValueError("--episodes must be nonnegative")
    return episodes


def _episodes_arg(text: str) -> list[int]:
    try:
        return parse_episodes(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _nonnegative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return value


def feature_names(info: Mapping[str, Any], key: str) -> list[str] | None:
    feature = (info.get("features") or {}).get(key) or {}
    names = feature.get("names")
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    return None


def label_names(names: Sequence[str] | None, dim: int) -> list[str]:
    if names is not None and len(names) == dim:
        return [str(name) for name in names]
    if dim == len(JOINT_FALLBACK_NAMES):
        return list(JOINT_FALLBACK_NAMES)
    return [f"dim_{i}" for i in range(dim)]


def find_joint_indices(names: Sequence[str] | None, dim: int) -> tuple[int, int]:
    """Return (J2 index, J3 index), falling back to 1 and 2."""
    if dim < 3:
        raise ValueError(f"state/action must have at least 3 dims to locate J2/J3, got {dim}")
    found: dict[int, int] = {}
    if names is not None:
        for idx, raw in enumerate(names):
            compact = "".join(ch for ch in str(raw).lower() if ch.isalnum())
            for joint in (2, 3):
                if compact in (f"joint{joint}", f"j{joint}") or compact.startswith(
                    (f"joint{joint}deg", f"j{joint}deg")
                ):
                    found.setdefault(joint, idx)
    return found.get(2, 1), found.get(3, 2)


def _mask_stats(
    mask: np.ndarray,
    values: np.ndarray | None,
    kind: str,
    frames: np.ndarray | None,
) -> dict[str, Any]:
    count = int(mask.sum())
    result: dict[str, Any] = {
        "count": count,
        "ratio": float(count / mask.size) if mask.size else 0.0,
        "extreme": None,
        "first_frame": None,
        "first_value": None,
    }
    if count:
        first = int(np.flatnonzero(mask)[0])
        result["first_frame"] = int(frames[first]) if frames is not None else first
        if values is not None:
            selected = values[mask]
            result["extreme"] = float(selected.min() if kind == "min" else selected.max())
            result["first_value"] = float(values[first])
    return result


def boundary_stats(
    values: Any,
    j2_idx: int,
    j3_idx: int,
    episodes: Any | None = None,
    frames: Any | None = None,
    eps: float = BOUNDARY_EPS_DEG,
) -> dict[str, Any]:
    """J2/J3 zero-pinning and zero-cross statistics for a (N, D) array."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"boundary_stats expects a 2-D array, got {arr.shape}")
    if not 0 <= j2_idx < arr.shape[1] or not 0 <= j3_idx < arr.shape[1]:
        raise ValueError("J2/J3 index out of range")
    ep = None if episodes is None else np.asarray(episodes).astype(np.int64).reshape(-1)
    fr = None if frames is None else np.asarray(frames).astype(np.int64).reshape(-1)
    if ep is not None and ep.size != arr.shape[0]:
        raise ValueError("episodes length does not match values")
    if fr is not None and fr.size != arr.shape[0]:
        raise ValueError("frames length does not match values")

    j2 = arr[:, j2_idx]
    j3 = arr[:, j3_idx]
    pinned = (j2 <= eps) | (j3 >= -eps)
    crossed = (j2 < 0.0) | (j3 > 0.0)

    out: dict[str, Any] = {
        "frames": int(arr.shape[0]),
        "eps_deg": float(eps),
        "j2_min": float(j2.min()),
        "j3_max": float(j3.max()),
        "pinned": _mask_stats(pinned, None, "max", fr),
        "j2_pinned": _mask_stats(j2 <= eps, j2, "min", fr),
        "j3_pinned": _mask_stats(j3 >= -eps, j3, "max", fr),
        "zero_cross": _mask_stats(crossed, None, "max", fr),
        "j2_negative": _mask_stats(j2 < 0.0, j2, "min", fr),
        "j3_positive": _mask_stats(j3 > 0.0, j3, "max", fr),
    }
    if ep is not None:
        per_episode: dict[str, Any] = {}
        for episode in sorted({int(x) for x in ep}):
            mask = ep == episode
            sub_j2 = j2[mask]
            sub_j3 = j3[mask]
            sub_frames = fr[mask] if fr is not None else None
            per_episode[str(episode)] = {
                "frames": int(mask.sum()),
                "j2_min": float(sub_j2.min()),
                "j3_max": float(sub_j3.max()),
                "pinned": _mask_stats(
                    (sub_j2 <= eps) | (sub_j3 >= -eps), None, "max", sub_frames
                ),
                "zero_cross": _mask_stats(
                    (sub_j2 < 0.0) | (sub_j3 > 0.0), None, "max", sub_frames
                ),
            }
        out["per_episode"] = per_episode
    return out


# ---------------------------------------------------------------------------
# Dataset loading / summary
# ---------------------------------------------------------------------------


def load_dataset_metadata(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing dataset metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    return info, stats


def load_dataset_arrays(root: Path, info: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Read state/action/episode/frame columns from data/**/*.parquet (read-only)."""
    try:
        import pyarrow.dataset as pa_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pyarrow is required for dataset auditing. Run with the LeRobot env, e.g. "
            "/home/jetson/miniconda3/envs/lerobot/bin/python"
        ) from exc

    data_dir = root / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"missing dataset data directory: {data_dir}")
    columns = ["observation.state", "action", "episode_index", "frame_index"]
    try:
        table = pa_dataset.dataset(str(data_dir), format="parquet").to_table(columns=columns)
    except Exception as exc:
        raise RuntimeError(f"failed to read parquet columns {columns} under {data_dir}: {exc}") from exc

    def as_array(name: str, dtype: Any) -> np.ndarray:
        try:
            column = table.column(name)
        except KeyError as exc:
            raise RuntimeError(f"parquet data is missing required column {name!r}") from exc
        return np.asarray(column.to_pylist(), dtype=dtype)

    state = as_array("observation.state", np.float32)
    action = as_array("action", np.float32)
    ep = as_array("episode_index", np.int64).reshape(-1)
    frame = as_array("frame_index", np.int64).reshape(-1)
    if state.ndim != 2 or action.ndim != 2:
        raise RuntimeError(f"expected 2-D state/action, got {state.shape} / {action.shape}")
    if not (state.shape[0] == action.shape[0] == ep.size == frame.size):
        raise RuntimeError("state/action/episode/frame row counts disagree")
    order = np.lexsort((frame, ep))
    return {
        "state": state[order],
        "action": action[order],
        "episode_index": ep[order],
        "frame_index": frame[order],
    }


def _extrema(arr: np.ndarray, labels: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "name": labels[i],
            "min": float(arr[:, i].min()),
            "max": float(arr[:, i].max()),
            "mean": float(arr[:, i].mean()),
        }
        for i in range(arr.shape[1])
    ]


def summarize_arrays(
    state: Any,
    action: Any,
    episode_index: Any,
    frame_index: Any,
    info: Mapping[str, Any] | None = None,
    eps: float = BOUNDARY_EPS_DEG,
) -> dict[str, Any]:
    """Pure-numpy dataset summary (unit-testable without pyarrow/videos)."""
    st = np.asarray(state, dtype=np.float32)
    ac = np.asarray(action, dtype=np.float32)
    ep = np.asarray(episode_index).astype(np.int64).reshape(-1)
    fr = np.asarray(frame_index).astype(np.int64).reshape(-1)
    if st.ndim != 2 or ac.ndim != 2 or st.shape != ac.shape:
        raise ValueError(f"state/action must be same-shape 2-D arrays, got {st.shape} / {ac.shape}")
    if st.shape[0] == 0:
        raise ValueError("dataset contains zero frames")
    if not (ep.size == fr.size == st.shape[0]):
        raise ValueError("episode/frame arrays must match state rows")
    if not np.isfinite(st).all() or not np.isfinite(ac).all():
        raise ValueError("state/action contain NaN/Inf")

    order = np.lexsort((fr, ep))
    st, ac, ep, fr = st[order], ac[order], ep[order], fr[order]
    info = info or {}
    state_names = label_names(feature_names(info, "observation.state"), st.shape[1])
    action_names = label_names(feature_names(info, "action"), ac.shape[1])
    try:
        state_j2_idx, state_j3_idx = find_joint_indices(state_names, st.shape[1])
    except ValueError:
        state_j2_idx, state_j3_idx = 1, 2
    try:
        action_j2_idx, action_j3_idx = find_joint_indices(action_names, ac.shape[1])
    except ValueError:
        action_j2_idx, action_j3_idx = 1, 2

    first_frames: dict[str, Any] = {}
    episode_lengths: dict[str, int] = {}
    for episode in sorted({int(x) for x in ep}):
        mask = ep == episode
        idxs = np.flatnonzero(mask)
        first_local = idxs[np.argmin(fr[mask])]
        first_frames[str(episode)] = {
            "frame_index": int(fr[first_local]),
            "state": {state_names[i]: float(st[first_local, i]) for i in range(st.shape[1])},
            "action": {action_names[i]: float(ac[first_local, i]) for i in range(ac.shape[1])},
        }
        episode_lengths[str(episode)] = int(mask.sum())

    total_episodes = int(len(episode_lengths))
    total_frames = int(st.shape[0])
    integrity = {
        "info_total_episodes": info.get("total_episodes"),
        "info_total_frames": info.get("total_frames"),
        "actual_total_episodes": total_episodes,
        "actual_total_frames": total_frames,
        "episodes_match": info.get("total_episodes") in (None, total_episodes),
        "frames_match": info.get("total_frames") in (None, total_frames),
    }
    return {
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "codebase_version": info.get("codebase_version"),
        "state_names": state_names,
        "action_names": action_names,
        "state_extrema": _extrema(st, state_names),
        "action_extrema": _extrema(ac, action_names),
        "first_frames": first_frames,
        "episode_lengths": episode_lengths,
        "boundary": {
            "state": boundary_stats(st, state_j2_idx, state_j3_idx, ep, fr, eps),
            "action": boundary_stats(ac, action_j2_idx, action_j3_idx, ep, fr, eps),
        },
        "integrity": integrity,
    }


def camera_features(info: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    features = info.get("features") or {}
    return {
        str(key): dict(value)
        for key, value in features.items()
        if isinstance(value, Mapping) and value.get("dtype") in ("video", "image")
    }


# ---------------------------------------------------------------------------
# Checkpoint contract
# ---------------------------------------------------------------------------


def resolve_policy_dir(policy_path: Path) -> tuple[Path, dict[str, Any], Path]:
    path = policy_path.expanduser()
    if path.is_file() and path.name == "config.json":
        config_path = path
        policy_dir = path.parent
    else:
        direct = path / "config.json"
        nested = path / "pretrained_model" / "config.json"
        if direct.is_file():
            policy_dir = path
            config_path = direct
        elif nested.is_file():
            policy_dir = path / "pretrained_model"
            config_path = nested
        else:
            config_path = direct
            policy_dir = path
    if not config_path.is_file():
        raise FileNotFoundError(
            f"checkpoint config not found (looked for config.json or pretrained_model/config.json under {path})"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return policy_dir.resolve(), config, config_path.resolve()


def _shape(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, (list, tuple)) and all(isinstance(x, (int, np.integer)) for x in value):
        return tuple(int(x) for x in value)
    return None


def _to_hwc(shape: tuple[int, ...] | None) -> tuple[int, ...] | None:
    if shape is None:
        return None
    if len(shape) == 3:
        if shape[-1] in (1, 3, 4):
            return shape
        if shape[0] in (1, 3, 4):
            return (shape[1], shape[2], shape[0])
    return shape


def _stats_dim(stats: Mapping[str, Any], key: str) -> int | None:
    entry = stats.get(key) or {}
    for field in ("mean", "std", "min", "max"):
        value = entry.get(field)
        if isinstance(value, list):
            return len(value)
    return None


def check_contract(
    info: Mapping[str, Any], policy_config: Mapping[str, Any], stats: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Compare checkpoint input/output features with the dataset features."""
    stats = stats or {}
    dataset_features = info.get("features") or {}
    inputs = policy_config.get("input_features") or {}
    outputs = policy_config.get("output_features") or {}
    errors: list[str] = []
    warnings: list[str] = []

    checkpoint_type = str(policy_config.get("type", "unknown"))
    if checkpoint_type != "act":
        warnings.append(f"checkpoint type is {checkpoint_type!r}, expected 'act'")

    # --- state / action vector contract ---
    def vector_check(role: str, key: str, features: Mapping[str, Any]) -> dict[str, Any]:
        dataset_shape = _shape((dataset_features.get(key) or {}).get("shape"))
        checkpoint_shape = _shape((features.get(key) or {}).get("shape"))
        status = "match"
        if key not in dataset_features:
            status = "missing_in_dataset"
            errors.append(f"{role} key {key!r} is required by checkpoint but absent from dataset")
        elif checkpoint_shape is None:
            status = "missing_in_checkpoint"
            errors.append(f"{role} key {key!r} has no usable shape in checkpoint")
        elif dataset_shape != checkpoint_shape:
            status = "shape_mismatch"
            errors.append(
                f"{role} {key!r} dim mismatch: dataset {dataset_shape} vs checkpoint {checkpoint_shape}"
            )
        return {"key": key, "dataset_shape": dataset_shape, "checkpoint_shape": checkpoint_shape, "status": status}

    state_check = vector_check("input", "observation.state", inputs)
    action_check = vector_check("output", "action", outputs) if outputs else {
        "key": "action",
        "dataset_shape": None,
        "checkpoint_shape": None,
        "status": "missing_in_checkpoint",
    }
    if not outputs:
        errors.append("checkpoint has no output_features")

    # --- camera contract ---
    dataset_cameras = camera_features(info)
    checkpoint_cameras = {
        str(key): dict(value)
        for key, value in inputs.items()
        if isinstance(value, Mapping) and value.get("type") == "VISUAL"
    }
    missing = sorted(set(checkpoint_cameras) - set(dataset_cameras))
    extra = sorted(set(dataset_cameras) - set(checkpoint_cameras))
    if missing:
        errors.append(f"checkpoint cameras absent from dataset: {missing}")
    if extra:
        warnings.append(
            f"dataset cameras unused by checkpoint: {extra} "
            f"(recorded={len(dataset_cameras)}, expected={len(checkpoint_cameras)}; "
            "classic 2-camera vs 3-camera mismatch)"
        )

    camera_checks: list[dict[str, Any]] = []
    for key in sorted(set(checkpoint_cameras) & set(dataset_cameras)):
        dataset_shape = _shape(dataset_cameras[key].get("shape"))
        dataset_hwc = _to_hwc(dataset_shape)
        checkpoint_shape = _shape(checkpoint_cameras[key].get("shape"))
        checkpoint_hwc = _to_hwc(checkpoint_shape)
        status = "match"
        if dataset_hwc != checkpoint_hwc:
            status = "shape_mismatch"
            errors.append(
                f"camera {key!r} shape mismatch: dataset HWC {dataset_hwc} vs checkpoint CHW->HWC {checkpoint_hwc}"
            )
        camera_checks.append(
            {
                "key": key,
                "dataset_shape": dataset_shape,
                "dataset_hwc": dataset_hwc,
                "checkpoint_shape": checkpoint_shape,
                "checkpoint_hwc": checkpoint_hwc,
                "status": status,
            }
        )

    # --- extra non-camera inputs ---
    for key, value in inputs.items():
        if key == "observation.state" or (isinstance(value, Mapping) and value.get("type") == "VISUAL"):
            continue
        if key not in dataset_features:
            errors.append(f"checkpoint input {key!r} is absent from dataset")

    # --- normalization stats (diagnostic; checkpoint carries its own stats) ---
    normalization_mapping = policy_config.get("normalization_mapping") or {}
    normalization: list[dict[str, Any]] = []
    for key in list(inputs) + list(outputs):
        feature = inputs.get(key) or outputs.get(key) or {}
        feature_type = str(feature.get("type", ""))
        mode = normalization_mapping.get(feature_type)
        expected_dim = _shape(feature.get("shape"))
        stats_dim = _stats_dim(stats, key)
        status = "not_checked"
        if mode is None:
            status = "no_mapping"
        elif not stats:
            status = "dataset_stats_missing"
            warnings.append(f"dataset stats.json is missing; cannot cross-check {key!r}")
        elif stats_dim is None:
            status = "feature_stats_missing"
            warnings.append(f"dataset stats.json has no mean/std for {key!r}")
        else:
            expected = None
            if expected_dim is not None:
                expected = expected_dim[0] if len(expected_dim) == 1 else (
                    expected_dim[0] if feature_type == "VISUAL" else None
                )
            if feature_type == "VISUAL":
                expected = 3
            if expected is not None and stats_dim != expected:
                status = "dim_mismatch"
                warnings.append(
                    f"dataset normalization stats for {key!r} have dim {stats_dim}, expected {expected}"
                )
            else:
                status = "match"
        normalization.append(
            {
                "key": key,
                "type": feature_type,
                "mode": mode,
                "checkpoint_shape": expected_dim,
                "dataset_stats_dim": stats_dim,
                "status": status,
            }
        )

    deployable = not errors
    strict_match = deployable and not extra and all(c["status"] == "match" for c in camera_checks)
    return {
        "checkpoint_type": checkpoint_type,
        "dataset_camera_keys": sorted(dataset_cameras),
        "checkpoint_camera_keys": sorted(checkpoint_cameras),
        "missing_cameras": missing,
        "extra_cameras": extra,
        "state": state_check,
        "action": action_check,
        "cameras": camera_checks,
        "normalization": normalization,
        "errors": errors,
        "warnings": warnings,
        "ok": deployable,
        "strict_match": strict_match,
    }


# ---------------------------------------------------------------------------
# Open-loop inference
# ---------------------------------------------------------------------------


def build_observation(sample: Mapping[str, Any], expected_inputs: Sequence[str]) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for key in expected_inputs:
        if key not in sample:
            missing.append(key)
            continue
        if key.startswith("observation.images."):
            observation[key] = _rgb_hwc_uint8(sample[key], key)
        else:
            observation[key] = _flat_vector(sample[key], key)
    if missing:
        raise ValueError(f"sample is missing checkpoint inputs: {missing}")
    extra = set(observation) - set(expected_inputs)
    if extra:  # pragma: no cover - defensive
        raise ValueError(f"unexpected observation keys: {sorted(extra)}")
    return observation


def reset_pipeline(policy: Any, preprocessor: Any = None, postprocessor: Any = None) -> None:
    for obj in (policy, preprocessor, postprocessor):
        if obj is not None and hasattr(obj, "reset"):
            obj.reset()


def raw_image_audit(sample: Mapping[str, Any], expected_inputs: Sequence[str]) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for key in sorted(expected_inputs):
        if not key.startswith("observation.images."):
            continue
        raw = _as_numpy(sample[key])
        converted = _rgb_hwc_uint8(raw, key)
        audit[key] = {
            "dataset_dtype": str(raw.dtype),
            "dataset_shape": list(raw.shape),
            "dataset_min": float(raw.min()),
            "dataset_max": float(raw.max()),
            "deploy_dtype": str(converted.dtype),
            "deploy_shape": list(converted.shape),
            "deploy_min": int(converted.min()),
            "deploy_max": int(converted.max()),
        }
    return audit


def run_open_loop_episode(
    dataset: Any,
    episode: int,
    max_frames: int,
    expected_inputs: Sequence[str],
    predict_fn: Callable[[dict[str, np.ndarray]], Any],
    reset_fn: Callable[[], None] | None = None,
    image_audit: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run frame-by-frame inference for one episode, resetting at its start."""
    total = len(dataset)
    if total <= 0:
        raise RuntimeError(f"episode {episode} has zero frames")
    limit = total if max_frames <= 0 else min(total, max_frames)
    if reset_fn is not None:
        reset_fn()

    records: list[dict[str, Any]] = []
    for local_index in range(limit):
        sample = dataset[local_index]
        if "episode_index" in sample and _scalar(sample["episode_index"]) != episode:
            raise RuntimeError(
                f"dataset episode mismatch at local index {local_index}: "
                f"expected {episode}, got {_scalar(sample['episode_index'])}"
            )
        if image_audit is not None and local_index == 0:
            image_audit.update(raw_image_audit(sample, expected_inputs))
        observation = build_observation(sample, expected_inputs)
        pred = np.asarray(_as_numpy(predict_fn(observation)), dtype=np.float32).reshape(-1)
        state = _flat_vector(sample["observation.state"], "observation.state")
        label = _flat_vector(sample["action"], "action")
        if not (pred.size == state.size == label.size):
            raise RuntimeError(
                f"dimension mismatch at episode {episode} frame {local_index}: "
                f"pred={pred.size}, state={state.size}, label={label.size}"
            )
        if not np.isfinite(pred).all():
            raise RuntimeError(f"non-finite prediction at episode {episode} frame {local_index}")
        records.append(
            {
                "episode": int(episode),
                "frame": int(local_index),
                "dataset_frame": int(_scalar(sample["frame_index"])) if "frame_index" in sample else int(local_index),
                "state": state.copy(),
                "action": label.copy(),
                "pred": pred.copy(),
            }
        )
    return records


def _bound_record_first(records: Sequence[Mapping[str, Any]], mask: np.ndarray) -> dict[str, Any] | None:
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        return None
    rec = records[int(idxs[0])]
    return {"episode": int(rec["episode"]), "frame": int(rec["frame"])}


def compute_action_metrics(
    records: Sequence[Mapping[str, Any]], names: Sequence[str] | None = None
) -> dict[str, Any]:
    """MAE / max error / out-of-bound counts / first jump / adjacent step."""
    if not records:
        raise ValueError("no prediction records")
    dim = int(np.asarray(records[0]["pred"]).reshape(-1).size)
    labels = label_names(names, dim)
    try:
        j2_idx, j3_idx = find_joint_indices(labels, dim)
    except ValueError:
        j2_idx, j3_idx = 1, 2

    pred = np.stack([np.asarray(r["pred"], dtype=np.float64).reshape(-1) for r in records])
    label = np.stack([np.asarray(r["action"], dtype=np.float64).reshape(-1) for r in records])
    state = np.stack([np.asarray(r["state"], dtype=np.float64).reshape(-1) for r in records])
    if not (pred.shape == label.shape == state.shape):
        raise ValueError("pred/action/state shapes disagree")

    abs_err = np.abs(pred - label)
    per_joint: dict[str, Any] = {}
    for i, name in enumerate(labels):
        col = abs_err[:, i]
        best = int(np.argmax(col))
        per_joint[name] = {
            "mae": float(col.mean()),
            "max_abs_err": float(col[best]),
            "max_abs_err_episode": int(records[best]["episode"]),
            "max_abs_err_frame": int(records[best]["frame"]),
        }

    bounds: dict[str, Any] = {}
    for cond_label, which, op, threshold, extreme_kind in PRED_BOUND_CONDITIONS:
        idx = j2_idx if which == "j2" else j3_idx
        values = pred[:, idx]
        mask = values < threshold if op == "lt" else values > threshold
        count = int(mask.sum())
        extreme = None
        if count:
            extreme = float(values[mask].min() if extreme_kind == "min" else values[mask].max())
        bounds[cond_label] = {
            "count": count,
            "ratio": float(count / pred.shape[0]),
            "extreme": extreme,
            "first": _bound_record_first(records, mask),
            "first_value": float(values[np.flatnonzero(mask)[0]]) if count else None,
        }

    episode_ids = sorted({int(r["episode"]) for r in records}, key=lambda x: int(x))
    per_episode: dict[str, Any] = {}
    overall_jump = {name: 0.0 for name in labels}
    overall_step = {name: {"max_step": 0.0, "episode": None, "frame_from": None, "frame_to": None} for name in labels}
    for episode in episode_ids:
        idxs = [i for i, r in enumerate(records) if int(r["episode"]) == episode]
        ep_records = [records[i] for i in idxs]
        ep_pred = pred[idxs]
        ep_state = state[idxs]
        jump = np.abs(ep_pred[0] - ep_state[0])
        jump_dict = {
            name: {"jump": float(jump[i]), "pred": float(ep_pred[0, i]), "state": float(ep_state[0, i])}
            for i, name in enumerate(labels)
        }
        for name in labels:
            overall_jump[name] = max(overall_jump[name], jump_dict[name]["jump"])
        step_dict: dict[str, Any] = {}
        if ep_pred.shape[0] >= 2:
            steps = np.abs(np.diff(ep_pred, axis=0))
            for i, name in enumerate(labels):
                col = steps[:, i]
                best = int(np.argmax(col))
                step_dict[name] = {
                    "max_step": float(col[best]),
                    "frame_from": int(ep_records[best]["frame"]),
                    "frame_to": int(ep_records[best + 1]["frame"]),
                }
                current = overall_step[name]
                if step_dict[name]["max_step"] > current["max_step"]:
                    overall_step[name] = {
                        "max_step": step_dict[name]["max_step"],
                        "episode": int(episode),
                        "frame_from": step_dict[name]["frame_from"],
                        "frame_to": step_dict[name]["frame_to"],
                    }
        else:
            step_dict = {name: {"max_step": 0.0, "frame_from": None, "frame_to": None} for name in labels}
        per_episode[str(episode)] = {
            "frames": len(ep_records),
            "first_frame_jump": jump_dict,
            "adjacent_max_step": step_dict,
        }

    return {
        "frames": int(pred.shape[0]),
        "episodes": per_episode,
        "per_joint": per_joint,
        "pred_extrema": _extrema(pred, labels),
        "label_extrema": _extrema(label, labels),
        "bounds": bounds,
        "first_frame_jump_max": overall_jump,
        "adjacent_max_step": overall_step,
    }


# ---------------------------------------------------------------------------
# Visual grid / reports
# ---------------------------------------------------------------------------


def build_visual_grid(
    images: Sequence[Sequence[np.ndarray]],
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    padding: int = 8,
    label_height: int = 26,
    background: tuple[int, int, int] = (24, 24, 24),
    text_color: tuple[int, int, int] = (235, 235, 235),
):
    """Build a labelled rows x columns PNG-ready PIL image (RGB uint8)."""
    from PIL import Image, ImageDraw

    if not images or not images[0]:
        raise ValueError("visual grid needs at least one image")
    rows, cols = len(images), len(images[0])
    if len(row_labels) != rows or len(column_labels) != cols:
        raise ValueError("row/column label counts do not match the image grid")
    cell_h, cell_w = images[0][0].shape[:2]
    for row in images:
        if len(row) != cols:
            raise ValueError("ragged visual grid")
        for img in row:
            if img.shape[:2] != (cell_h, cell_w):
                raise ValueError("all grid images must share the same resolution")

    width = cols * cell_w + (cols + 1) * padding
    height = rows * (cell_h + label_height) + (rows + 1) * padding
    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    for r in range(rows):
        for c in range(cols):
            x = padding + c * (cell_w + padding)
            y = padding + r * (cell_h + label_height + padding)
            label = f"{row_labels[r]} | {column_labels[c]}"
            draw.text((x + 2, y + max(2, (label_height - 12) // 2)), label, fill=text_color)
            arr = np.asarray(images[r][c], dtype=np.uint8)
            canvas.paste(Image.fromarray(arr, mode="RGB"), (x, y + label_height))
    return canvas


def export_visual_grid(
    dataset: Any,
    episode: int,
    camera_keys: Sequence[str],
    output_path: Path,
    max_frames: int = 0,
) -> dict[str, Any]:
    """Save start + middle frames of one episode as a labelled PNG grid."""
    total = len(dataset)
    if total <= 0:
        raise RuntimeError(f"episode {episode} has zero frames")
    usable = total if max_frames <= 0 else min(total, max_frames)
    indices = sorted({0, usable // 2})
    frames: list[list[np.ndarray]] = []
    labels: list[str] = []
    for frame_idx in indices:
        sample = dataset[frame_idx]
        row = [_rgb_hwc_uint8(sample[key], key) for key in camera_keys]
        frames.append(row)
        labels.append(f"episode {episode} frame {frame_idx}")
    grid = build_visual_grid(frames, labels, list(camera_keys))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return {
        "path": str(output_path),
        "episode": int(episode),
        "frames": indices,
        "camera_keys": list(camera_keys),
        "grid_shape": [int(grid.height), int(grid.width)],
    }


def write_json_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_predictions_csv(
    path: Path, records: Sequence[Mapping[str, Any]], names: Sequence[str] | None = None
) -> None:
    if not records:
        raise ValueError("no records to write")
    dim = int(np.asarray(records[0]["pred"]).reshape(-1).size)
    labels = label_names(names, dim)
    fieldnames = ["episode", "frame", "dataset_frame"]
    for prefix in ("state", "label", "pred", "abs_err"):
        fieldnames.extend(f"{prefix}_{name}" for name in labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            state = np.asarray(rec["state"], dtype=float)
            label = np.asarray(rec["action"], dtype=float)
            pred = np.asarray(rec["pred"], dtype=float)
            row: dict[str, Any] = {
                "episode": int(rec["episode"]),
                "frame": int(rec["frame"]),
                "dataset_frame": int(rec["dataset_frame"]),
            }
            for i, name in enumerate(labels):
                row[f"state_{name}"] = float(state[i])
                row[f"label_{name}"] = float(label[i])
                row[f"pred_{name}"] = float(pred[i])
                row[f"abs_err_{name}"] = abs(float(pred[i] - label[i]))
            writer.writerow(row)


def ensure_safe_output_dir(
    output_dir: Path,
    dataset_root: Path,
    policy_dir: Path,
    protected_paths: Sequence[Path] = (),
) -> Path:
    out = output_dir.expanduser().resolve()
    protected = [("dataset", dataset_root.resolve()), ("checkpoint", policy_dir.resolve())]
    protected.extend(("checkpoint", path.expanduser().resolve()) for path in protected_paths)
    for label, directory in protected:
        if out == directory or directory in out.parents:
            raise ValueError(
                f"refusing to write output inside the read-only {label} directory: {out} (protected: {directory})"
            )
    if out.exists() and not out.is_dir():
        raise ValueError(f"--output-dir exists but is not a directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Console rendering
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_dataset_summary(summary: Mapping[str, Any], info: Mapping[str, Any]) -> None:
    print("\n=== 1. 数据集一致性 ===")
    print(
        f"episodes={summary['total_episodes']} frames={summary['total_frames']} "
        f"fps={summary.get('fps')} robot={summary.get('robot_type')} "
        f"version={summary.get('codebase_version')}"
    )
    integrity = summary["integrity"]
    if not integrity["episodes_match"] or not integrity["frames_match"]:
        print(
            f"[WARN] info.json vs parquet mismatch: episodes "
            f"{integrity['info_total_episodes']}->{integrity['actual_total_episodes']}, frames "
            f"{integrity['info_total_frames']}->{integrity['actual_total_frames']}"
        )
    print("相机: key | storage | shape(HWC) | resolution")
    for key, feature in camera_features(info).items():
        shape = _shape(feature.get("shape"))
        hwc = _to_hwc(shape)
        resolution = f"{hwc[1]}x{hwc[0]}" if hwc is not None and len(hwc) == 3 else "-"
        print(f"  {key} | {feature.get('dtype')} | {shape} | {resolution}")

    for key in ("state_extrema", "action_extrema"):
        print(f"{key}: name | min | max | mean")
        for item in summary[key]:
            unit = "mm" if "mm" in item["name"].lower() else "deg"
            print(
                f"  {item['name']:<14} {_fmt(item['min'], 2):>10} {_fmt(item['max'], 2):>10} "
                f"{_fmt(item['mean'], 2):>10} {unit}"
            )

    print("每条 episode 第 0 帧（state / action）:")
    for episode, entry in summary["first_frames"].items():
        st = " ".join(f"{k}={_fmt(v, 2)}" for k, v in entry["state"].items())
        ac = " ".join(f"{k}={_fmt(v, 2)}" for k, v in entry["action"].items())
        print(f"  ep {episode:>3} f{entry['frame_index']:<4} state[{st}] action[{ac}]")

    print("J2/J3 边界钉死（J2<=0.55° 或 J3>=-0.55°）与越零（J2<0° 或 J3>0°）:")
    for key in ("state", "action"):
        stat = summary["boundary"][key]
        print(
            f"  {key}: pinned={stat['pinned']['count']}/{stat['frames']} "
            f"({stat['pinned']['ratio'] * 100:.2f}%) zero_cross={stat['zero_cross']['count']} "
            f"({stat['zero_cross']['ratio'] * 100:.2f}%) j2_min={stat['j2_min']:.3f} "
            f"j3_max={stat['j3_max']:.3f}"
        )


def print_contract(contract: Mapping[str, Any]) -> None:
    print("\n=== 2. Checkpoint 与数据集契约 ===")
    print(
        f"type={contract['checkpoint_type']} cameras: checkpoint={contract['checkpoint_camera_keys']} "
        f"dataset={contract['dataset_camera_keys']}"
    )
    print(
        f"state: {contract['state']['status']} dataset={contract['state']['dataset_shape']} "
        f"checkpoint={contract['state']['checkpoint_shape']}"
    )
    print(
        f"action: {contract['action']['status']} dataset={contract['action']['dataset_shape']} "
        f"checkpoint={contract['action']['checkpoint_shape']}"
    )
    for cam in contract["cameras"]:
        print(
            f"  camera {cam['key']}: {cam['status']} dataset(HWC)={cam['dataset_hwc']} "
            f"checkpoint(CHW)={cam['checkpoint_shape']}"
        )
    if contract["missing_cameras"]:
        print(f"  [MISSING] checkpoint requires cameras absent from dataset: {contract['missing_cameras']}")
    if contract["extra_cameras"]:
        print(f"  [EXTRA] dataset cameras not consumed by checkpoint: {contract['extra_cameras']}")
    for warning in contract["warnings"]:
        print(f"  [WARN] {warning}")
    for error in contract["errors"]:
        print(f"  [ERROR] {error}")
    verdict = "STRICT MATCH" if contract["strict_match"] else ("DEPLOYABLE (not strict)" if contract["ok"] else "MISMATCH")
    print(f"verdict: {verdict}")


def print_inference(metrics: Mapping[str, Any], image_audit: Mapping[str, Any] | None = None) -> None:
    print("\n=== 3. 离线开环推理 ===")
    print(f"evaluated frames={metrics['frames']} episodes={list(metrics['episodes'])}")
    if image_audit:
        print("输入图像通道/范围核验（数据集原始 vs 部署转换）:")
        for key, item in image_audit.items():
            print(
                f"  {key}: dataset {item['dataset_dtype']} {item['dataset_shape']} "
                f"[{item['dataset_min']:.3f},{item['dataset_max']:.3f}] -> deploy {item['deploy_dtype']} "
                f"{item['deploy_shape']} [{item['deploy_min']},{item['deploy_max']}]"
            )
    print("逐关节 MAE / max |err|:")
    for name, item in metrics["per_joint"].items():
        unit = "mm" if "mm" in name.lower() else "deg"
        print(
            f"  {name:<14} MAE={_fmt(item['mae']):>8} {unit}  max={_fmt(item['max_abs_err']):>8} "
            f"@ep{item['max_abs_err_episode']} f{item['max_abs_err_frame']}"
        )
    print("预测动作越界统计:")
    for label, item in metrics["bounds"].items():
        first = item["first"]
        where = f"ep{first['episode']} f{first['frame']}" if first else "-"
        print(
            f"  {label:<10} count={item['count']:>5} ratio={item['ratio'] * 100:6.2f}% "
            f"extreme={_fmt(item['extreme'])} first={where}"
        )
    print("首帧跳变 |pred[0]-state[0]| 最大值 / 相邻帧最大步长:")
    for name in metrics["first_frame_jump_max"]:
        jump = metrics["first_frame_jump_max"][name]
        step = metrics["adjacent_max_step"][name]
        unit = "mm" if "mm" in name.lower() else "deg"
        where = (
            f"ep{step['episode']} f{step['frame_from']}->f{step['frame_to']}" if step["episode"] is not None else "-"
        )
        print(f"  {name:<14} jump={_fmt(jump):>8} {unit}  max_step={_fmt(step['max_step']):>8} {unit} @{where}")


# ---------------------------------------------------------------------------
# Policy loading / main
# ---------------------------------------------------------------------------


def load_act_policy(
    policy_dir: Path,
    device: str,
    n_action_steps: int | None = None,
    temporal_ensemble_coeff: float | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.utils import get_safe_torch_device

    path = str(policy_dir)
    config = PreTrainedConfig.from_pretrained(path)
    if config.type != "act":
        raise ValueError(f"expected an ACT checkpoint, got type={config.type!r}")
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
            # Standard ACT temporal ensembling needs a fresh chunk every step.
            config.n_action_steps = 1

    queue_steps = int(config.n_action_steps)
    chunk_size = int(getattr(config, "chunk_size", queue_steps))
    if queue_steps < 1 or queue_steps > chunk_size:
        raise ValueError(f"n_action_steps={queue_steps} must be within 1..chunk_size({chunk_size})")
    if getattr(config, "temporal_ensemble_coeff", None) is not None and queue_steps != 1:
        raise ValueError("ACT temporal ensemble requires n_action_steps=1")

    torch_device = get_safe_torch_device(config.device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False; use --device cpu")

    print(
        f"loading ACT: {path} device={torch_device} chunk_size={chunk_size} "
        f"n_action_steps={queue_steps} temporal_ensemble={getattr(config, 'temporal_ensemble_coeff', None)}"
    )
    policy = ACTPolicy.from_pretrained(path, config=config)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": config.device}},
    )
    reset_pipeline(policy, preprocessor, postprocessor)
    return config, policy, preprocessor, postprocessor, torch_device


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only offline audit for LeRobot ACT dataset + checkpoint (NO CAN / NO CAMERA / NO ROBOT)"
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="local LeRobot dataset root")
    parser.add_argument("--policy-path", type=Path, required=True, help="ACT pretrained_model directory")
    parser.add_argument("--episodes", type=_episodes_arg, default=list(DEFAULT_EPISODES), help="e.g. 0,1,2")
    parser.add_argument(
        "--max-frames", type=_nonnegative_int, default=DEFAULT_MAX_FRAMES, help="per episode; 0 = all frames"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/act_offline_audit_out"))
    parser.add_argument("--n-action-steps", type=int, default=None, help="override checkpoint n_action_steps")
    parser.add_argument(
        "--temporal-ensemble-coeff", type=float, default=None, help="override ACT temporal ensemble coeff"
    )
    parser.add_argument("--task", default="", help="task string passed to predict_action (deployment default: blank)")
    parser.add_argument("--repo-id", default=None, help="local LeRobot repo id (default: local/<dataset-dir-name>)")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--skip-inference", action="store_true", help="dataset + contract audit only")
    parser.add_argument("--skip-vision-export", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    dataset_root = args.dataset_root.expanduser().resolve()
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"dataset metadata missing: {dataset_root / 'meta' / 'info.json'}")
    policy_dir, policy_config, config_path = resolve_policy_dir(args.policy_path)
    output_dir = ensure_safe_output_dir(
        args.output_dir, dataset_root, policy_dir, protected_paths=[args.policy_path]
    )

    print("NO CAN / NO CAMERA / NO ROBOT. Dataset and checkpoint are opened read-only.")
    print(f"dataset={dataset_root}")
    print(f"checkpoint={policy_dir} (config={config_path})")
    print(f"output={output_dir}")

    info, stats = load_dataset_metadata(dataset_root)
    arrays = load_dataset_arrays(dataset_root, info)
    summary = summarize_arrays(
        arrays["state"],
        arrays["action"],
        arrays["episode_index"],
        arrays["frame_index"],
        info=info,
    )
    contract = check_contract(info, policy_config, stats)
    print_dataset_summary(summary, info)
    print_contract(contract)

    report: dict[str, Any] = {
        "tool": "act_offline_audit",
        "mode": "READ_ONLY_OFFLINE",
        "dataset_root": str(dataset_root),
        "policy_path": str(policy_dir),
        "arguments": {
            "episodes": list(args.episodes),
            "max_frames": int(args.max_frames),
            "device": args.device,
            "n_action_steps": args.n_action_steps,
            "temporal_ensemble_coeff": args.temporal_ensemble_coeff,
            "task": args.task,
        },
        "dataset": summary,
        "contract": contract,
        "inference": None,
        "visual_samples": None,
    }

    records: list[dict[str, Any]] = []
    metrics: dict[str, Any] | None = None
    image_audit: dict[str, Any] = {}
    first_dataset = None
    first_episode = args.episodes[0] if args.episodes else 0

    if args.skip_inference:
        print("\n=== 3. 离线开环推理 ===\nskipped by --skip-inference")
    elif not contract["ok"]:
        print("\n=== 3. 离线开环推理 ===\nskipped: checkpoint/dataset contract is not deployable")
    else:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        cfg, policy, preprocessor, postprocessor, torch_device = load_act_policy(
            policy_dir,
            args.device,
            n_action_steps=args.n_action_steps,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        )
        from lerobot.utils.control_utils import predict_action

        repo_id = args.repo_id or f"local/{dataset_root.name}"
        expected_inputs = list((cfg.input_features or {}).keys())

        def make_predict_fn(task: str, robot_type: str) -> Callable[[dict[str, np.ndarray]], Any]:
            def predict(observation: dict[str, np.ndarray]) -> Any:
                return predict_action(
                    observation=observation,
                    policy=policy,
                    device=torch_device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=bool(getattr(cfg, "use_amp", False)),
                    task=task,
                    robot_type=robot_type,
                )

            return predict

        predict_fn = make_predict_fn(args.task, str(info.get("robot_type") or "piper_pika"))
        for episode in args.episodes:
            LOG.info("loading recorded episode %s", episode)
            dataset = LeRobotDataset(
                repo_id,
                root=dataset_root,
                episodes=[episode],
                video_backend=args.video_backend,
                download_videos=False,
            )
            if len(dataset) == 0:
                raise RuntimeError(f"episode {episode} returned zero frames")
            print(f"episode {episode}: frames={len(dataset)} (max {args.max_frames or 'all'})")
            audit = image_audit if episode == first_episode else None
            records.extend(
                run_open_loop_episode(
                    dataset=dataset,
                    episode=episode,
                    max_frames=args.max_frames,
                    expected_inputs=expected_inputs,
                    predict_fn=predict_fn,
                    reset_fn=lambda: reset_pipeline(policy, preprocessor, postprocessor),
                    image_audit=audit,
                )
            )
            if first_dataset is None:
                first_dataset = dataset

        metrics = compute_action_metrics(records, feature_names(info, "action"))
        print_inference(metrics, image_audit)
        report["inference"] = {
            "checkpoint_config": {
                "chunk_size": int(getattr(cfg, "chunk_size", 0)),
                "n_action_steps": int(getattr(cfg, "n_action_steps", 0)),
                "temporal_ensemble_coeff": getattr(cfg, "temporal_ensemble_coeff", None),
                "input_features": list(expected_inputs),
            },
            "image_audit": image_audit,
            "metrics": metrics,
        }
        predictions_path = output_dir / "act_predictions.csv"
        write_predictions_csv(predictions_path, records, feature_names(info, "action"))
        report["inference"]["predictions_csv"] = str(predictions_path)

        camera_keys = sorted(camera_features(info))
        if camera_keys and not args.skip_vision_export and first_dataset is not None:
            png_path = output_dir / f"act_visual_samples_ep{first_episode:03d}.png"
            visual = export_visual_grid(first_dataset, first_episode, camera_keys, png_path, args.max_frames)
            report["visual_samples"] = visual
            print("\n=== 4. 视觉样本 === ")
            print(f"saved {visual['path']} frames={visual['frames']} cameras={visual['camera_keys']}")

    report_path = output_dir / "act_audit_report.json"
    write_json_report(report_path, report)
    print(f"\nJSON report: {report_path}")
    if metrics is None and not args.skip_inference:
        print("NOTE: open-loop inference did not run; dataset/contract evidence is still in the JSON report.")

    if contract["errors"]:
        print("\nCONTRACT ERRORS (blocking):")
        for error in contract["errors"]:
            print(f"  - {error}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return run(args)
    except Exception as exc:
        LOG.exception("offline audit failed")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
