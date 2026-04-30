"""Camera utilities for SynCamVideo-Dataset.

This module only reads metadata from SynCamVideo-Dataset. It never writes into the
original dataset directory.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


Number = float | int


def parse_extrinsic_string(value: str | Sequence[Number]) -> np.ndarray:
    """Parse one camera extrinsic into a 4x4 float32 matrix.

    The dataset JSON stores matrices as strings such as:
        "[r11 r12 r13 tx] [r21 r22 r23 ty] [r31 r32 r33 tz]"

    This function also accepts a flat list of 12 or 16 numbers for robustness.
    """
    if isinstance(value, str):
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)]
    else:
        nums = [float(x) for x in value]

    arr = np.asarray(nums, dtype=np.float32)

    if arr.size == 12:
        mat = arr.reshape(3, 4)
        mat = np.vstack([mat, np.array([[0, 0, 0, 1]], dtype=np.float32)])
    elif arr.size == 16:
        mat = arr.reshape(4, 4)
    else:
        raise ValueError(f"Expected 12 or 16 numbers for extrinsic, got {arr.size}: {value}")

    return mat.astype(np.float32)


def load_scene_extrinsics(camera_extrinsics_json: str | Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Load scene camera extrinsics.

    Returns:
        dict[frame_key][cam_id] = 4x4 matrix
    """
    camera_extrinsics_json = Path(camera_extrinsics_json)
    with camera_extrinsics_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    parsed: Dict[str, Dict[str, np.ndarray]] = {}
    for frame_key, cam_dict in raw.items():
        parsed[frame_key] = {}
        for cam_id, matrix_like in cam_dict.items():
            parsed[frame_key][cam_id] = parse_extrinsic_string(matrix_like)
    return parsed


def matrix_to_12d(mat: np.ndarray) -> List[float]:
    """Convert 4x4 or 3x4 matrix to flattened 12D [R|t]."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape == (4, 4):
        mat = mat[:3, :4]
    elif mat.shape != (3, 4):
        raise ValueError(f"Expected matrix shape (4,4) or (3,4), got {mat.shape}")
    return mat.reshape(-1).astype(float).tolist()


def as_homogeneous_4x4(mat: np.ndarray) -> np.ndarray:
    """Convert 3x4 or 4x4 matrix to 4x4 homogeneous matrix."""
    mat = np.asarray(mat, dtype=np.float32)

    if mat.shape == (4, 4):
        return mat

    if mat.shape == (3, 4):
        bottom = np.array([[0, 0, 0, 1]], dtype=np.float32)
        return np.vstack([mat, bottom])

    raise ValueError(f"Expected matrix shape (3,4) or (4,4), got {mat.shape}")


def raw_extrinsic_to_syncam_c2w(mat: np.ndarray) -> np.ndarray:
    """Convert SynCamVideo raw camera matrix to SynCamMaster-style c2w.

    This follows the official SynCamMaster preprocessing:
        mat.T
        column reorder [1, 2, 0, 3]
        flip y axis
        translation /= 200
    """
    mat = as_homogeneous_4x4(mat)

    c2w = mat.T.copy()
    c2w = c2w[:, [1, 2, 0, 3]]
    c2w[:3, 1] *= -1.0
    c2w[:3, 3] /= 200.0

    return c2w.astype(np.float32)


def normalize_syncam_c2ws_to_anchor(
    c2ws: Sequence[np.ndarray],
    anchor_index: int = 0,
) -> np.ndarray:
    """Normalize c2w cameras by setting anchor camera as identity.

    Returns:
        [V, 12], flattened relative [R|t].
    """
    c2ws = [as_homogeneous_4x4(c2w) for c2w in c2ws]

    target_anchor_c2w = np.eye(4, dtype=np.float32)
    anchor_w2c = np.linalg.inv(c2ws[anchor_index])
    abs2rel = target_anchor_c2w @ anchor_w2c

    rels = []
    for c2w in c2ws:
        rel = abs2rel @ c2w
        rels.append(rel[:3, :4].reshape(-1))

    return np.stack(rels, axis=0).astype(np.float32)


def normalize_extrinsics_syncam(
    raw_mats: Sequence[np.ndarray],
    anchor_index: int = 0,
) -> np.ndarray:
    """Full SynCamMaster-compatible camera preprocessing.

    raw dataset matrix -> syncam c2w -> relative [R|t].
    """
    c2ws = [raw_extrinsic_to_syncam_c2w(m) for m in raw_mats]
    return normalize_syncam_c2ws_to_anchor(c2ws, anchor_index=anchor_index)


def rotation_angle_deg(Ea: np.ndarray, Eb: np.ndarray) -> float:
    """Rotation angle in degrees between two canonical c2w cameras.

    Important:
        Ea and Eb should already be processed by raw_extrinsic_to_syncam_c2w().
    """
    Ra = np.asarray(Ea, dtype=np.float32)[:3, :3]
    Rb = np.asarray(Eb, dtype=np.float32)[:3, :3]

    R = Ra.T @ Rb
    cos = (float(np.trace(R)) - 1.0) / 2.0
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def max_pairwise_rotation_angle_deg(mats: Sequence[np.ndarray]) -> float:
    """Maximum pairwise rotation angle.

    mats should be SynCamMaster-style c2w matrices, not raw JSON matrices.
    """
    if len(mats) < 2:
        return 0.0

    max_angle = 0.0
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            max_angle = max(max_angle, rotation_angle_deg(mats[i], mats[j]))

    return float(max_angle)