"""Flat-field illumination calibration and correction.

A flat field estimates multiplicative spatial response from uniform reference
frames. The implementation uses robust aggregation rather than a single frame,
rejects invalid calibration fields, and exposes field statistics so calibration
quality can be reported before deployment.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FlatFieldStats:
    """Diagnostics for a normalized illumination field."""

    minimum: float
    maximum: float
    mean: float
    std: float
    coefficient_of_variation: float


def build_flat_field(reference_frames: np.ndarray, *, robust: bool = True) -> np.ndarray:
    """Build a normalized 2-D illumination field from uniform reference frames.

    ``robust=True`` uses the per-pixel median across captures, which is less
    sensitive than a mean to a transient dust particle, glare event, or bad
    frame. Calibration captures should still be inspected for stability.
    """
    frames = np.asarray(reference_frames, dtype=np.float32)
    if frames.ndim not in {3, 4} or frames.shape[0] < 3:
        raise ValueError("reference_frames must contain at least three 2D/3D frames")
    if frames.ndim == 4:
        if frames.shape[3] != 3:
            raise ValueError("color frames must have three channels")
        frames = np.mean(frames, axis=3)
    if not np.all(np.isfinite(frames)) or np.any(frames < 0):
        raise ValueError("reference frames must be finite and non-negative")
    field = np.median(frames, axis=0) if robust else np.mean(frames, axis=0)
    reference_level = float(np.median(field))
    if reference_level <= 0:
        raise ValueError("reference illumination must have positive median intensity")
    normalized = field / reference_level
    return np.clip(normalized, 1e-3, None).astype(np.float32)


def flat_field_stats(field: np.ndarray) -> FlatFieldStats:
    """Summarize spatial non-uniformity of a normalized flat field."""
    values = np.asarray(field, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("field must be a finite positive 2D array")
    mean = float(np.mean(values))
    std = float(np.std(values))
    return FlatFieldStats(float(np.min(values)), float(np.max(values)), mean, std, std / mean if mean else float("inf"))


def apply_flat_field(image: np.ndarray, field: np.ndarray, *, clip: bool = True) -> np.ndarray:
    """Correct multiplicative illumination variation using a reference field."""
    image_array = np.asarray(image)
    correction = np.asarray(field, dtype=np.float32)
    if correction.ndim != 2 or image_array.shape[:2] != correction.shape:
        raise ValueError("image and flat-field must have matching spatial dimensions")
    if not np.all(np.isfinite(correction)) or np.any(correction <= 0):
        raise ValueError("flat-field must be finite and strictly positive")
    if image_array.ndim == 3:
        corrected = image_array.astype(np.float32) / correction[..., None]
    elif image_array.ndim == 2:
        corrected = image_array.astype(np.float32) / correction
    else:
        raise ValueError("image must be 2D or 3D")
    if clip and np.issubdtype(image_array.dtype, np.integer):
        info = np.iinfo(image_array.dtype)
        corrected = np.clip(corrected, info.min, info.max)
    return corrected.astype(image_array.dtype)


def smooth_flat_field(field: np.ndarray, kernel_size: int = 31) -> np.ndarray:
    """Remove high-frequency sensor noise from a field while preserving trends."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd and >=3")
    values = np.asarray(field, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("field must be a finite positive 2D array")
    smoothed = cv2.GaussianBlur(values, (kernel_size, kernel_size), 0)
    return np.clip(smoothed, 1e-3, None).astype(np.float32)
