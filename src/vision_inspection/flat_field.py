"""Flat-field correction for repeatable machine-vision illumination."""

from __future__ import annotations

import cv2
import numpy as np


def build_flat_field(reference_frames: np.ndarray) -> np.ndarray:
    """Build a normalized illumination field from uniform reference frames."""
    frames = np.asarray(reference_frames, dtype=np.float32)
    if frames.ndim not in {3, 4} or frames.shape[0] < 2:
        raise ValueError("reference_frames must contain at least two 2D/3D frames")
    if frames.ndim == 4:
        if frames.shape[3] != 3:
            raise ValueError("color frames must have three channels")
        frames = np.mean(frames, axis=3)
    if not np.all(np.isfinite(frames)):
        raise ValueError("reference frames must be finite")
    field = np.mean(frames, axis=0)
    median = float(np.median(field))
    if median <= 0:
        raise ValueError("reference illumination must have positive median intensity")
    normalized = field / median
    return np.clip(normalized, 1e-3, None).astype(np.float32)


def apply_flat_field(image: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Correct multiplicative illumination variation using a reference field."""
    img = np.asarray(image)
    correction = np.asarray(field, dtype=np.float32)
    if img.ndim != 2 or correction.shape != img.shape:
        raise ValueError("image and flat-field must be matching 2D arrays")
    corrected = img.astype(np.float32) / correction
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        corrected = np.clip(corrected, info.min, info.max)
    return corrected.astype(img.dtype)


def smooth_flat_field(field: np.ndarray, kernel_size: int = 31) -> np.ndarray:
    """Suppress high-frequency sensor noise in a flat-field calibration."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd and >=3")
    field = np.asarray(field, dtype=np.float32)
    if field.ndim != 2 or not np.all(np.isfinite(field)):
        raise ValueError("field must be a finite 2D array")
    return cv2.GaussianBlur(field, (kernel_size, kernel_size), 0)
