"""Image-quality diagnostics used before inspection decisions.

A production inspection must distinguish an unacceptable image from an
acceptable part. The gate therefore reports exposure, contrast, blur, clipping,
and robust intensity percentiles instead of returning a single quality bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds used to reject unusable acquisition frames."""

    min_mean: float = 10.0
    max_mean: float = 245.0
    min_std: float = 5.0
    min_sharpness: float = 20.0
    max_saturated_fraction: float = 0.01
    min_p1: float = 2.0
    max_p99: float = 253.0

    def __post_init__(self) -> None:
        if not 0 <= self.min_mean < self.max_mean <= 255:
            raise ValueError("mean bounds must satisfy 0 <= min < max <= 255")
        if self.min_std < 0 or self.min_sharpness < 0:
            raise ValueError("quality thresholds must be non-negative")
        if not 0 <= self.max_saturated_fraction <= 1:
            raise ValueError("max_saturated_fraction must be in [0,1]")
        if not 0 <= self.min_p1 < self.max_p99 <= 255:
            raise ValueError("percentile bounds must lie in [0,255] and be ordered")


@dataclass(frozen=True)
class ImageQuality:
    """Measured frame quality and explicit failure reasons."""

    mean: float
    std: float
    p1: float
    p99: float
    sharpness: float
    saturated_fraction: float
    passed: bool
    failures: tuple[str, ...]


def assess_image_quality(image: np.ndarray, config: QualityConfig | None = None) -> ImageQuality:
    """Compute deterministic image-quality metrics and acceptance reasons."""
    config = config or QualityConfig()
    values = np.asarray(image)
    if values.ndim not in {2, 3} or values.size == 0:
        raise ValueError("image must be a non-empty 2D grayscale or 3D color array")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("image must contain numeric pixel data")
    values = np.clip(values, 0, 255).astype(np.uint8)
    gray = values if values.ndim == 2 else cv2.cvtColor(values, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    p1, p99 = (float(v) for v in np.percentile(gray, [1, 99]))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    saturated_fraction = float(np.mean((gray <= 0) | (gray >= 255)))
    failures: list[str] = []
    if mean < config.min_mean:
        failures.append("underexposed")
    if mean > config.max_mean:
        failures.append("overexposed")
    if std < config.min_std:
        failures.append("low_contrast")
    if p1 < config.min_p1:
        failures.append("low_shadow_margin")
    if p99 > config.max_p99:
        failures.append("low_highlight_margin")
    if sharpness < config.min_sharpness:
        failures.append("low_sharpness")
    if saturated_fraction > config.max_saturated_fraction:
        failures.append("excessive_saturation")
    return ImageQuality(mean, std, p1, p99, sharpness, saturated_fraction, not failures, tuple(failures))
