"""Image-quality gates used before making an inspection decision."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityConfig:
    """Acceptance thresholds for image acquisition quality."""

    min_mean: float = 10.0
    max_mean: float = 245.0
    min_std: float = 5.0
    min_sharpness: float = 20.0
    max_saturated_fraction: float = 0.01

    def __post_init__(self) -> None:
        if not 0 <= self.min_mean < self.max_mean <= 255:
            raise ValueError("mean bounds must satisfy 0 <= min < max <= 255")
        if self.min_std < 0 or self.min_sharpness < 0:
            raise ValueError("quality thresholds must be non-negative")
        if not 0 <= self.max_saturated_fraction <= 1:
            raise ValueError("max_saturated_fraction must be in [0,1]")


@dataclass(frozen=True)
class ImageQuality:
    """Measured acquisition quality with explicit gate reasons."""

    mean: float
    std: float
    sharpness: float
    saturated_fraction: float
    passed: bool
    failures: tuple[str, ...]


def assess_image_quality(image: np.ndarray, config: QualityConfig | None = None) -> ImageQuality:
    """Measure exposure, contrast, sharpness, and saturation."""
    config = config or QualityConfig()
    image = np.asarray(image)
    if image.ndim not in {2, 3}:
        raise ValueError("image must be 2D grayscale or 3D color")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.number):
            raise ValueError("image must contain numeric pixel data")
        image = np.clip(image, 0, 255).astype(np.uint8)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    saturated = np.logical_or(gray <= 0, gray >= 255)
    saturated_fraction = float(np.mean(saturated))

    failures: list[str] = []
    if mean < config.min_mean:
        failures.append("underexposed")
    if mean > config.max_mean:
        failures.append("overexposed")
    if std < config.min_std:
        failures.append("low_contrast")
    if sharpness < config.min_sharpness:
        failures.append("low_sharpness")
    if saturated_fraction > config.max_saturated_fraction:
        failures.append("excessive_saturation")
    return ImageQuality(mean, std, sharpness, saturated_fraction, not failures, tuple(failures))
