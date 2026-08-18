"""Golden-reference residual inspection.

Reference comparison is useful for appearance changes after part registration,
but raw pixel differences are sensitive to brightness shifts. The implementation
therefore reports both raw residual magnitude and thresholded localized changes,
with optional normalization of global intensity before comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ReferenceConfig:
    """Parameters for deterministic reference comparison."""

    threshold: int = 25
    blur_kernel: int = 3
    min_area_px: int = 20
    max_area_px: int = 100_000
    normalize_brightness: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 255 or self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ValueError("threshold must be in [0,255] and blur_kernel positive odd")
        if self.min_area_px <= 0 or self.max_area_px < self.min_area_px:
            raise ValueError("invalid area bounds")


@dataclass(frozen=True)
class ReferenceResidual:
    """Residual evidence from a reference comparison."""

    mask: np.ndarray
    changed_pixels: int
    changed_fraction: float
    mean_absolute_difference: float
    p95_absolute_difference: float
    max_absolute_difference: int


def _gray_uint8(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 2:
        gray = values
    elif values.ndim == 3 and values.shape[2] == 3:
        gray = cv2.cvtColor(values.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("image must be grayscale or BGR")
    return np.clip(gray, 0, 255).astype(np.uint8)


def _normalize_brightness(reference: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref_mean = float(np.mean(reference))
    img_mean = float(np.mean(image))
    if ref_mean <= 1e-6 or img_mean <= 1e-6:
        return reference, image
    scale = ref_mean / img_mean
    corrected = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return reference, corrected


def compare_to_reference(reference: np.ndarray, image: np.ndarray, config: ReferenceConfig | None = None) -> ReferenceResidual:
    """Detect localized appearance changes against a registered reference image."""
    config = config or ReferenceConfig()
    ref = _gray_uint8(reference)
    current = _gray_uint8(image)
    if ref.shape != current.shape:
        raise ValueError("reference and image must have matching spatial dimensions")
    if config.normalize_brightness:
        ref, current = _normalize_brightness(ref, current)
    ref_blur = cv2.GaussianBlur(ref, (config.blur_kernel, config.blur_kernel), 0)
    cur_blur = cv2.GaussianBlur(current, (config.blur_kernel, config.blur_kernel), 0)
    difference = cv2.absdiff(ref_blur, cur_blur)
    _, thresholded = cv2.threshold(difference, config.threshold, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded, connectivity=8)
    keep = np.zeros(num_labels, dtype=bool)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        keep[label] = config.min_area_px <= area <= config.max_area_px
    mask = np.where(keep[labels], 255, 0).astype(np.uint8)
    changed = int(np.count_nonzero(mask))
    total = mask.size
    return ReferenceResidual(
        mask=mask,
        changed_pixels=changed,
        changed_fraction=changed / total if total else 0.0,
        mean_absolute_difference=float(np.mean(difference)) if total else 0.0,
        p95_absolute_difference=float(np.percentile(difference, 95)) if total else 0.0,
        max_absolute_difference=int(np.max(difference)) if total else 0,
    )
