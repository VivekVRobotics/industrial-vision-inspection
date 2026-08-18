"""Golden-reference residual inspection for stable part appearance."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ReferenceConfig:
    """Parameters for comparing a part image against a golden reference."""

    threshold: int = 25
    blur_kernel: int = 3
    min_area_px: int = 20
    max_area_px: int = 100_000

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 255:
            raise ValueError("threshold must be in [0,255]")
        if self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd number")
        if self.min_area_px <= 0 or self.max_area_px < self.min_area_px:
            raise ValueError("invalid area bounds")


@dataclass(frozen=True)
class ReferenceResidual:
    """Residual mask and summary statistics from golden-image comparison."""

    mask: np.ndarray
    changed_pixels: int
    changed_fraction: float
    mean_absolute_difference: float


def compare_to_reference(
    reference: np.ndarray,
    image: np.ndarray,
    config: ReferenceConfig | None = None,
) -> ReferenceResidual:
    """Detect localized appearance changes using absolute residuals."""
    config = config or ReferenceConfig()
    ref = np.asarray(reference)
    current = np.asarray(image)
    if ref.shape != current.shape or ref.ndim not in {2, 3}:
        raise ValueError("reference and image must have the same 2D/3D shape")
    if ref.ndim == 3 and ref.shape[2] != 3:
        raise ValueError("color reference images must be BGR with three channels")
    ref_gray = ref if ref.ndim == 2 else cv2.cvtColor(ref.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    cur_gray = current if current.ndim == 2 else cv2.cvtColor(current.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    ref_blur = cv2.GaussianBlur(ref_gray.astype(np.uint8), (config.blur_kernel, config.blur_kernel), 0)
    cur_blur = cv2.GaussianBlur(cur_gray.astype(np.uint8), (config.blur_kernel, config.blur_kernel), 0)
    difference = cv2.absdiff(ref_blur, cur_blur)
    _, thresholded = cv2.threshold(difference, config.threshold, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresholded, connectivity=8)
    mask = np.zeros_like(thresholded)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if config.min_area_px <= area <= config.max_area_px:
            mask[stats[label, cv2.CC_STAT_TOP]:stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT],
                 stats[label, cv2.CC_STAT_LEFT]:stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]] = np.maximum(
                mask[stats[label, cv2.CC_STAT_TOP]:stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT],
                     stats[label, cv2.CC_STAT_LEFT]:stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]],
                thresholded[stats[label, cv2.CC_STAT_TOP]:stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT],
                            stats[label, cv2.CC_STAT_LEFT]:stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]],
            )
    changed = int(np.count_nonzero(mask))
    total = mask.size
    return ReferenceResidual(
        mask=mask,
        changed_pixels=changed,
        changed_fraction=changed / total if total else 0.0,
        mean_absolute_difference=float(np.mean(difference)) if total else 0.0,
    )
