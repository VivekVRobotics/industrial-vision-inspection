"""Configurable classical-vision inspection pipeline."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class InspectionConfig:
    """Parameters for detecting dark, localized surface defects."""

    threshold: int = 80
    blur_kernel: int = 5
    min_area: int = 25
    max_area: int = 100_000
    roi: tuple[int, int, int, int] | None = None  # x, y, width, height

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 255:
            raise ValueError("threshold must be in [0, 255]")
        if self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd number")
        if self.min_area < 1 or self.max_area < self.min_area:
            raise ValueError("invalid defect area bounds")
        if self.roi is not None:
            x, y, w, h = self.roi
            if min(x, y, w, h) < 0 or w == 0 or h == 0:
                raise ValueError("roi must contain positive width and height")


@dataclass(frozen=True)
class Defect:
    area: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class InspectionResult:
    passed: bool
    defect_count: int
    defects: tuple[Defect, ...]
    defect_fraction: float


def inspect_image(image_path: str | Path, config: InspectionConfig | None = None) -> InspectionResult:
    """Inspect an image and return deterministic defect measurements.

    The baseline algorithm is deliberately explainable: grayscale conversion,
    Gaussian denoising, thresholding, and connected-component filtering.
    """
    config = config or InspectionConfig()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi_offset = (0, 0)
    if config.roi is not None:
        x, y, w, h = config.roi
        if x + w > gray.shape[1] or y + h > gray.shape[0]:
            raise ValueError("roi extends beyond image dimensions")
        gray = gray[y : y + h, x : x + w]
        roi_offset = (x, y)

    blurred = cv2.GaussianBlur(gray, (config.blur_kernel, config.blur_kernel), 0)
    _, mask = cv2.threshold(blurred, config.threshold, 255, cv2.THRESH_BINARY_INV)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    defects: list[Defect] = []
    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if not (config.min_area <= area <= config.max_area):
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT]) + roi_offset[0]
        y = int(stats[label, cv2.CC_STAT_TOP]) + roi_offset[1]
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        defects.append(Defect(area=area, bbox=(x, y, w, h)))

    inspected_area = gray.shape[0] * gray.shape[1]
    defect_fraction = sum(d.area for d in defects) / inspected_area if inspected_area else 0.0
    return InspectionResult(
        passed=not defects,
        defect_count=len(defects),
        defects=tuple(defects),
        defect_fraction=defect_fraction,
    )
