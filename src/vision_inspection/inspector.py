"""Recipe-driven, explainable industrial surface-defect inspection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .calibration import PixelScale
from .metrology import RegionMeasurement, measure_regions
from .preprocessing import PreprocessConfig, apply_morphology, normalize_illumination, threshold_image, to_grayscale


@dataclass(frozen=True)
class InspectionConfig:
    """An auditable inspection recipe for localized surface defects."""

    preprocess: PreprocessConfig = PreprocessConfig()
    segmentation_mode: str = "otsu"
    threshold: int = 80
    adaptive_block: int = 31
    adaptive_c: float = 3.0
    opening: int = 0
    closing: int = 0
    min_area_px: float = 25.0
    max_area_px: float = 100_000.0
    max_defects: int = 0
    max_defect_fraction: float = 0.0
    min_circularity: float | None = None
    max_aspect_ratio: float | None = None
    min_solidity: float | None = None
    reject_border_touching: bool = False
    roi: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.segmentation_mode not in {"fixed", "otsu", "adaptive"}:
            raise ValueError("segmentation_mode must be fixed, otsu, or adaptive")
        if not 0 <= self.threshold <= 255:
            raise ValueError("threshold must be in [0,255]")
        if self.min_area_px <= 0 or self.max_area_px < self.min_area_px:
            raise ValueError("invalid defect area bounds")
        if self.max_defects < 0:
            raise ValueError("max_defects must be non-negative")
        if not 0 <= self.max_defect_fraction <= 1:
            raise ValueError("max_defect_fraction must be in [0,1]")
        for name, value in (("min_circularity", self.min_circularity), ("min_solidity", self.min_solidity)):
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if self.max_aspect_ratio is not None and self.max_aspect_ratio < 1:
            raise ValueError("max_aspect_ratio must be >= 1")
        if self.roi is not None:
            x, y, w, h = self.roi
            if min(x, y, w, h) < 0 or w <= 0 or h <= 0:
                raise ValueError("roi must contain non-negative origin and positive size")


@dataclass(frozen=True)
class Defect:
    """One accepted defect candidate with measured geometry."""

    measurement: RegionMeasurement

    @property
    def area(self) -> float:
        return self.measurement.area_px

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.measurement.bbox_px


@dataclass(frozen=True)
class InspectionResult:
    """Immutable inspection decision and measured evidence."""

    passed: bool
    defect_count: int
    defect_fraction: float
    defects: tuple[Defect, ...]
    measurements: tuple[RegionMeasurement, ...]
    image_shape: tuple[int, int]
    roi: tuple[int, int, int, int] | None


def _touches_border(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    x, y, w, h = bbox
    width, height = shape[1], shape[0]
    return x <= 0 or y <= 0 or x + w >= width or y + h >= height


def inspect_array(
    image: np.ndarray,
    config: InspectionConfig | None = None,
    *,
    pixel_scale: PixelScale | None = None,
) -> InspectionResult:
    """Inspect an image array and return a deterministic, auditable decision."""
    config = config or InspectionConfig()
    gray = to_grayscale(image)
    roi_offset = (0, 0)
    inspected = gray
    if config.roi is not None:
        x, y, w, h = config.roi
        if x + w > gray.shape[1] or y + h > gray.shape[0]:
            raise ValueError("roi extends beyond image dimensions")
        inspected = gray[y : y + h, x : x + w]
        roi_offset = (x, y)

    corrected = normalize_illumination(inspected, config.preprocess)
    # Black-hat/top-hat output is a defect-enhancement image, so binary
    # segmentation always selects positive response values.
    mask = threshold_image(
        corrected,
        mode=config.segmentation_mode,
        threshold=config.threshold,
        adaptive_block=config.adaptive_block,
        adaptive_c=config.adaptive_c,
    )
    mask = apply_morphology(mask, opening=config.opening, closing=config.closing)
    measurements_local = measure_regions(mask, pixel_scale)

    measurements: list[RegionMeasurement] = []
    defects: list[Defect] = []
    for measurement in measurements_local:
        x, y, w, h = measurement.bbox_px
        shifted = RegionMeasurement(
            label=measurement.label,
            area_px=measurement.area_px,
            perimeter_px=measurement.perimeter_px,
            centroid_px=(measurement.centroid_px[0] + roi_offset[0], measurement.centroid_px[1] + roi_offset[1]),
            bbox_px=(x + roi_offset[0], y + roi_offset[1], w, h),
            width_px=measurement.width_px,
            height_px=measurement.height_px,
            aspect_ratio=measurement.aspect_ratio,
            circularity=measurement.circularity,
            extent=measurement.extent,
            solidity=measurement.solidity,
            area_physical=measurement.area_physical,
            perimeter_physical=measurement.perimeter_physical,
        )
        measurements.append(shifted)
        if not config.min_area_px <= shifted.area_px <= config.max_area_px:
            continue
        if config.reject_border_touching and _touches_border(measurement.bbox_px, inspected.shape):
            continue
        if config.min_circularity is not None and shifted.circularity < config.min_circularity:
            continue
        if config.max_aspect_ratio is not None and shifted.aspect_ratio > config.max_aspect_ratio:
            continue
        if config.min_solidity is not None and shifted.solidity < config.min_solidity:
            continue
        defects.append(Defect(shifted))

    inspected_area = float(inspected.shape[0] * inspected.shape[1])
    defect_fraction = sum(d.measurement.area_px for d in defects) / inspected_area if inspected_area else 0.0
    passed = len(defects) <= config.max_defects and defect_fraction <= config.max_defect_fraction
    return InspectionResult(
        passed=passed,
        defect_count=len(defects),
        defect_fraction=float(defect_fraction),
        defects=tuple(defects),
        measurements=tuple(measurements),
        image_shape=tuple(int(v) for v in gray.shape),
        roi=config.roi,
    )


def inspect_image(
    image_path: str | Path,
    config: InspectionConfig | None = None,
    *,
    pixel_scale: PixelScale | None = None,
) -> InspectionResult:
    """Load an image from disk and inspect it."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    return inspect_array(image, config, pixel_scale=pixel_scale)
