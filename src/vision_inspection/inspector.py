"""Recipe-driven, traceable industrial surface-defect inspection engine.

The inspection engine is deliberately a deterministic composition of stages:
quality gate -> ROI -> illumination normalization -> segmentation -> morphology
-> metrology -> acceptance rules. Each stage produces explainable evidence.

The result object retains recipe identity, input-image digest, measurements, and
processing time so a production wrapper can correlate a decision with the
exact image and rule set that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .calibration import PixelScale
from .metrology import RegionMeasurement, measure_regions
from .preprocessing import PreprocessConfig, apply_morphology, normalize_illumination, threshold_image, to_grayscale
from .quality import ImageQuality, QualityConfig, assess_image_quality


@dataclass(frozen=True)
class InspectionConfig:
    """Immutable inspection recipe.

    A recipe should be versioned outside this class when used in production.
    ``version`` is included in the evidence digest so changing a threshold or
    preprocessing parameter creates a different recipe identity.
    """

    version: str = "0.4.0"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    segmentation_mode: str = "otsu"
    threshold: int = 80
    adaptive_block: int = 31
    adaptive_c: float = 3.0
    segmentation_polarity: str = "positive"
    opening: int = 0
    closing: int = 0
    morphology_iterations: int = 1
    min_area_px: float = 25.0
    max_area_px: float = 100_000.0
    max_defects: int = 0
    max_defect_fraction: float = 0.0
    min_circularity: float | None = None
    max_aspect_ratio: float | None = None
    min_solidity: float | None = None
    reject_border_touching: bool = False
    roi: tuple[int, int, int, int] | None = None
    reject_bad_image_quality: bool = True

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("version must be non-empty")
        if self.segmentation_mode not in {"fixed", "otsu", "adaptive"}:
            raise ValueError("segmentation_mode must be fixed, otsu, or adaptive")
        if not 0 <= self.threshold <= 255:
            raise ValueError("threshold must be in [0,255]")
        if self.segmentation_polarity not in {"positive", "negative"}:
            raise ValueError("segmentation_polarity must be positive or negative")
        if self.min_area_px <= 0 or self.max_area_px < self.min_area_px:
            raise ValueError("invalid defect area bounds")
        if self.max_defects < 0 or self.morphology_iterations < 1:
            raise ValueError("max_defects must be >=0 and morphology_iterations >=1")
        if not 0 <= self.max_defect_fraction <= 1:
            raise ValueError("max_defect_fraction must be in [0,1]")
        if self.min_circularity is not None and not 0 <= self.min_circularity <= 1:
            raise ValueError("min_circularity must be in [0,1]")
        if self.min_solidity is not None and not 0 <= self.min_solidity <= 1:
            raise ValueError("min_solidity must be in [0,1]")
        if self.max_aspect_ratio is not None and self.max_aspect_ratio < 1:
            raise ValueError("max_aspect_ratio must be >=1")
        if self.roi is not None:
            x, y, width, height = self.roi
            if min(x, y) < 0 or width <= 0 or height <= 0:
                raise ValueError("roi origin must be non-negative and size positive")

    @property
    def recipe_sha256(self) -> str:
        """Stable digest of recipe configuration used for audit correlation."""
        payload = repr(self).encode("utf-8")
        return sha256(payload).hexdigest()


@dataclass(frozen=True)
class Defect:
    """Accepted defect candidate plus all measured geometry."""

    measurement: RegionMeasurement
    rule_reasons: tuple[str, ...] = ()

    @property
    def area(self) -> float:
        return self.measurement.area_px

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.measurement.bbox_px


@dataclass(frozen=True)
class InspectionResult:
    """Immutable inspection decision and traceability evidence."""

    passed: bool
    quality: ImageQuality
    defect_count: int
    defect_fraction: float
    defects: tuple[Defect, ...]
    measurements: tuple[RegionMeasurement, ...]
    image_shape: tuple[int, int]
    roi: tuple[int, int, int, int] | None
    recipe_version: str
    recipe_sha256: str
    image_sha256: str
    processing_ms: float
    reject_reasons: tuple[str, ...]


def _touches_border(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    x, y, width, height = bbox
    return x <= 0 or y <= 0 or x + width >= shape[1] or y + height >= shape[0]


def _shift_measurement(measurement: RegionMeasurement, offset: tuple[int, int]) -> RegionMeasurement:
    ox, oy = offset
    x, y, width, height = measurement.bbox_px
    return RegionMeasurement(
        label=measurement.label,
        area_px=measurement.area_px,
        perimeter_px=measurement.perimeter_px,
        centroid_px=(measurement.centroid_px[0] + ox, measurement.centroid_px[1] + oy),
        bbox_px=(x + ox, y + oy, width, height),
        width_px=measurement.width_px,
        height_px=measurement.height_px,
        aspect_ratio=measurement.aspect_ratio,
        circularity=measurement.circularity,
        extent=measurement.extent,
        solidity=measurement.solidity,
        equivalent_diameter_px=measurement.equivalent_diameter_px,
        min_rect_width_px=measurement.min_rect_width_px,
        min_rect_height_px=measurement.min_rect_height_px,
        min_rect_angle_deg=measurement.min_rect_angle_deg,
        area_physical=measurement.area_physical,
        perimeter_physical=measurement.perimeter_physical,
        equivalent_diameter_physical=measurement.equivalent_diameter_physical,
    )


def inspect_array(
    image: np.ndarray,
    config: InspectionConfig | None = None,
    *,
    pixel_scale: PixelScale | None = None,
) -> InspectionResult:
    """Inspect an image array and return a deterministic decision with evidence."""
    started = perf_counter()
    config = config or InspectionConfig()
    values = np.asarray(image)
    if values.size == 0:
        raise ValueError("image must not be empty")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("image pixels must be numeric")
    image_bytes = values.tobytes(order="C")
    image_digest = sha256(image_bytes).hexdigest()
    quality = assess_image_quality(values, config.quality)
    gray = to_grayscale(values)
    inspected = gray
    roi_offset = (0, 0)
    if config.roi is not None:
        x, y, width, height = config.roi
        if x + width > gray.shape[1] or y + height > gray.shape[0]:
            raise ValueError("roi extends beyond image dimensions")
        inspected = gray[y:y + height, x:x + width]
        roi_offset = (x, y)

    corrected = normalize_illumination(inspected, config.preprocess)
    mask = threshold_image(
        corrected,
        mode=config.segmentation_mode,
        threshold=config.threshold,
        adaptive_block=config.adaptive_block,
        adaptive_c=config.adaptive_c,
        polarity=config.segmentation_polarity,
    )
    mask = apply_morphology(
        mask,
        opening=config.opening,
        closing=config.closing,
        operation_iterations=config.morphology_iterations,
    )
    local_measurements = measure_regions(mask, pixel_scale)

    measurements: list[RegionMeasurement] = []
    defects: list[Defect] = []
    candidates_rejected = 0
    for local in local_measurements:
        measurement = _shift_measurement(local, roi_offset)
        measurements.append(measurement)
        reasons: list[str] = []
        if not config.min_area_px <= measurement.area_px <= config.max_area_px:
            reasons.append("area_out_of_range")
        if config.reject_border_touching and _touches_border(local.bbox_px, inspected.shape):
            reasons.append("border_touching")
        if config.min_circularity is not None and measurement.circularity < config.min_circularity:
            reasons.append("circularity_below_minimum")
        if config.max_aspect_ratio is not None and measurement.aspect_ratio > config.max_aspect_ratio:
            reasons.append("aspect_ratio_above_maximum")
        if config.min_solidity is not None and measurement.solidity < config.min_solidity:
            reasons.append("solidity_below_minimum")
        if reasons:
            candidates_rejected += 1
            continue
        defects.append(Defect(measurement, ()))

    inspected_area = float(inspected.shape[0] * inspected.shape[1])
    defect_fraction = sum(item.area for item in defects) / inspected_area if inspected_area else 0.0
    reject_reasons: list[str] = list(quality.failures)
    if config.reject_bad_image_quality and not quality.passed:
        reject_reasons.append("image_quality_gate")
    if len(defects) > config.max_defects:
        reject_reasons.append("max_defect_count_exceeded")
    if defect_fraction > config.max_defect_fraction:
        reject_reasons.append("defect_fraction_exceeded")
    if candidates_rejected:
        # This is diagnostic only; rejected candidates are not defects and do not
        # automatically fail the part unless a rule above is violated.
        reject_reasons.append(f"candidate_regions_filtered:{candidates_rejected}")
    passed = not reject_reasons
    return InspectionResult(
        passed=passed,
        quality=quality,
        defect_count=len(defects),
        defect_fraction=float(defect_fraction),
        defects=tuple(defects),
        measurements=tuple(measurements),
        image_shape=(int(gray.shape[0]), int(gray.shape[1])),
        roi=config.roi,
        recipe_version=config.version,
        recipe_sha256=config.recipe_sha256,
        image_sha256=image_digest,
        processing_ms=(perf_counter() - started) * 1000.0,
        reject_reasons=tuple(reject_reasons),
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
