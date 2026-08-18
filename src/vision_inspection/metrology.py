"""Geometric measurements extracted from detected industrial regions."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import PixelScale


@dataclass(frozen=True)
class RegionMeasurement:
    """Shape and location measurements for one connected defect region."""

    label: int
    area_px: float
    perimeter_px: float
    centroid_px: tuple[float, float]
    bbox_px: tuple[int, int, int, int]
    width_px: float
    height_px: float
    aspect_ratio: float
    circularity: float
    extent: float
    solidity: float
    area_physical: float | None = None
    perimeter_physical: float | None = None


def measure_regions(mask: np.ndarray, scale: PixelScale | None = None) -> tuple[RegionMeasurement, ...]:
    """Extract contour-based geometry from a binary mask."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    measurements: list[RegionMeasurement] = []
    for label, contour in enumerate(contours, start=1):
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if abs(moments["m00"]) > 1e-12:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx = x + w / 2.0
            cy = y + h / 2.0
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
        extent = area / float(w * h) if w > 0 and h > 0 else 0.0
        solidity = area / hull_area if hull_area > 0 else 0.0
        values = (area, perimeter, cx, cy, w, h, circularity, extent, solidity)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("non-finite region measurement")
        area_physical = scale.area(area) if scale is not None else None
        perimeter_physical = scale.length(perimeter) if scale is not None else None
        measurements.append(
            RegionMeasurement(
                label=label,
                area_px=area,
                perimeter_px=perimeter,
                centroid_px=(float(cx), float(cy)),
                bbox_px=(int(x), int(y), int(w), int(h)),
                width_px=float(w),
                height_px=float(h),
                aspect_ratio=float(max(w, h) / max(1, min(w, h))),
                circularity=float(circularity),
                extent=float(extent),
                solidity=float(solidity),
                area_physical=area_physical,
                perimeter_physical=perimeter_physical,
            )
        )
    return tuple(measurements)
