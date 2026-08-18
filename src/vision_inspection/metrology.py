"""Contour-based region metrology for industrial inspection.

Measurements are kept in pixel coordinates and optionally converted to physical
units through a supplied local scale. The implementation reports multiple
shape descriptors so inspection recipes can distinguish compact defects,
stringers, scratches, blobs, and irregular regions without hiding the decision
inside a learned model.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import PixelScale


@dataclass(frozen=True)
class RegionMeasurement:
    """Geometry and shape descriptors for one extracted region."""

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
    equivalent_diameter_px: float
    min_rect_width_px: float
    min_rect_height_px: float
    min_rect_angle_deg: float
    area_physical: float | None = None
    perimeter_physical: float | None = None
    equivalent_diameter_physical: float | None = None

    @property
    def compactness(self) -> float:
        """Return perimeter-normalized compactness; lower is more compact."""
        return (self.perimeter_px**2 / self.area_px) if self.area_px > 0 else float("inf")


def measure_regions(mask: np.ndarray, scale: PixelScale | None = None) -> tuple[RegionMeasurement, ...]:
    """Extract deterministic external-contour measurements from a binary mask."""
    values = np.asarray(mask)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("mask must be a non-empty 2D array")
    binary = np.where(values > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results: list[RegionMeasurement] = []
    for label, contour in enumerate(contours, start=1):
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area <= 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        cx = moments["m10"] / moments["m00"] if abs(moments["m00"]) > 1e-12 else x + w / 2.0
        cy = moments["m01"] / moments["m00"] if abs(moments["m00"]) > 1e-12 else y + h / 2.0
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        circularity = min(1.0, 4.0 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
        extent = area / float(w * h) if w > 0 and h > 0 else 0.0
        solidity = area / hull_area if hull_area > 0 else 0.0
        equivalent = float(np.sqrt(4.0 * area / np.pi))
        rect = cv2.minAreaRect(contour)
        rect_width, rect_height = sorted((float(rect[1][0]), float(rect[1][1])), reverse=True)
        angle = float(rect[2])
        values_to_check = (area, perimeter, cx, cy, circularity, extent, solidity, equivalent, rect_width, rect_height, angle)
        if not all(np.isfinite(v) for v in values_to_check):
            raise ValueError("non-finite region measurement")
        results.append(
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
                equivalent_diameter_px=equivalent,
                min_rect_width_px=rect_width,
                min_rect_height_px=rect_height,
                min_rect_angle_deg=angle,
                area_physical=scale.area(area) if scale else None,
                perimeter_physical=scale.length(perimeter) if scale else None,
                equivalent_diameter_physical=scale.length(equivalent) if scale else None,
            )
        )
    return tuple(results)
