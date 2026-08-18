"""Operator-review and debugging visualizations.

Visualization is intentionally downstream of the immutable inspection result:
rendering must never change the inspection decision. Annotated images are
therefore evidence artifacts for review, troubleshooting, and dataset curation.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .inspector import InspectionResult


def _as_bgr(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.size == 0:
        raise ValueError("image must not be empty")
    if values.ndim == 2:
        return cv2.cvtColor(np.clip(values, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if values.ndim == 3 and values.shape[2] == 3:
        return np.clip(values, 0, 255).astype(np.uint8).copy()
    raise ValueError("image must be grayscale or BGR")


def annotate_result(image: np.ndarray, result: InspectionResult) -> np.ndarray:
    """Render PASS/FAIL, quality, traceability, and accepted defect geometry."""
    canvas = _as_bgr(image)
    for defect in result.defects:
        x, y, width, height = defect.bbox
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (0, 0, 255), 2)
        cx, cy = defect.measurement.centroid_px
        cv2.circle(canvas, (round(cx), round(cy)), 3, (0, 0, 255), -1)
    status = "PASS" if result.passed else "FAIL"
    status_color = (0, 150, 0) if result.passed else (0, 0, 255)
    quality = "OK" if result.quality.passed else ",".join(result.quality.failures)
    lines = [
        f"{status} | defects={result.defect_count} | quality={quality}",
        f"recipe={result.recipe_version} | {result.processing_ms:.2f} ms",
        f"image={result.image_sha256[:12]} | recipe_sha={result.recipe_sha256[:12]}",
    ]
    for index, text in enumerate(lines):
        cv2.putText(canvas, text, (12, 28 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
    return canvas


def save_annotated(image: np.ndarray, result: InspectionResult, path: str | Path) -> None:
    """Write an annotated evidence image and fail loudly on write errors."""
    output = annotate_result(image, result)
    if not cv2.imwrite(str(path), output):
        raise OSError(f"could not write annotated image: {path}")
