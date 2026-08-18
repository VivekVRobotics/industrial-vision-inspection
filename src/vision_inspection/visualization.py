"""Inspection visualization for debugging and operator review."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .inspector import InspectionResult


def annotate_result(image: np.ndarray, result: InspectionResult) -> np.ndarray:
    """Draw accepted defect boxes and acquisition status onto a copy of an image."""
    image = np.asarray(image)
    if image.ndim == 2:
        canvas = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        canvas = image.astype(np.uint8).copy()
    else:
        raise ValueError("image must be grayscale or BGR")

    for defect in result.defects:
        x, y, w, h = defect.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx, cy = defect.measurement.centroid_px
        cv2.circle(canvas, (round(cx), round(cy)), 3, (0, 0, 255), -1)

    status = "PASS" if result.passed else "FAIL"
    quality = "Q:OK" if result.quality.passed else "Q:" + ",".join(result.quality.failures)
    cv2.putText(canvas, f"{status} | defects={result.defect_count} | {quality}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if not result.passed else (0, 150, 0), 2)
    return canvas


def save_annotated(image: np.ndarray, result: InspectionResult, path: str | Path) -> None:
    """Write an annotated inspection image and fail loudly on write errors."""
    output = annotate_result(image, result)
    if not cv2.imwrite(str(path), output):
        raise OSError(f"could not write annotated image: {path}")
