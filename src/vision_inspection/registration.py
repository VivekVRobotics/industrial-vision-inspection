"""Pose normalization through ECC registration and perspective rectification.

ECC (Enhanced Correlation Coefficient) alignment is useful for small, smooth
pose variation after image acquisition. It should be bounded by a correlation
quality gate and validated on representative production fixtures; it is not a
replacement for mechanical fixturing or fiducials.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RegistrationResult:
    """Registered image, transform, score, motion model, and iteration budget."""

    image: np.ndarray
    warp_matrix: np.ndarray
    correlation: float
    iterations: int
    motion: str

    @property
    def accepted(self) -> bool:
        """Return whether the optimizer produced a finite similarity score."""
        return bool(np.isfinite(self.correlation))


def _gray(values: np.ndarray) -> np.ndarray:
    image = np.asarray(values)
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("images must be grayscale or BGR")
    return np.clip(gray, 0, 255).astype(np.uint8)


def _scale_warp(warp: np.ndarray, motion: str) -> np.ndarray:
    """Scale translation terms when moving an affine/homography warp upward in a pyramid."""
    scaled = warp.copy()
    if motion == "homography":
        scaled[0:2, 2] *= 2.0
    else:
        scaled[:, 2] *= 2.0
    return scaled


def register_ecc(
    reference: np.ndarray,
    image: np.ndarray,
    *,
    motion: str = "affine",
    iterations: int = 200,
    epsilon: float = 1e-6,
    pyramid_levels: int = 1,
    min_correlation: float | None = None,
) -> RegistrationResult:
    """Align ``image`` to ``reference`` using optional multiscale ECC."""
    if motion not in {"translation", "euclidean", "affine", "homography"}:
        raise ValueError("invalid motion model")
    if iterations <= 0 or epsilon <= 0 or pyramid_levels < 1:
        raise ValueError("iterations/epsilon must be positive and pyramid_levels >=1")
    if min_correlation is not None and not -1.0 <= min_correlation <= 1.0:
        raise ValueError("min_correlation must be in [-1,1]")
    ref = _gray(reference)
    current = _gray(image)
    if ref.shape != current.shape:
        raise ValueError("reference and image must have matching spatial dimensions")
    modes = {
        "translation": (cv2.MOTION_TRANSLATION, np.eye(2, 3, dtype=np.float32)),
        "euclidean": (cv2.MOTION_EUCLIDEAN, np.eye(2, 3, dtype=np.float32)),
        "affine": (cv2.MOTION_AFFINE, np.eye(2, 3, dtype=np.float32)),
        "homography": (cv2.MOTION_HOMOGRAPHY, np.eye(3, 3, dtype=np.float32)),
    }
    mode, warp = modes[motion]
    reference_pyramid = [ref]
    image_pyramid = [current]
    for _ in range(1, pyramid_levels):
        reference_pyramid.insert(0, cv2.pyrDown(reference_pyramid[0]))
        image_pyramid.insert(0, cv2.pyrDown(image_pyramid[0]))

    correlation = float("nan")
    for level, (reference_level, image_level) in enumerate(zip(reference_pyramid, image_pyramid)):
        if level > 0:
            warp = _scale_warp(warp, motion)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, epsilon)
        try:
            correlation, warp = cv2.findTransformECC(reference_level, image_level, warp, mode, criteria)
        except cv2.error as exc:
            raise RuntimeError(f"ECC registration failed at pyramid level {level}") from exc
        if not np.isfinite(correlation):
            raise RuntimeError("ECC returned a non-finite correlation score")
    if min_correlation is not None and correlation < min_correlation:
        raise RuntimeError(f"registration correlation {correlation:.6f} below minimum {min_correlation:.6f}")

    flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    size = (ref.shape[1], ref.shape[0])
    aligned = cv2.warpPerspective(image, warp, size, flags=flags) if motion == "homography" else cv2.warpAffine(image, warp, size, flags=flags)
    return RegistrationResult(aligned, warp, float(correlation), iterations, motion)


def rectify_perspective(image: np.ndarray, source_points: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Warp four ordered source corners to a rectangular target plane."""
    values = np.asarray(image)
    points = np.asarray(source_points, dtype=np.float32)
    width, height = size
    if values.size == 0 or points.shape != (4, 2) or width <= 0 or height <= 0:
        raise ValueError("invalid image, source_points, or target size")
    if abs(cv2.contourArea(points.reshape(-1, 1, 2))) < 1e-6:
        raise ValueError("source_points are degenerate")
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, target)
    return cv2.warpPerspective(values, matrix, (width, height))
