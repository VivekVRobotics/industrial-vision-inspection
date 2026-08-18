"""Pose normalization through ECC registration and perspective rectification.

OpenCV's ECC optimizer estimates geometric transforms by maximizing image
correlation. It is useful for small pose variation after acquisition, but it is
not a substitute for a physical fixture or a validated fiducial strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RegistrationResult:
    """Registered image, transform, score, and iteration diagnostics."""

    image: np.ndarray
    warp_matrix: np.ndarray
    correlation: float
    iterations: int
    motion: str

    @property
    def accepted(self) -> bool:
        return bool(np.isfinite(self.correlation))


def _gray(values: np.ndarray) -> np.ndarray:
    image = np.asarray(values)
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("images must be grayscale or BGR")
    if gray.size == 0:
        raise ValueError("image must not be empty")
    return np.clip(gray, 0, 255).astype(np.uint8)


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
        raise ValueError("iterations and epsilon must be positive; pyramid_levels >=1")
    ref = _gray(reference)
    current = _gray(image)
    if ref.shape != current.shape:
        raise ValueError("reference and image must have matching spatial dimensions")
    modes = {
        "translation": (cv2.MOTION_TRANSLATION, np.eye(2, 3, dtype=np.float32)),
        "euclidean": (cv2.MOTION_EUCLIDEAN, np.eye(2, 3, dtype=np.float32)),
        "affine": (cv2.MOTION_AFFINE, np.eye(2, 3, dtype=np.float32)),
        "homography": (cv2.MOTION_HOMOGRAPHY, np.eye(3, dtype=3, dtype=np.float32)),
    }
    mode, warp = modes[motion]
    if pyramid_levels == 1:
        reference_pyramid = [ref]
        image_pyramid = [current]
    else:
        reference_pyramid = [ref]
        image_pyramid = [current]
        for _ in range(1, pyramid_levels):
            reference_pyramid.insert(0, cv2.pyrDown(reference_pyramid[0]))
            image_pyramid.insert(0, cv2.pyrDown(image_pyramid[0]))
    correlation = float("nan")
    for level, (reference_level, image_level) in enumerate(zip(reference_pyramid, image_pyramid)):
        scale = 2 ** (pyramid_levels - 1 - level)
        if level > 0:
            if motion == "homography":
                warp[:2, 2] *= 2.0
                warp[:2, :2] = warp[:2, :2]
            else:
                warp[:, 2] *= 2.0
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, epsilon)
        correlation, warp = cv2.findTransformECC(reference_level, image_level, warp, mode, criteria)
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
