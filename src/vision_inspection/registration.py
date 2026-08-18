"""Image registration and perspective-rectification utilities."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RegistrationResult:
    """Result of ECC-based alignment to a reference image."""

    image: np.ndarray
    warp_matrix: np.ndarray
    correlation: float
    iterations: int


def _validate_pair(reference: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference)
    image = np.asarray(image)
    if reference.ndim not in {2, 3} or image.ndim not in {2, 3}:
        raise ValueError("reference and image must be 2D or 3D arrays")
    if reference.shape[:2] != image.shape[:2]:
        raise ValueError("reference and image must have matching spatial dimensions")
    ref_gray = reference if reference.ndim == 2 else cv2.cvtColor(reference.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    img_gray = image if image.ndim == 2 else cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return ref_gray.astype(np.uint8), img_gray.astype(np.uint8)


def register_ecc(
    reference: np.ndarray,
    image: np.ndarray,
    *,
    motion: str = "affine",
    iterations: int = 200,
    epsilon: float = 1e-6,
) -> RegistrationResult:
    """Align ``image`` to ``reference`` using OpenCV's ECC optimizer."""
    if motion not in {"translation", "euclidean", "affine", "homography"}:
        raise ValueError("motion must be translation, euclidean, affine, or homography")
    if iterations <= 0 or epsilon <= 0:
        raise ValueError("iterations and epsilon must be positive")
    ref_gray, img_gray = _validate_pair(reference, image)
    modes = {
        "translation": (cv2.MOTION_TRANSLATION, np.eye(2, 3, dtype=np.float32)),
        "euclidean": (cv2.MOTION_EUCLIDEAN, np.eye(2, 3, dtype=np.float32)),
        "affine": (cv2.MOTION_AFFINE, np.eye(2, 3, dtype=np.float32)),
        "homography": (cv2.MOTION_HOMOGRAPHY, np.eye(3, 3, dtype=np.float32)),
    }
    mode, warp = modes[motion]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, epsilon)
    correlation, warp = cv2.findTransformECC(ref_gray, img_gray, warp, mode, criteria)
    flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    size = (reference.shape[1], reference.shape[0])
    if motion == "homography":
        aligned = cv2.warpPerspective(image, warp, size, flags=flags)
    else:
        aligned = cv2.warpAffine(image, warp, size, flags=flags)
    return RegistrationResult(aligned, warp, float(correlation), iterations)


def rectify_perspective(image: np.ndarray, source_points: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Perspective-warp four ordered source corners to a rectangular target."""
    image = np.asarray(image)
    points = np.asarray(source_points, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("source_points must have shape (4,2)")
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("target size must be positive")
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(points, target)
    return cv2.warpPerspective(image, H, (width, height))
