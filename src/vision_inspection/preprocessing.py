"""Explainable preprocessing and segmentation primitives.

OpenCV documents morphology as a useful way to suppress or emphasize spatial
structures. This module keeps those transformations explicit so recipe changes
are easy to audit and benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    """Controls grayscale conversion, denoising, contrast, and illumination."""

    blur_kernel: int = 5
    background_kernel: int = 51
    polarity: str = "dark"
    denoise: str = "gaussian"
    clahe_clip_limit: float | None = None
    clahe_grid: int = 8

    def __post_init__(self) -> None:
        if self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd number")
        if self.background_kernel < 3 or self.background_kernel % 2 == 0:
            raise ValueError("background_kernel must be an odd number >=3")
        if self.polarity not in {"dark", "light"}:
            raise ValueError("polarity must be 'dark' or 'light'")
        if self.denoise not in {"none", "gaussian", "median"}:
            raise ValueError("denoise must be none, gaussian, or median")
        if self.clahe_clip_limit is not None and self.clahe_clip_limit <= 0:
            raise ValueError("clahe_clip_limit must be positive")
        if self.clahe_grid < 2:
            raise ValueError("clahe_grid must be >=2")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert grayscale/BGR/BGRA numeric data to a bounded uint8 image."""
    values = np.asarray(image)
    if values.size == 0:
        raise ValueError("image must not be empty")
    if values.ndim == 2:
        gray = values
    elif values.ndim == 3 and values.shape[2] in {3, 4}:
        code = cv2.COLOR_BGR2GRAY if values.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        gray = cv2.cvtColor(values.astype(np.uint8), code)
    else:
        raise ValueError("image must have shape (H,W), (H,W,3), or (H,W,4)")
    if not np.issubdtype(gray.dtype, np.number):
        raise ValueError("image pixels must be numeric")
    return np.clip(gray, 0, 255).astype(np.uint8)


def normalize_illumination(gray: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Enhance localized bright/dark structures relative to a slow background."""
    image = to_grayscale(gray)
    if config.denoise == "gaussian":
        filtered = cv2.GaussianBlur(image, (config.blur_kernel, config.blur_kernel), 0)
    elif config.denoise == "median":
        filtered = cv2.medianBlur(image, config.blur_kernel)
    else:
        filtered = image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.background_kernel, config.background_kernel))
    if config.polarity == "dark":
        corrected = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, kernel)
    else:
        corrected = cv2.morphologyEx(filtered, cv2.MORPH_TOPHAT, kernel)
    if config.clahe_clip_limit is not None:
        clahe = cv2.createCLAHE(config.clahe_clip_limit, (config.clahe_grid, config.clahe_grid))
        corrected = clahe.apply(corrected)
    return corrected


def threshold_image(
    image: np.ndarray,
    *,
    mode: str = "otsu",
    threshold: int = 80,
    adaptive_block: int = 31,
    adaptive_c: float = 3.0,
    polarity: str = "positive",
) -> np.ndarray:
    """Segment an 8-bit image using fixed, Otsu, or adaptive thresholding.

    ``polarity='positive'`` extracts high response; ``negative`` extracts low
    response. Separating segmentation polarity from image polarity makes recipe
    semantics explicit.
    """
    image = to_grayscale(image)
    if mode == "fixed":
        if not 0 <= threshold <= 255:
            raise ValueError("threshold must be in [0,255]")
        threshold_type = cv2.THRESH_BINARY if polarity == "positive" else cv2.THRESH_BINARY_INV
        _, mask = cv2.threshold(image, threshold, 255, threshold_type)
    elif mode == "otsu":
        threshold_type = cv2.THRESH_BINARY if polarity == "positive" else cv2.THRESH_BINARY_INV
        _, mask = cv2.threshold(image, 0, 255, threshold_type | cv2.THRESH_OTSU)
    elif mode == "adaptive":
        if adaptive_block < 3 or adaptive_block % 2 == 0:
            raise ValueError("adaptive_block must be odd and >=3")
        threshold_type = cv2.THRESH_BINARY if polarity == "positive" else cv2.THRESH_BINARY_INV
        mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type, adaptive_block, adaptive_c)
    else:
        raise ValueError("mode must be fixed, otsu, or adaptive")
    if polarity not in {"positive", "negative"}:
        raise ValueError("polarity must be positive or negative")
    return mask


def apply_morphology(mask: np.ndarray, *, opening: int = 0, closing: int = 0, operation_iterations: int = 1) -> np.ndarray:
    """Apply deterministic binary opening/closing cleanup."""
    values = np.asarray(mask)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("mask must be a non-empty 2D array")
    if operation_iterations < 1:
        raise ValueError("operation_iterations must be >=1")
    result = np.where(values > 0, 255, 0).astype(np.uint8)
    for size, operation in ((opening, cv2.MORPH_OPEN), (closing, cv2.MORPH_CLOSE)):
        if size:
            if size < 1 or size % 2 == 0:
                raise ValueError("morphology kernel sizes must be positive odd numbers")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            result = cv2.morphologyEx(result, operation, kernel, iterations=operation_iterations)
    return result
