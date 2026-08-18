"""Image normalization and segmentation primitives for industrial inspection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    """Controls grayscale conversion, denoising, and illumination correction."""

    blur_kernel: int = 5
    background_kernel: int = 51
    polarity: str = "dark"

    def __post_init__(self) -> None:
        if self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd number")
        if self.background_kernel < 3 or self.background_kernel % 2 == 0:
            raise ValueError("background_kernel must be an odd number >= 3")
        if self.polarity not in {"dark", "light"}:
            raise ValueError("polarity must be 'dark' or 'light'")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a uint8 grayscale/BGR image to uint8 grayscale."""
    image = np.asarray(image)
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] in {3, 4}:
        code = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        gray = cv2.cvtColor(image, code)
    else:
        raise ValueError("image must have shape (H,W), (H,W,3), or (H,W,4)")
    if gray.dtype != np.uint8:
        if not np.issubdtype(gray.dtype, np.number):
            raise ValueError("image must contain numeric pixel data")
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def normalize_illumination(gray: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Flatten slow illumination variation while preserving localized defects."""
    gray = to_grayscale(gray)
    blurred = cv2.GaussianBlur(gray, (config.blur_kernel, config.blur_kernel), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.background_kernel, config.background_kernel))
    if config.polarity == "dark":
        corrected = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
    else:
        corrected = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    return corrected


def threshold_image(
    image: np.ndarray,
    *,
    mode: str = "otsu",
    threshold: int = 80,
    adaptive_block: int = 31,
    adaptive_c: float = 3.0,
) -> np.ndarray:
    """Segment an 8-bit image with fixed, Otsu, or adaptive thresholding."""
    image = to_grayscale(image)
    if mode == "fixed":
        _, mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    elif mode == "otsu":
        _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "adaptive":
        if adaptive_block < 3 or adaptive_block % 2 == 0:
            raise ValueError("adaptive_block must be an odd number >= 3")
        mask = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            adaptive_block,
            adaptive_c,
        )
    else:
        raise ValueError("mode must be 'fixed', 'otsu', or 'adaptive'")
    return mask


def apply_morphology(mask: np.ndarray, *, opening: int = 0, closing: int = 0) -> np.ndarray:
    """Apply optional opening/closing operations to a binary mask."""
    result = np.asarray(mask, dtype=np.uint8)
    for size, op in ((opening, cv2.MORPH_OPEN), (closing, cv2.MORPH_CLOSE)):
        if size:
            if size < 1 or size % 2 == 0:
                raise ValueError("morphology kernel sizes must be positive odd numbers")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            result = cv2.morphologyEx(result, op, kernel)
    return result
