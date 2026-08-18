"""Camera calibration and planar metrology foundations.

Calibration is part of the measurement chain, not an optional visualization
step. The module therefore stores image size, reprojection-quality diagnostics,
per-view errors, and intrinsic parameter uncertainty when the OpenCV API
provides them. ChArUco support is isolated so the core package can still expose
standard pinhole calibration even when an adapter uses another capture method.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole calibration plus reprojection diagnostics."""

    camera_matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    rms_error: float
    per_view_errors: np.ndarray | None = None
    intrinsic_std: np.ndarray | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.camera_matrix, dtype=float)
        distortion = np.asarray(self.distortion, dtype=float).reshape(-1, 1)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("camera_matrix must be a finite 3x3 matrix")
        if distortion.size < 4 or not np.all(np.isfinite(distortion)):
            raise ValueError("distortion must contain at least four finite coefficients")
        width, height = self.image_size
        if width <= 0 or height <= 0 or not np.isfinite(self.rms_error) or self.rms_error < 0:
            raise ValueError("invalid calibration metadata")
        if not np.isclose(matrix[2, 2], 1.0, atol=1e-6):
            raise ValueError("camera_matrix[2,2] must be normalized to 1")
        object.__setattr__(self, "camera_matrix", matrix.copy())
        object.__setattr__(self, "distortion", distortion.copy())
        if self.per_view_errors is not None:
            values = np.asarray(self.per_view_errors, dtype=float).reshape(-1)
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("per-view reprojection errors must be finite and non-negative")
            object.__setattr__(self, "per_view_errors", values.copy())
        if self.intrinsic_std is not None:
            values = np.asarray(self.intrinsic_std, dtype=float).reshape(-1)
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("intrinsic standard deviations must be finite and non-negative")
            object.__setattr__(self, "intrinsic_std", values.copy())

    @property
    def max_per_view_error(self) -> float | None:
        return float(np.max(self.per_view_errors)) if self.per_view_errors is not None and self.per_view_errors.size else None

    def assert_quality(self, *, max_rms: float | None = None, max_view_error: float | None = None) -> None:
        """Raise if this calibration exceeds caller-specified quality limits."""
        if max_rms is not None and (max_rms < 0 or self.rms_error > max_rms):
            raise ValueError(f"calibration RMS error {self.rms_error:.6g} exceeds limit {max_rms:.6g}")
        if max_view_error is not None:
            observed = self.max_per_view_error
            if observed is not None and observed > max_view_error:
                raise ValueError(f"maximum per-view error {observed:.6g} exceeds limit {max_view_error:.6g}")

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Undistort an image; spatial dimensions must match calibration."""
        image = np.asarray(image)
        if image.ndim not in {2, 3} or image.size == 0:
            raise ValueError("image must be a non-empty 2D or 3D array")
        height, width = image.shape[:2]
        if (width, height) != self.image_size:
            raise ValueError(f"image size {(width, height)} does not match calibration {self.image_size}")
        return cv2.undistort(image, self.camera_matrix, self.distortion)

    def save(self, path: str | Path) -> None:
        """Persist calibration and optional uncertainty diagnostics."""
        np.savez_compressed(
            path,
            camera_matrix=self.camera_matrix,
            distortion=self.distortion,
            image_size=np.asarray(self.image_size, dtype=np.int64),
            rms_error=np.asarray(self.rms_error),
            per_view_errors=np.asarray([] if self.per_view_errors is None else self.per_view_errors),
            intrinsic_std=np.asarray([] if self.intrinsic_std is None else self.intrinsic_std),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CameraCalibration":
        """Load a calibration archive written by :meth:`save`."""
        with np.load(path) as data:
            size = tuple(int(v) for v in data["image_size"].tolist())
            view = data["per_view_errors"].reshape(-1)
            intrinsic = data["intrinsic_std"].reshape(-1)
            return cls(
                data["camera_matrix"],
                data["distortion"],
                size,
                float(data["rms_error"]),
                None if view.size == 0 else view,
                None if intrinsic.size == 0 else intrinsic,
            )


@dataclass(frozen=True)
class CharucoCalibration:
    """ChArUco calibration and the diagnostic arrays returned by OpenCV."""

    calibration: CameraCalibration
    intrinsic_std: np.ndarray
    per_view_errors: np.ndarray

    def __post_init__(self) -> None:
        if self.intrinsic_std.size == 0 or self.per_view_errors.size == 0:
            raise ValueError("ChArUco calibration must contain diagnostics")


def _validate_correspondences(object_points: list[np.ndarray], image_points: list[np.ndarray], image_size: tuple[int, int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if not object_points or len(object_points) != len(image_points):
        raise ValueError("object_points and image_points must contain the same non-empty number of views")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    obj = [np.asarray(points, dtype=np.float32) for points in object_points]
    img = [np.asarray(points, dtype=np.float32) for points in image_points]
    for object_view, image_view in zip(obj, img):
        if object_view.ndim != 2 or object_view.shape[1] != 3:
            raise ValueError("each object-point view must have shape (N,3)")
        if image_view.ndim != 2 or image_view.shape[1] != 2:
            raise ValueError("each image-point view must have shape (N,2)")
        if len(object_view) != len(image_view) or len(object_view) < 4:
            raise ValueError("each calibration view needs matching points and at least four correspondences")
    return obj, img


def calibrate_camera(object_points: list[np.ndarray], image_points: list[np.ndarray], image_size: tuple[int, int]) -> CameraCalibration:
    """Estimate a pinhole camera model and retain per-view errors/std-dev diagnostics."""
    obj, img = _validate_correspondences(object_points, image_points, image_size)
    width, height = image_size
    rms, matrix, distortion, _, _, intrinsic_std, _, per_view_errors = cv2.calibrateCameraExtended(
        obj, img, (width, height), None, None
    )
    return CameraCalibration(matrix, distortion, image_size, float(rms), per_view_errors.reshape(-1), intrinsic_std.reshape(-1))


def calibrate_charuco(
    charuco_corners: list[np.ndarray],
    charuco_ids: list[np.ndarray],
    board,
    image_size: tuple[int, int],
) -> CharucoCalibration:
    """Calibrate from detected ChArUco corners across multiple views."""
    aruco = getattr(cv2, "aruco", None)
    if aruco is None or not hasattr(aruco, "calibrateCameraCharucoExtended"):
        raise RuntimeError("ChArUco calibration requires an OpenCV contrib build with cv2.aruco")
    if len(charuco_corners) != len(charuco_ids) or not charuco_corners:
        raise ValueError("charuco_corners and charuco_ids must have matching non-empty views")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    corners = [np.asarray(values, dtype=np.float32) for values in charuco_corners]
    ids = [np.asarray(values, dtype=np.int32) for values in charuco_ids]
    for corner_view, id_view in zip(corners, ids):
        if corner_view.ndim not in {2, 3} or corner_view.shape[-1] != 2 or len(corner_view) != len(id_view) or len(corner_view) < 4:
            raise ValueError("each ChArUco view needs >=4 matched corner/id observations")
    result = aruco.calibrateCameraCharucoExtended(corners, ids, board, (width, height), None, None)
    rms, matrix, distortion, _, _, intrinsic_std, _, per_view_errors = result
    calibration = CameraCalibration(matrix, distortion, image_size, float(rms), per_view_errors, intrinsic_std)
    return CharucoCalibration(calibration, calibration.intrinsic_std, calibration.per_view_errors)


@dataclass(frozen=True)
class PixelScale:
    """Local isotropic pixel-to-physical scale for a planar measurement plane."""

    units_per_pixel: float
    units: str = "mm"

    def __post_init__(self) -> None:
        if not self.units or not np.isfinite(self.units_per_pixel) or self.units_per_pixel <= 0:
            raise ValueError("units must be non-empty and units_per_pixel positive/finite")

    def length(self, pixels: float) -> float:
        """Convert a non-negative pixel distance to physical units."""
        if not np.isfinite(pixels) or pixels < 0:
            raise ValueError("pixels must be non-negative and finite")
        return float(pixels * self.units_per_pixel)

    def area(self, pixels_squared: float) -> float:
        """Convert a pixel area to square physical units."""
        if not np.isfinite(pixels_squared) or pixels_squared < 0:
            raise ValueError("pixel area must be non-negative and finite")
        return float(pixels_squared * self.units_per_pixel**2)
