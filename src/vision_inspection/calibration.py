"""Camera calibration and pixel-to-physical metrology utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole camera calibration result with distortion coefficients."""

    camera_matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    rms_error: float

    def __post_init__(self) -> None:
        K = np.asarray(self.camera_matrix, dtype=float)
        d = np.asarray(self.distortion, dtype=float)
        if K.shape != (3, 3):
            raise ValueError("camera_matrix must be 3x3")
        if d.size < 4:
            raise ValueError("distortion must contain at least four coefficients")
        w, h = self.image_size
        if w <= 0 or h <= 0 or self.rms_error < 0 or not np.isfinite(self.rms_error):
            raise ValueError("invalid calibration metadata")
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(d)):
            raise ValueError("calibration parameters must be finite")
        object.__setattr__(self, "camera_matrix", K.copy())
        object.__setattr__(self, "distortion", d.reshape(-1, 1).copy())

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Return an undistorted image using the stored calibration."""
        image = np.asarray(image)
        if image.ndim not in {2, 3}:
            raise ValueError("image must be 2D or 3D")
        h, w = image.shape[:2]
        if (w, h) != self.image_size:
            raise ValueError(f"image size {(w, h)} does not match calibration {self.image_size}")
        return cv2.undistort(image, self.camera_matrix, self.distortion)

    def save(self, path: str | Path) -> None:
        """Persist calibration as a portable NumPy archive."""
        np.savez_compressed(
            path,
            camera_matrix=self.camera_matrix,
            distortion=self.distortion,
            image_size=np.asarray(self.image_size, dtype=np.int64),
            rms_error=np.asarray(self.rms_error),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CameraCalibration":
        """Load a previously saved calibration."""
        with np.load(path) as data:
            size = tuple(int(v) for v in data["image_size"].tolist())
            return cls(data["camera_matrix"], data["distortion"], size, float(data["rms_error"]))


@dataclass(frozen=True)
class CharucoCalibration:
    """ChArUco calibration result with intrinsic uncertainty diagnostics."""

    calibration: CameraCalibration
    intrinsic_std: np.ndarray
    per_view_errors: np.ndarray

    def __post_init__(self) -> None:
        intrinsic = np.asarray(self.intrinsic_std, dtype=float).reshape(-1)
        per_view = np.asarray(self.per_view_errors, dtype=float).reshape(-1)
        if not np.all(np.isfinite(intrinsic)) or not np.all(np.isfinite(per_view)):
            raise ValueError("calibration uncertainty values must be finite")
        object.__setattr__(self, "intrinsic_std", intrinsic.copy())
        object.__setattr__(self, "per_view_errors", per_view.copy())


def calibrate_camera(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> CameraCalibration:
    """Estimate intrinsic camera parameters from calibration correspondences."""
    if not object_points or len(object_points) != len(image_points):
        raise ValueError("object_points and image_points must contain the same non-zero number of views")
    w, h = image_size
    if w <= 0 or h <= 0:
        raise ValueError("image_size must be positive")
    obj = [np.asarray(v, dtype=np.float32) for v in object_points]
    img = [np.asarray(v, dtype=np.float32) for v in image_points]
    if any(v.ndim != 2 or v.shape[1] != 3 for v in obj):
        raise ValueError("each object-point view must have shape (N,3)")
    if any(v.ndim != 2 or v.shape[1] != 2 for v in img):
        raise ValueError("each image-point view must have shape (N,2)")
    if any(len(o) != len(i) or len(o) < 4 for o, i in zip(obj, img)):
        raise ValueError("every calibration view needs matching points and at least four correspondences")

    rms, K, distortion, _, _ = cv2.calibrateCamera(obj, img, (w, h), None, None)
    return CameraCalibration(K, distortion, (w, h), float(rms))


def calibrate_charuco(
    charuco_corners: list[np.ndarray],
    charuco_ids: list[np.ndarray],
    board,
    image_size: tuple[int, int],
) -> CharucoCalibration:
    """Calibrate camera from ChArUco corners collected across multiple views.

    Requires an OpenCV build exposing ``cv2.aruco`` (the contrib package).
    ``board`` is an OpenCV CharucoBoard object created with the desired board
    geometry and marker dictionary.
    """
    aruco = getattr(cv2, "aruco", None)
    if aruco is None or not hasattr(aruco, "calibrateCameraCharucoExtended"):
        raise RuntimeError("ChArUco calibration requires an OpenCV build with the aruco module")
    if len(charuco_corners) != len(charuco_ids) or not charuco_corners:
        raise ValueError("charuco_corners and charuco_ids must contain matching non-empty views")
    w, h = image_size
    if w <= 0 or h <= 0:
        raise ValueError("image_size must be positive")

    corners = [np.asarray(v, dtype=np.float32) for v in charuco_corners]
    ids = [np.asarray(v, dtype=np.int32) for v in charuco_ids]
    for c, i in zip(corners, ids):
        if c.ndim not in {2, 3} or c.shape[-1] != 2 or len(c) != len(i) or len(c) < 4:
            raise ValueError("each ChArUco view needs matching corner/id arrays and >=4 corners")

    result = aruco.calibrateCameraCharucoExtended(corners, ids, board, (w, h), None, None)
    rms, K, distortion, _, _, intrinsic_std, _, per_view_errors = result
    calibration = CameraCalibration(K, distortion, (w, h), float(rms))
    return CharucoCalibration(calibration, intrinsic_std, per_view_errors)


@dataclass(frozen=True)
class PixelScale:
    """A local isotropic pixel-to-physical scale for planar metrology."""

    units_per_pixel: float
    units: str = "mm"

    def __post_init__(self) -> None:
        if not np.isfinite(self.units_per_pixel) or self.units_per_pixel <= 0:
            raise ValueError("units_per_pixel must be positive and finite")
        if not self.units:
            raise ValueError("units must be non-empty")

    def length(self, pixels: float) -> float:
        """Convert a pixel length to physical units."""
        if not np.isfinite(pixels) or pixels < 0:
            raise ValueError("pixels must be non-negative and finite")
        return float(pixels * self.units_per_pixel)

    def area(self, pixels_squared: float) -> float:
        """Convert a pixel area to square physical units."""
        if not np.isfinite(pixels_squared) or pixels_squared < 0:
            raise ValueError("pixel area must be non-negative and finite")
        return float(pixels_squared * self.units_per_pixel**2)
