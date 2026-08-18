"""Industrial machine-vision inspection toolkit."""

from .calibration import CameraCalibration, PixelScale, calibrate_camera
from .evaluation import BinaryConfusion, confusion_from_labels
from .inspector import Defect, InspectionConfig, InspectionResult, inspect_array, inspect_image
from .metrology import RegionMeasurement, measure_regions
from .preprocessing import PreprocessConfig
from .quality import ImageQuality, QualityConfig, assess_image_quality
from .reference import ReferenceConfig, ReferenceResidual, compare_to_reference
from .registration import RegistrationResult, rectified_perspective, register_ecc, rectify_perspective

__all__ = [
    "BinaryConfusion", "CameraCalibration", "Defect", "ImageQuality", "InspectionConfig", "InspectionResult",
    "PixelScale", "PreprocessConfig", "QualityConfig", "ReferenceConfig", "ReferenceResidual", "RegionMeasurement",
    "RegistrationResult", "assess_image_quality", "calibrate_camera", "compare_to_reference", "confusion_from_labels",
    "inspect_array", "inspect_image", "measure_regions", "register_ecc", "rectify_perspective",
]
