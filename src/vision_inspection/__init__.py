"""Industrial machine-vision inspection toolkit."""

from .calibration import CameraCalibration, PixelScale, calibrate_camera
from .evaluation import BinaryConfusion, confusion_from_labels
from .inspector import Defect, InspectionConfig, InspectionResult, inspect_array, inspect_image
from .metrology import RegionMeasurement, measure_regions
from .preprocessing import PreprocessConfig
from .quality import ImageQuality, QualityConfig, assess_image_quality

__all__ = [
    "BinaryConfusion",
    "CameraCalibration",
    "Defect",
    "ImageQuality",
    "InspectionConfig",
    "InspectionResult",
    "PixelScale",
    "PreprocessConfig",
    "QualityConfig",
    "RegionMeasurement",
    "assess_image_quality",
    "calibrate_camera",
    "confusion_from_labels",
    "inspect_array",
    "inspect_image",
    "measure_regions",
]
