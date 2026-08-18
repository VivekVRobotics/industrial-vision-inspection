"""Industrial machine-vision inspection toolkit."""

from .acquisition import Frame, TriggeredSequenceCamera
from .benchmarks import InspectionBenchmark, summarize_benchmark
from .calibration import CameraCalibration, CharucoCalibration, PixelScale, calibrate_camera, calibrate_charuco
from .evaluation import BinaryConfusion, confusion_from_labels
from .flat_field import apply_flat_field, build_flat_field, smooth_flat_field
from .golden import GoldenSample, load_golden_registry, save_golden_registry
from .inspector import Defect, InspectionConfig, InspectionResult, inspect_array, inspect_image
from .measurement_system import GRRResult, crossed_grr, process_drift
from .metrology import RegionMeasurement, measure_regions
from .preprocessing import PreprocessConfig
from .quality import ImageQuality, QualityConfig, assess_image_quality
from .reference import ReferenceConfig, ReferenceResidual, compare_to_reference
from .registration import RegistrationResult, register_ecc, rectify_perspective
from .uncertainty import MeasurementUncertainty, propagate_length_uncertainty

__all__ = [
    "BinaryConfusion", "CameraCalibration", "CharucoCalibration", "Defect", "Frame", "GRRResult", "GoldenSample",
    "ImageQuality", "InspectionBenchmark", "InspectionConfig", "InspectionResult", "MeasurementUncertainty",
    "PixelScale", "PreprocessConfig", "QualityConfig", "ReferenceConfig", "ReferenceResidual",
    "RegionMeasurement", "RegistrationResult", "TriggeredSequenceCamera", "apply_flat_field",
    "assess_image_quality", "build_flat_field", "calibrate_camera", "calibrate_charuco", "compare_to_reference",
    "confusion_from_labels", "crossed_grr", "inspect_array", "inspect_image", "load_golden_registry",
    "measure_regions", "process_drift", "propagate_length_uncertainty", "rectify_perspective", "register_ecc",
    "save_golden_registry", "smooth_flat_field", "summarize_benchmark",
]
