"""Public API for the industrial machine-vision inspection toolkit.

The package intentionally exposes data models and deterministic processing
boundaries rather than vendor-specific camera integrations. Applications can
compose the public objects into acquisition, inspection, metrology, and MSA
workflows while keeping transport code outside the core package.
"""

from .acquisition import AcquisitionStats, Camera, Frame, TriggeredSequenceCamera
from .benchmarks import InspectionBenchmark, summarize_benchmark
from .calibration import CameraCalibration, CharucoCalibration, PixelScale, calibrate_camera, calibrate_charuco
from .evaluation import BinaryConfusion, confusion_from_labels
from .flat_field import FlatFieldStats, apply_flat_field, build_flat_field, flat_field_stats, smooth_flat_field
from .golden import GoldenSample, load_golden_registry, registry_digest, save_golden_registry
from .inspector import Defect, InspectionConfig, InspectionResult, inspect_array, inspect_image
from .measurement_system import GRRResult, crossed_grr, process_drift, stability_summary
from .metrology import RegionMeasurement, measure_regions
from .preprocessing import PreprocessConfig, apply_morphology, normalize_illumination, threshold_image, to_grayscale
from .quality import ImageQuality, QualityConfig, assess_image_quality
from .reference import ReferenceConfig, ReferenceResidual, compare_to_reference
from .registration import RegistrationResult, register_ecc, rectify_perspective
from .uncertainty import MeasurementUncertainty, monte_carlo_length_uncertainty, propagate_length_uncertainty

__all__ = [
    "AcquisitionStats", "BinaryConfusion", "Camera", "CameraCalibration", "CharucoCalibration", "Defect",
    "FlatFieldStats", "Frame", "GRRResult", "GoldenSample", "ImageQuality", "InspectionBenchmark",
    "InspectionConfig", "InspectionResult", "MeasurementUncertainty", "PixelScale", "PreprocessConfig",
    "QualityConfig", "ReferenceConfig", "ReferenceResidual", "RegionMeasurement", "RegistrationResult",
    "TriggeredSequenceCamera", "apply_flat_field", "apply_morphology", "assess_image_quality", "build_flat_field",
    "calibrate_camera", "calibrate_charuco", "compare_to_reference", "confusion_from_labels", "crossed_grr",
    "flat_field_stats", "inspect_array", "inspect_image", "load_golden_registry", "measure_regions", "monte_carlo_length_uncertainty",
    "normalize_illumination", "process_drift", "propagate_length_uncertainty", "rectify_perspective", "register_ecc",
    "registry_digest", "save_golden_registry", "smooth_flat_field", "stability_summary", "summarize_benchmark",
    "threshold_image", "to_grayscale",
]
