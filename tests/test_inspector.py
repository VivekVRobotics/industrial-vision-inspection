import cv2
import numpy as np
import pytest

from vision_inspection import InspectionConfig, PixelScale, PreprocessConfig, QualityConfig, inspect_array, inspect_image


def _reference_image(height=160, width=200):
    x = np.linspace(0, 1, width, dtype=np.float32)
    gradient = np.tile((175 + 25 * x).astype(np.uint8), (height, 1))
    image = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    return image


def test_clean_reference_image_passes():
    result = inspect_array(
        _reference_image(),
        InspectionConfig(quality=QualityConfig(min_sharpness=0.0, min_std=2.0)),
    )
    assert result.passed
    assert result.defect_count == 0
    assert result.quality.passed


def test_dark_defect_is_detected_with_metrology():
    image = _reference_image()
    cv2.rectangle(image, (70, 50), (98, 78), (25, 25, 25), -1)
    result = inspect_array(
        image,
        InspectionConfig(
            quality=QualityConfig(min_sharpness=0.0, min_std=2.0),
            min_area_px=100,
            max_area_px=2000,
        ),
        pixel_scale=PixelScale(0.1, "mm"),
    )
    assert not result.passed
    assert result.defect_count == 1
    defect = result.defects[0].measurement
    assert defect.area_px > 500
    assert defect.area_physical is not None
    assert defect.perimeter_physical is not None
    assert defect.bbox_px[0] >= 65


def test_roi_restricts_inspection():
    image = _reference_image(120, 120)
    cv2.rectangle(image, (5, 5), (24, 24), (20, 20, 20), -1)
    result = inspect_array(
        image,
        InspectionConfig(
            quality=QualityConfig(min_sharpness=0.0, min_std=2.0),
            roi=(40, 40, 60, 60),
        ),
    )
    assert result.passed


def test_quality_gate_rejects_flat_frame():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = inspect_array(image)
    assert not result.passed
    assert "low_contrast" in result.quality.failures
    assert "low_sharpness" in result.quality.failures


def test_bad_quality_can_be_overridden():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    config = InspectionConfig(reject_bad_image_quality=False)
    result = inspect_array(image, config)
    assert result.passed


def test_adaptive_and_morphology_recipe_options():
    image = _reference_image()
    cv2.circle(image, (100, 80), 12, (15, 15, 15), -1)
    config = InspectionConfig(
        preprocess=PreprocessConfig(background_kernel=31, blur_kernel=3),
        quality=QualityConfig(min_sharpness=0.0, min_std=2.0),
        segmentation_mode="adaptive",
        adaptive_block=31,
        opening=3,
        closing=3,
        min_area_px=50,
    )
    result = inspect_array(image, config)
    assert result.defect_count >= 1


def test_shape_filters_can_exclude_candidate():
    image = _reference_image()
    cv2.rectangle(image, (55, 70), (145, 73), (10, 10, 10), -1)
    config = InspectionConfig(
        quality=QualityConfig(min_sharpness=0.0, min_std=2.0),
        min_area_px=20,
        max_aspect_ratio=5.0,
    )
    result = inspect_array(image, config)
    assert result.passed


def test_border_rejection_is_explicit():
    image = _reference_image()
    cv2.rectangle(image, (0, 60), (20, 80), (10, 10, 10), -1)
    config = InspectionConfig(
        quality=QualityConfig(min_sharpness=0.0, min_std=2.0),
        reject_border_touching=True,
    )
    result = inspect_array(image, config)
    assert result.passed


def test_missing_image_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_image(tmp_path / "missing.png")
