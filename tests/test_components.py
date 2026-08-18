import numpy as np
import pytest

from vision_inspection import (
    CameraCalibration,
    PixelScale,
    QualityConfig,
    assess_image_quality,
    confusion_from_labels,
    measure_regions,
)
from vision_inspection.preprocessing import PreprocessConfig, apply_morphology, normalize_illumination, threshold_image, to_grayscale


def test_quality_metrics_include_percentile_margins():
    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    quality = assess_image_quality(
        image,
        QualityConfig(min_sharpness=0.0, min_std=1.0, min_p1=0.0, max_p99=255.0),
    )
    assert quality.mean > 0
    assert quality.std > 1
    assert 0 <= quality.p1 <= quality.p99 <= 255


def test_preprocessing_modes_and_grayscale():
    image = np.full((20, 20, 3), (80, 100, 120), dtype=np.uint8)
    gray = to_grayscale(image)
    assert gray.shape == (20, 20)
    corrected = normalize_illumination(gray, PreprocessConfig(background_kernel=5, blur_kernel=3))
    assert corrected.dtype == np.uint8
    for mode in ("fixed", "otsu", "adaptive"):
        mask = threshold_image(corrected, mode=mode, adaptive_block=5, polarity="positive")
        assert mask.shape == gray.shape
    with pytest.raises(ValueError):
        threshold_image(gray, mode="adaptive", adaptive_block=4)
    with pytest.raises(ValueError):
        threshold_image(gray, polarity="invalid")
    with pytest.raises(ValueError):
        PreprocessConfig(polarity="invalid")


def test_metrology_reports_extended_shape_descriptors():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:50, 30:60] = 255
    measurements = measure_regions(mask, PixelScale(0.1))
    assert len(measurements) == 1
    measurement = measurements[0]
    assert measurement.area_px > 0
    assert measurement.area_physical == pytest.approx(measurement.area_px * 0.01)
    assert measurement.circularity > 0
    assert measurement.solidity == pytest.approx(1.0, abs=0.02)
    assert measurement.equivalent_diameter_px > 0
    assert measurement.equivalent_diameter_physical > 0
    assert measurement.min_rect_width_px > 0
    assert measurement.min_rect_height_px > 0
    assert np.isfinite(measurement.compactness)
    with pytest.raises(ValueError):
        PixelScale(0)


def test_confusion_metrics_include_mcc_and_intervals():
    confusion = confusion_from_labels(
        np.array([False, True, True, False, True]),
        np.array([False, True, False, True, True]),
    )
    assert confusion.true_positive == 2
    assert confusion.false_positive == 1
    assert confusion.false_negative == 1
    assert confusion.specificity == pytest.approx(0.5)
    assert confusion.f1 == pytest.approx(2 / 3)
    assert -1 <= confusion.matthews_correlation <= 1
    assert confusion.recall_interval()[0] < confusion.recall < confusion.recall_interval()[1]


def test_camera_calibration_persistence(tmp_path):
    calibration = CameraCalibration(
        np.array([[500.0, 0, 50], [0, 500.0, 50], [0, 0, 1]]),
        np.zeros((5, 1)),
        (100, 100),
        0.2,
        per_view_errors=np.array([0.1, 0.2]),
        intrinsic_std=np.ones(18) * 0.01,
    )
    path = tmp_path / "camera.npz"
    calibration.save(path)
    loaded = CameraCalibration.load(path)
    assert np.allclose(loaded.camera_matrix, calibration.camera_matrix)
    assert np.allclose(loaded.distortion, calibration.distortion)
    assert np.allclose(loaded.per_view_errors, calibration.per_view_errors)
    assert np.allclose(loaded.intrinsic_std, calibration.intrinsic_std)
    assert loaded.max_per_view_error == pytest.approx(0.2)
    loaded.assert_quality(max_rms=0.3, max_view_error=0.3)


def test_camera_calibration_rejects_wrong_image_size():
    calibration = CameraCalibration(np.eye(3), np.zeros((5, 1)), (20, 20), 0.1)
    with pytest.raises(ValueError):
        calibration.undistort(np.zeros((10, 10), dtype=np.uint8))
