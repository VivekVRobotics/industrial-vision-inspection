import numpy as np
import pytest

from vision_inspection import PixelScale, QualityConfig, assess_image_quality, confusion_from_labels, measure_regions
from vision_inspection.calibration import CameraCalibration
from vision_inspection.preprocessing import PreprocessConfig, apply_morphology, normalize_illumination, threshold_image, to_grayscale


def test_quality_metrics_and_threshold_validation():
    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    quality = assess_image_quality(image, QualityConfig(min_sharpness=0.0, min_std=1.0))
    assert quality.mean > 0
    assert quality.std > 1
    with pytest.raises(ValueError):
        QualityConfig(max_saturated_fraction=1.2)


def test_preprocessing_modes_and_grayscale():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :] = (80, 100, 120)
    gray = to_grayscale(image)
    assert gray.shape == (20, 20)
    corrected = normalize_illumination(gray, PreprocessConfig(background_kernel=5, blur_kernel=3))
    assert corrected.dtype == np.uint8
    for mode in ("fixed", "otsu", "adaptive"):
        mask = threshold_image(corrected, mode=mode, adaptive_block=5)
        assert mask.shape == gray.shape
    with pytest.raises(ValueError):
        threshold_image(gray, mode="adaptive", adaptive_block=4)
    with pytest.raises(ValueError):
        PreprocessConfig(polarity="invalid")


def test_metrology_and_pixel_scale():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:50, 30:60] = 255
    measurements = measure_regions(mask, PixelScale(0.1))
    assert len(measurements) == 1
    measurement = measurements[0]
    assert measurement.area_px > 0
    assert measurement.area_physical == pytest.approx(measurement.area_px * 0.01)
    assert measurement.circularity > 0
    assert measurement.solidity == pytest.approx(1.0, abs=0.02)
    with pytest.raises(ValueError):
        PixelScale(0)


def test_confusion_metrics():
    confusion = confusion_from_labels(
        np.array([False, True, True, False, True]),
        np.array([False, True, False, True, True]),
    )
    assert confusion.true_positive == 2
    assert confusion.false_positive == 1
    assert confusion.false_negative == 1
    assert confusion.specificity == pytest.approx(0.5)
    assert confusion.f1 == pytest.approx(2 / 3)


def test_camera_calibration_persistence(tmp_path):
    calibration = CameraCalibration(
        np.array([[500.0, 0, 50], [0, 500.0, 50], [0, 0, 1]]),
        np.zeros((5, 1)),
        (100, 100),
        0.2,
    )
    path = tmp_path / "camera.npz"
    calibration.save(path)
    loaded = CameraCalibration.load(path)
    assert np.allclose(loaded.camera_matrix, calibration.camera_matrix)
    assert np.allclose(loaded.distortion, calibration.distortion)
    assert loaded.image_size == calibration.image_size
    assert loaded.rms_error == pytest.approx(calibration.rms_error)


def test_camera_calibration_rejects_wrong_image_size():
    calibration = CameraCalibration(np.eye(3), np.zeros((5, 1)), (20, 20), 0.1)
    with pytest.raises(ValueError):
        calibration.undistort(np.zeros((10, 10), dtype=np.uint8))
