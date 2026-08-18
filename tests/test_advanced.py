import cv2
import numpy as np
import pytest

from vision_inspection import ReferenceConfig, compare_to_reference, register_ecc, rectify_perspective


def test_reference_residual_detects_local_change_and_reports_statistics():
    reference = np.full((100, 100), 150, dtype=np.uint8)
    image = reference.copy()
    cv2.rectangle(image, (40, 40), (60, 60), 40, -1)
    result = compare_to_reference(reference, image, ReferenceConfig(threshold=10, min_area_px=25))
    assert result.changed_pixels > 0
    assert result.changed_fraction > 0
    assert result.mean_absolute_difference > 0
    assert result.p95_absolute_difference >= result.mean_absolute_difference
    assert result.max_absolute_difference >= int(result.p95_absolute_difference)


def test_reference_brightness_normalization_avoids_global_shift_false_alarm():
    reference = np.full((80, 80), 100, dtype=np.uint8)
    image = np.full((80, 80), 120, dtype=np.uint8)
    result = compare_to_reference(reference, image, ReferenceConfig(threshold=10, normalize_brightness=True))
    assert result.changed_pixels == 0


def test_reference_residual_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        compare_to_reference(np.zeros((20, 20), dtype=np.uint8), np.zeros((21, 20), dtype=np.uint8))


def test_perspective_rectification_preserves_target_shape():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 10), (90, 80), (255, 255, 255), -1)
    rectified = rectify_perspective(
        image,
        np.array([[20, 10], [90, 10], [90, 80], [20, 80]], dtype=np.float32),
        (71, 71),
    )
    assert rectified.shape == (71, 71, 3)
    assert int(rectified[35, 35].mean()) > 200


def test_perspective_rectification_rejects_degenerate_points():
    image = np.zeros((50, 50), dtype=np.uint8)
    points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)
    with pytest.raises(ValueError):
        rectify_perspective(image, points, (20, 20))


def test_ecc_registration_recovers_translation_with_quality_gate():
    reference = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(reference, (60, 60), 20, 180, -1)
    cv2.rectangle(reference, (20, 30), (40, 50), 230, -1)
    matrix = np.float32([[1, 0, 4], [0, 1, -3]])
    shifted = cv2.warpAffine(reference, matrix, (120, 120))
    result = register_ecc(reference, shifted, motion="translation", iterations=100, min_correlation=0.98)
    assert result.accepted
    assert result.correlation > 0.98
    assert result.motion == "translation"
    assert np.mean(np.abs(result.image.astype(float) - reference.astype(float))) < 3.0


def test_ecc_parameter_validation():
    image = np.zeros((20, 20), dtype=np.uint8)
    with pytest.raises(ValueError):
        register_ecc(image, image, motion="invalid")
    with pytest.raises(ValueError):
        register_ecc(image, image, iterations=0)
    with pytest.raises(ValueError):
        register_ecc(image, image, min_correlation=2.0)
