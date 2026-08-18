import cv2
import numpy as np
import pytest

from vision_inspection import InspectionConfig, inspect_image


def test_clean_image_passes(tmp_path):
    image = np.full((120, 160, 3), 200, dtype=np.uint8)
    path = tmp_path / "clean.png"
    cv2.imwrite(str(path), image)

    result = inspect_image(path)
    assert result.passed
    assert result.defect_count == 0


def test_dark_defect_is_detected(tmp_path):
    image = np.full((120, 160, 3), 200, dtype=np.uint8)
    cv2.rectangle(image, (60, 40), (85, 65), (20, 20, 20), -1)
    path = tmp_path / "defect.png"
    cv2.imwrite(str(path), image)

    result = inspect_image(path, InspectionConfig(min_area=100))
    assert not result.passed
    assert result.defect_count == 1
    assert result.defects[0].area >= 100


def test_roi_restricts_inspection(tmp_path):
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    cv2.rectangle(image, (5, 5), (20, 20), (0, 0, 0), -1)
    path = tmp_path / "roi.png"
    cv2.imwrite(str(path), image)

    result = inspect_image(path, InspectionConfig(roi=(50, 50, 40, 40)))
    assert result.passed


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError):
        InspectionConfig(threshold=300)
