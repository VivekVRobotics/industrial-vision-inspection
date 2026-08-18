import json

import cv2
import numpy as np
import pytest

from vision_inspection import (
    GoldenSample,
    TriggeredSequenceCamera,
    apply_flat_field,
    build_flat_field,
    crossed_grr,
    process_drift,
    propagate_length_uncertainty,
    save_golden_registry,
    load_golden_registry,
    summarize_benchmark,
)


def test_triggered_sequence_camera_requires_start_and_preserves_metadata():
    image = np.full((16, 16), 100, dtype=np.uint8)
    camera = TriggeredSequenceCamera([image], timestamps=[1.0])
    with pytest.raises(RuntimeError):
        camera.trigger(7)
    camera.start()
    frame = camera.trigger(7)
    assert frame.frame_id == 0
    assert frame.trigger_id == 7
    assert frame.timestamp == 1.0
    assert np.array_equal(frame.image, image)


def test_flat_field_corrects_multiplicative_gradient():
    base = np.full((20, 20), 100, dtype=np.float32)
    field = np.linspace(0.5, 1.5, 20, dtype=np.float32)[None, :].repeat(20, axis=0)
    references = base[None, :, :] * field[None, :, :]
    calibration = build_flat_field(references)
    corrected = apply_flat_field((base * field).astype(np.uint16), calibration)
    assert np.mean(corrected) == pytest.approx(100, abs=2)


def test_uncertainty_propagates_independent_sources():
    result = propagate_length_uncertainty(100.0, 0.01, 1.0, 0.0001)
    assert result.value == pytest.approx(1.0)
    assert result.standard_uncertainty > 0
    assert result.expanded_uncertainty == pytest.approx(2 * result.standard_uncertainty)


def test_crossed_grr_reports_low_repeatability_for_stable_measurements():
    rng = np.random.default_rng(42)
    part_means = np.array([10.0, 10.5, 11.0, 11.5])
    values = part_means[:, None, None] + rng.normal(0, 0.002, size=(4, 3, 3))
    result = crossed_grr(values)
    assert result.repeatability_std < 0.01
    assert result.part_to_part_std > result.total_grr_std
    assert result.ndc > 1


def test_process_drift_flags_an_outlier():
    values = np.array([1.0] * 10 + [1.01, 1.02, 2.0])
    flags = process_drift(values, window=5, z_limit=3.0)
    assert flags[-1]


def test_golden_sample_registry_round_trip(tmp_path):
    sample = GoldenSample.create("part-001", b"image-bytes", "recipe-v2")
    path = tmp_path / "golden.json"
    save_golden_registry([sample], path)
    loaded = load_golden_registry(path)
    assert loaded == [sample]
    assert json.loads(path.read_text(encoding="utf-8"))[0]["recipe_version"] == "recipe-v2"


def test_benchmark_exposes_false_accept_and_reject_rates():
    result = summarize_benchmark(
        [True, True, False, False],
        [True, False, True, False],
        [1.0, 2.0, 3.0, 4.0],
    )
    assert result.true_positive == 1
    assert result.false_positive == 1
    assert result.false_negative == 1
    assert result.false_accept_rate == pytest.approx(0.5)
    assert result.false_reject_rate == pytest.approx(0.5)
    assert result.p95_latency_ms == pytest.approx(3.0)
