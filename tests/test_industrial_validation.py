import json

import numpy as np
import pytest

from vision_inspection import (
    GoldenSample,
    TriggeredSequenceCamera,
    apply_flat_field,
    build_flat_field,
    crossed_grr,
    load_golden_registry,
    monte_carlo_length_uncertainty,
    process_drift,
    propagate_length_uncertainty,
    registry_digest,
    save_golden_registry,
    stability_summary,
    summarize_benchmark,
)


def test_triggered_camera_requires_start_and_preserves_metadata():
    image = np.full((16, 16), 100, dtype=np.uint8)
    camera = TriggeredSequenceCamera([image], timestamps=[1.0], exposures_us=[500.0], gains_db=[6.0])
    with pytest.raises(RuntimeError):
        camera.trigger(7)
    camera.start()
    frame = camera.trigger(7)
    assert frame.frame_id == 0
    assert frame.trigger_id == 7
    assert frame.timestamp == 1.0
    assert frame.exposure_us == 500.0
    assert frame.gain_db == 6.0
    assert camera.stats.remaining_frames == 0


def test_flat_field_corrects_multiplicative_gradient():
    base = np.full((20, 20), 100, dtype=np.float32)
    field = np.linspace(0.5, 1.5, 20, dtype=np.float32)[None, :].repeat(20, axis=0)
    references = base[None, :, :] * field[None, :, :]
    calibration = build_flat_field(references)
    corrected = apply_flat_field((base * field).astype(np.uint16), calibration)
    assert float(np.mean(corrected)) == pytest.approx(100, abs=2)


def test_uncertainty_analytic_and_monte_carlo_paths():
    analytic = propagate_length_uncertainty(100.0, 0.01, 1.0, 0.0001)
    monte_carlo = monte_carlo_length_uncertainty(100.0, 0.01, 1.0, 0.0001, samples=5000, seed=42)
    assert analytic.value == pytest.approx(1.0)
    assert analytic.standard_uncertainty > 0
    assert monte_carlo.standard_uncertainty > 0
    assert monte_carlo.method == "monte_carlo"


def test_crossed_grr_reports_repeatability_and_ndc():
    rng = np.random.default_rng(42)
    part_means = np.array([10.0, 10.5, 11.0, 11.5])
    values = part_means[:, None, None] + rng.normal(0, 0.002, size=(4, 3, 3))
    result = crossed_grr(values)
    assert result.repeatability_std < 0.01
    assert result.part_to_part_std > result.total_grr_std
    assert result.ndc > 1
    assert result.percent_grr_of_study < 10
    assert result.ndc_rule_pass


def test_process_drift_and_stability_summary():
    values = np.array([1.0] * 10 + [1.01, 1.02, 2.0])
    flags = process_drift(values, window=5, z_limit=3.0)
    assert flags[-1]
    mean, std, slope = stability_summary(np.arange(5, dtype=float))
    assert mean == pytest.approx(2.0)
    assert std > 0
    assert slope == pytest.approx(1.0)


def test_golden_registry_round_trip_and_integrity(tmp_path):
    sample = GoldenSample.create("part-001", b"image-bytes", "recipe-v2", approved_by="qa")
    path = tmp_path / "golden.json"
    save_golden_registry([sample], path)
    loaded = load_golden_registry(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == [sample]
    assert document["samples"][0]["recipe_version"] == "recipe-v2"
    assert document["registry_sha256"] == registry_digest([sample])
    assert sample.verify_bytes(b"image-bytes")
    assert not sample.verify_bytes(b"modified")


def test_benchmark_exposes_rates_confidence_intervals_and_latency_percentiles():
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
    assert result.p95_latency_ms == pytest.approx(3.85)
    assert 0 < result.recall_interval()[0] < result.recall_interval()[1] <= 1
