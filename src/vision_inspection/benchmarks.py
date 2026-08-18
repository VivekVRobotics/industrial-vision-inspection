"""Industrial inspection benchmark statistics.

The module separates two questions:

* classification quality — false accepts, false rejects, precision, recall;
* system performance — latency distribution and throughput.

For finite samples, point estimates alone can be misleading. Wilson score
intervals are included for binary rates so benchmark reports can expose the
amount of statistical uncertainty in the observed rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np


def _wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if trials <= 0 or successes < 0 or successes > trials or z <= 0:
        raise ValueError("invalid binomial inputs")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * sqrt((p * (1.0 - p) / trials) + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


@dataclass(frozen=True)
class InspectionBenchmark:
    """Immutable part-level classification and latency summary."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def specificity(self) -> float:
        denominator = self.true_negative + self.false_positive
        return self.true_negative / denominator if denominator else 0.0

    @property
    def false_accept_rate(self) -> float:
        """Rate at which actually-failing parts were accepted."""
        return self.false_negative / (self.true_positive + self.false_negative) if self.true_positive + self.false_negative else 0.0

    @property
    def false_reject_rate(self) -> float:
        """Rate at which actually-good parts were rejected."""
        return self.false_positive / (self.true_negative + self.false_positive) if self.true_negative + self.false_positive else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2.0 * p * r / (p + r) if p + r else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return 0.5 * (self.recall + self.specificity)

    @property
    def throughput_per_second(self) -> float:
        return 1000.0 / self.mean_latency_ms if self.mean_latency_ms > 0 else float("inf")

    def recall_interval(self, z: float = 1.96) -> tuple[float, float]:
        """Approximate confidence interval for recall."""
        return _wilson(self.true_positive, self.true_positive + self.false_negative, z)

    def false_accept_interval(self, z: float = 1.96) -> tuple[float, float]:
        """Approximate confidence interval for false-accept rate."""
        return _wilson(self.false_negative, self.true_positive + self.false_negative, z)

    def false_reject_interval(self, z: float = 1.96) -> tuple[float, float]:
        """Approximate confidence interval for false-reject rate."""
        return _wilson(self.false_positive, self.true_negative + self.false_positive, z)


def _percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0 or not 0 <= percentile <= 100:
        raise ValueError("percentile must be in [0,100] and values non-empty")
    return float(np.percentile(values, percentile, method="linear"))


def summarize_benchmark(
    y_true: list[bool],
    y_pred: list[bool],
    latency_ms: list[float],
) -> InspectionBenchmark:
    """Build a benchmark summary from part-level labels and per-frame latency."""
    if not (len(y_true) == len(y_pred) == len(latency_ms)) or not y_true:
        raise ValueError("labels and latency must have the same non-zero length")
    latencies = np.asarray(latency_ms, dtype=float)
    if np.any(~np.isfinite(latencies)) or np.any(latencies < 0):
        raise ValueError("latency must be finite and non-negative")
    actual = np.asarray(y_true, dtype=bool)
    predicted = np.asarray(y_pred, dtype=bool)
    return InspectionBenchmark(
        true_positive=int(np.sum(actual & predicted)),
        false_positive=int(np.sum(~actual & predicted)),
        true_negative=int(np.sum(~actual & ~predicted)),
        false_negative=int(np.sum(actual & ~predicted)),
        mean_latency_ms=float(np.mean(latencies)),
        p95_latency_ms=_percentile(latencies, 95),
        p99_latency_ms=_percentile(latencies, 99),
    )
