"""Benchmark summaries for industrial inspection performance."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InspectionBenchmark:
    """Part-level classification and latency summary."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    mean_latency_ms: float
    p95_latency_ms: float

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def false_accept_rate(self) -> float:
        denom = self.true_negative + self.false_positive
        return self.false_positive / denom if denom else 0.0

    @property
    def false_reject_rate(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.false_negative / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def summarize_benchmark(y_true: list[bool], y_pred: list[bool], latency_ms: list[float]) -> InspectionBenchmark:
    """Build a benchmark summary from part-level labels and per-frame latency."""
    if not (len(y_true) == len(y_pred) == len(latency_ms)) or not y_true:
        raise ValueError("labels and latency must have the same non-zero length")
    if any(latency < 0 for latency in latency_ms):
        raise ValueError("latency must be non-negative")
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    ordered = sorted(float(v) for v in latency_ms)
    p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return InspectionBenchmark(
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        mean_latency_ms=sum(ordered) / len(ordered),
        p95_latency_ms=ordered[p95_index],
    )
