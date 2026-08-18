"""Algorithm-independent inspection classification metrics.

This module operates on already-decided part-level labels. It intentionally
contains no image-processing code so detector experiments can be compared
without changing the evaluation layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np


def _wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials or z <= 0:
        raise ValueError("invalid binomial inputs")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * sqrt((p * (1.0 - p) / trials) + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


@dataclass(frozen=True)
class BinaryConfusion:
    """Immutable confusion matrix for part-level PASS/FAIL decisions."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    def __post_init__(self) -> None:
        counts = (self.true_positive, self.false_positive, self.true_negative, self.false_negative)
        if any(count < 0 for count in counts):
            raise ValueError("confusion counts must be non-negative")

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        d = self.true_positive + self.false_positive
        return self.true_positive / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positive + self.false_negative
        return self.true_positive / d if d else 0.0

    @property
    def specificity(self) -> float:
        d = self.true_negative + self.false_positive
        return self.true_negative / d if d else 0.0

    @property
    def false_accept_rate(self) -> float:
        d = self.true_positive + self.false_negative
        return self.false_negative / d if d else 0.0

    @property
    def false_reject_rate(self) -> float:
        d = self.true_negative + self.false_positive
        return self.false_positive / d if d else 0.0

    @property
    def f1(self) -> float:
        d = self.precision + self.recall
        return 2.0 * self.precision * self.recall / d if d else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return 0.5 * (self.recall + self.specificity)

    @property
    def matthews_correlation(self) -> float:
        denominator = sqrt(
            (self.true_positive + self.false_positive)
            * (self.true_positive + self.false_negative)
            * (self.true_negative + self.false_positive)
            * (self.true_negative + self.false_negative)
        )
        numerator = self.true_positive * self.true_negative - self.false_positive * self.false_negative
        return numerator / denominator if denominator else 0.0

    def recall_interval(self, z: float = 1.96) -> tuple[float, float]:
        return _wilson(self.true_positive, self.true_positive + self.false_negative, z)

    def false_accept_interval(self, z: float = 1.96) -> tuple[float, float]:
        return _wilson(self.false_negative, self.true_positive + self.false_negative, z)

    def false_reject_interval(self, z: float = 1.96) -> tuple[float, float]:
        return _wilson(self.false_positive, self.true_negative + self.false_positive, z)


def confusion_from_labels(actual_fail: np.ndarray, predicted_fail: np.ndarray) -> BinaryConfusion:
    """Compute a binary confusion matrix from aligned boolean label arrays."""
    actual = np.asarray(actual_fail, dtype=bool).reshape(-1)
    predicted = np.asarray(predicted_fail, dtype=bool).reshape(-1)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("actual and predicted labels must be non-empty and have matching shapes")
    return BinaryConfusion(
        true_positive=int(np.sum(actual & predicted)),
        false_positive=int(np.sum(~actual & predicted)),
        true_negative=int(np.sum(~actual & ~predicted)),
        false_negative=int(np.sum(actual & ~predicted)),
    )
