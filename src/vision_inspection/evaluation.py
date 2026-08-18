"""Inspection-performance metrics independent of the vision algorithm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryConfusion:
    """Confusion counts for part-level PASS/FAIL evaluation."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

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
        return 1.0 - self.recall

    @property
    def false_reject_rate(self) -> float:
        d = self.true_negative + self.false_positive
        return self.false_positive / d if d else 0.0

    @property
    def f1(self) -> float:
        d = self.precision + self.recall
        return 2.0 * self.precision * self.recall / d if d else 0.0


def confusion_from_labels(actual_fail: np.ndarray, predicted_fail: np.ndarray) -> BinaryConfusion:
    """Compute a part-level confusion matrix from boolean label arrays."""
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
