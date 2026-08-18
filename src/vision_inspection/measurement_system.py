"""Measurement-system studies: repeatability, reproducibility, and stability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GRRResult:
    """Summary of a crossed repeatability/reproducibility study."""

    mean: float
    repeatability_std: float
    reproducibility_std: float
    part_to_part_std: float
    total_grr_std: float
    study_variation: float
    ndc: float


def crossed_grr(values: np.ndarray) -> GRRResult:
    """Estimate crossed Gage R&R components from [part, operator, repeat].

    Uses a balanced crossed design and classical range/ANOVA variance
    decomposition. This is a screening implementation, not a substitute for
    a validated metrology package or formal uncertainty budget.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 3 or min(x.shape) < 2:
        raise ValueError("values must have shape [parts, operators, repeats] with >=2 in each dimension")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite")

    parts, operators, repeats = x.shape
    mean = float(np.mean(x))
    part_means = np.mean(x, axis=(1, 2))
    operator_means = np.mean(x, axis=(0, 2))
    cell_means = np.mean(x, axis=2)

    # Balanced ANOVA mean-square decomposition.
    ms_repeat = np.sum((x - cell_means[:, :, None]) ** 2) / (parts * operators * (repeats - 1))
    ms_operator = np.sum((operator_means - mean) ** 2) / (operators - 1)
    ms_part = np.sum((part_means - mean) ** 2) / (parts - 1)
    ms_cell = np.sum((cell_means - part_means[:, None] - operator_means[None, :] + mean) ** 2) / ((parts - 1) * (operators - 1))

    repeat_var = max(ms_repeat, 0.0)
    operator_part_var = max((ms_cell - ms_repeat) / repeats, 0.0)
    operator_var = max((ms_operator - ms_cell) / (parts * repeats), 0.0)
    part_var = max((ms_part - ms_cell) / (operators * repeats), 0.0)

    rr_var = repeat_var + operator_var + operator_part_var
    study_var = rr_var + part_var
    grr_std = float(np.sqrt(rr_var))
    study_std = float(np.sqrt(study_var))
    ndc = float(1.41 * np.sqrt(part_var / rr_var)) if rr_var > 0 else float("inf")

    return GRRResult(
        mean=mean,
        repeatability_std=float(np.sqrt(repeat_var)),
        reproducibility_std=float(np.sqrt(operator_var + operator_part_var)),
        part_to_part_std=float(np.sqrt(part_var)),
        total_grr_std=grr_std,
        study_variation=study_std,
        ndc=ndc,
    )


def process_drift(values: np.ndarray, *, window: int = 10, z_limit: float = 3.0) -> np.ndarray:
    """Flag rolling z-score excursions in a univariate measurement stream."""
    x = np.asarray(values, dtype=float).reshape(-1)
    if len(x) < 3 or window < 2 or z_limit <= 0:
        raise ValueError("values must contain >=3 samples; window >=2; z_limit >0")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite")
    flags = np.zeros(len(x), dtype=bool)
    for i in range(window, len(x)):
        history = x[max(0, i - window):i]
        std = float(np.std(history, ddof=1))
        if std > 0 and abs(x[i] - float(np.mean(history))) / std >= z_limit:
            flags[i] = True
    return flags
