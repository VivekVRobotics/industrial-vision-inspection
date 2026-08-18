"""Measurement-system analysis for repeatability, reproducibility, and drift.

The implementation is intentionally explicit about its balanced crossed-study
assumption. NIST treats repeatability, reproducibility, stability, bias,
linearity, and drift as separate measurement-process concerns, so the module
keeps these analyses separate rather than collapsing them into one score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GRRResult:
    """Variance-component summary for a balanced [part, operator, repeat] study."""

    mean: float
    repeatability_std: float
    reproducibility_std: float
    operator_std: float
    interaction_std: float
    part_to_part_std: float
    total_grr_std: float
    study_variation_std: float
    percent_grr_of_study: float
    ndc: float
    ndc_rule_pass: bool


def crossed_grr(values: np.ndarray, *, study_sigma: float = 6.0, ndc_threshold: int = 5) -> GRRResult:
    """Estimate classical crossed Gage R&R components from balanced data.

    Parameters
    ----------
    values:
        Numeric array with shape ``[parts, operators, repeats]``.
    study_sigma:
        Multiplier used when converting standard deviations to study variation.
        Six is a common engineering convention; it is a reporting convention,
        not a probability statement about every possible non-normal process.
    ndc_threshold:
        Minimum number of distinct categories used for a simple screening flag.

    This function is a screening calculation. A released metrology process
    should use the organization-approved MSA procedure and study design.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 3 or min(x.shape) < 2:
        raise ValueError("values must have shape [parts, operators, repeats] with >=2 in each dimension")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite")
    if study_sigma <= 0 or ndc_threshold < 1:
        raise ValueError("study_sigma must be positive and ndc_threshold >=1")

    parts, operators, repeats = x.shape
    mean = float(np.mean(x))
    part_means = np.mean(x, axis=(1, 2))
    operator_means = np.mean(x, axis=(0, 2))
    cell_means = np.mean(x, axis=2)
    ms_repeat = np.sum((x - cell_means[:, :, None]) ** 2) / (parts * operators * (repeats - 1))
    ms_operator = np.sum((operator_means - mean) ** 2) / (operators - 1)
    ms_part = np.sum((part_means - mean) ** 2) / (parts - 1)
    ms_interaction = np.sum((cell_means - part_means[:, None] - operator_means[None, :] + mean) ** 2) / ((parts - 1) * (operators - 1))

    repeat_var = max(ms_repeat, 0.0)
    interaction_var = max((ms_interaction - ms_repeat) / repeats, 0.0)
    operator_var = max((ms_operator - ms_interaction) / (parts * repeats), 0.0)
    part_var = max((ms_part - ms_interaction) / (operators * repeats), 0.0)
    grr_var = repeat_var + interaction_var + operator_var
    study_var = grr_var + part_var
    grr_std = float(np.sqrt(grr_var))
    study_std = float(np.sqrt(study_var))
    ndc = float(1.41 * np.sqrt(part_var / grr_var)) if grr_var > 0 else float("inf")
    percent = 100.0 * grr_std / study_std if study_std > 0 else 0.0
    return GRRResult(
        mean=mean,
        repeatability_std=float(np.sqrt(repeat_var)),
        reproducibility_std=float(np.sqrt(operator_var + interaction_var)),
        operator_std=float(np.sqrt(operator_var)),
        interaction_std=float(np.sqrt(interaction_var)),
        part_to_part_std=float(np.sqrt(part_var)),
        total_grr_std=grr_std,
        study_variation_std=study_std,
        percent_grr_of_study=percent,
        ndc=ndc,
        ndc_rule_pass=ndc >= ndc_threshold,
    )


def process_drift(
    values: np.ndarray,
    *,
    window: int = 10,
    z_limit: float = 3.0,
    min_history_std: float = 1e-12,
) -> np.ndarray:
    """Flag rolling z-score excursions while avoiding zero-variance false alarms."""
    x = np.asarray(values, dtype=float).reshape(-1)
    if len(x) < 3 or window < 2 or z_limit <= 0 or min_history_std < 0:
        raise ValueError("invalid drift parameters")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite")
    flags = np.zeros(len(x), dtype=bool)
    for index in range(window, len(x)):
        history = x[index - window:index]
        mean = float(np.mean(history))
        std = float(np.std(history, ddof=1))
        if std > min_history_std and abs(x[index] - mean) / std >= z_limit:
            flags[index] = True
    return flags


def stability_summary(values: np.ndarray) -> tuple[float, float, float]:
    """Return mean, standard deviation, and linear drift slope for a sequence."""
    x = np.asarray(values, dtype=float).reshape(-1)
    if len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("values must contain at least two finite samples")
    axis = np.arange(len(x), dtype=float)
    slope = float(np.polyfit(axis, x, 1)[0])
    return float(np.mean(x)), float(np.std(x, ddof=1)), slope
