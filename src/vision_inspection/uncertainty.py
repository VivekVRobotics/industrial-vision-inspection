"""Measurement uncertainty utilities.

The analytic functions implement first-order propagation for transparent
engineering estimates. A Monte Carlo path is also provided for nonlinear or
non-Gaussian input assumptions. NIST describes both first-order propagation and
Monte Carlo propagation as useful approaches under appropriate conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeasurementUncertainty:
    """Measurement value and uncertainty interval."""

    value: float
    standard_uncertainty: float
    expanded_uncertainty: float
    coverage_factor: float = 2.0
    method: str = "first_order"

    @property
    def lower(self) -> float:
        return self.value - self.expanded_uncertainty

    @property
    def upper(self) -> float:
        return self.value + self.expanded_uncertainty


def propagate_length_uncertainty(
    pixel_length: float,
    units_per_pixel: float,
    pixel_sigma: float,
    scale_sigma: float,
    *,
    coverage_factor: float = 2.0,
) -> MeasurementUncertainty:
    """Propagate independent pixel and scale uncertainty to physical length."""
    values = (pixel_length, units_per_pixel, pixel_sigma, scale_sigma, coverage_factor)
    if not all(np.isfinite(v) for v in values):
        raise ValueError("all uncertainty inputs must be finite")
    if pixel_length < 0 or units_per_pixel <= 0 or pixel_sigma < 0 or scale_sigma < 0 or coverage_factor <= 0:
        raise ValueError("invalid uncertainty inputs")
    value = pixel_length * units_per_pixel
    variance = (units_per_pixel * pixel_sigma) ** 2 + (pixel_length * scale_sigma) ** 2
    standard = float(np.sqrt(max(variance, 0.0)))
    return MeasurementUncertainty(value, standard, coverage_factor * standard, coverage_factor)


def monte_carlo_length_uncertainty(
    pixel_length: float,
    units_per_pixel: float,
    pixel_sigma: float,
    scale_sigma: float,
    *,
    samples: int = 100_000,
    coverage: float = 0.95,
    seed: int = 0,
) -> MeasurementUncertainty:
    """Propagate independent Gaussian input uncertainties by Monte Carlo sampling."""
    values = (pixel_length, units_per_pixel, pixel_sigma, scale_sigma)
    if not all(np.isfinite(v) for v in values) or pixel_length < 0 or units_per_pixel <= 0 or pixel_sigma < 0 or scale_sigma < 0:
        raise ValueError("invalid Monte Carlo inputs")
    if samples < 1000 or not 0 < coverage < 1:
        raise ValueError("samples must be >=1000 and coverage in (0,1)")
    rng = np.random.default_rng(seed)
    lengths = rng.normal(pixel_length, pixel_sigma, samples) * rng.normal(units_per_pixel, scale_sigma, samples)
    lengths = lengths[np.isfinite(lengths)]
    if lengths.size < samples * 0.99:
        raise ValueError("too many non-finite Monte Carlo samples")
    alpha = (1.0 - coverage) / 2.0
    lower, upper = np.quantile(lengths, [alpha, 1.0 - alpha])
    value = float(np.mean(lengths))
    standard = float(np.std(lengths, ddof=1))
    expanded = max(value - float(lower), float(upper) - value)
    return MeasurementUncertainty(value, standard, expanded, expanded / standard if standard > 0 else 0.0, "monte_carlo")
