"""Simple uncertainty propagation for calibrated planar metrology."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeasurementUncertainty:
    """Approximate 1-sigma uncertainty for a scalar length measurement."""

    value: float
    standard_uncertainty: float
    expanded_uncertainty: float
    coverage_factor: float = 2.0


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
    return MeasurementUncertainty(
        value=float(value),
        standard_uncertainty=standard,
        expanded_uncertainty=float(coverage_factor * standard),
        coverage_factor=float(coverage_factor),
    )
