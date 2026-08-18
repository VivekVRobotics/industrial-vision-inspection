# Measurement Uncertainty

The repository exposes two uncertainty paths for calibrated scalar length measurements.

## First-order propagation

For

`y = L_px * s`

with independent standard uncertainties `u(L_px)` and `u(s)`, the implementation uses

`u(y)^2 = (s * u(L_px))^2 + (L_px * u(s))^2`.

This is the first-order Taylor / root-sum-square approximation. It is transparent and appropriate when the model is locally linear and input uncertainty assumptions are credible.

## Monte Carlo propagation

`monte_carlo_length_uncertainty()` samples the pixel length and scale distributions, computes the derived physical length, and returns a central interval at the requested coverage. Use this path when nonlinearity or distribution shape makes first-order assumptions questionable.

## What is not included

The simple scalar model does not automatically include lens distortion residuals, perspective error, calibration target uncertainty, thermal effects, fixture motion, segmentation bias, operator effects, or correlation among inputs. Those contributions should be added to an application-specific uncertainty budget.

## Reporting

Store:

- measured value;
- standard uncertainty;
- expanded uncertainty;
- coverage factor or requested coverage;
- uncertainty method;
- calibration identifier;
- recipe version.

An uncertainty value without its measurement model and input assumptions is not sufficient evidence for a production metrology claim.
