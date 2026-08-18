# Module Reference

## `acquisition.py`
Defines the camera adapter protocol, immutable frame metadata, deterministic triggered sequence camera, and acquisition counters.

## `quality.py`
Measures exposure, contrast, percentile headroom, Laplacian sharpness, and saturation. Produces explicit gate reasons.

## `calibration.py`
Stores pinhole calibration, distortion, per-view reprojection error, intrinsic standard deviations, ChArUco results, and planar pixel scale.

## `flat_field.py`
Builds illumination fields from repeated uniform captures, reports field non-uniformity, and applies multiplicative correction.

## `registration.py`
Provides bounded ECC registration and four-corner perspective rectification. Registration is treated as a quality-controlled preprocessing step.

## `preprocessing.py`
Contains grayscale conversion, Gaussian/median filtering, black-hat/top-hat enhancement, optional CLAHE, thresholding, and morphology.

## `metrology.py`
Extracts region area, perimeter, centroid, bounding box, equivalent diameter, oriented minimum rectangle, aspect ratio, circularity, extent, solidity, and calibrated physical quantities.

## `inspector.py`
Composes the complete inspection recipe and returns the traceable immutable `InspectionResult`.

## `reference.py`
Performs golden-image residual inspection with optional global brightness normalization and localized connected-component filtering.

## `golden.py`
Provides immutable golden-sample metadata, content hashing, recipe version association, approval identity, registry integrity digest, and persistence.

## `uncertainty.py`
Provides first-order uncertainty propagation and Monte Carlo propagation for calibrated scalar length measurements.

## `evaluation.py`
Provides binary classification confusion metrics, confidence intervals, balanced accuracy, and Matthews correlation.

## `benchmarks.py`
Adds latency percentiles, throughput, false-accept/reject metrics, and Wilson confidence intervals to benchmark studies.

## `measurement_system.py`
Provides balanced crossed Gage R&R variance decomposition, screening metrics, drift detection, and simple stability slope estimation.

## `visualization.py`
Renders inspection evidence without changing the inspection decision. Annotations include status, quality failures, recipe identity, image identity, and processing time.

## `cli.py`
Provides a thin JSON-producing command-line boundary around the public inspection API.
