# Industrial Validation Protocol

This document defines how a real deployment should be validated. Code-level unit tests are necessary but are not sufficient for an inspection system.

## 1. Acquisition characterization

Record camera model, lens, working distance, field of view, exposure, gain, trigger mode, lighting configuration, temperature range, and image bit depth.

Validate:

- trigger-to-frame association;
- timestamp ordering and synchronization;
- dropped-frame behavior;
- exposure/gain repeatability;
- image-quality gate behavior;
- cycle-time distribution.

## 2. Calibration study

Capture multiple poses of a checkerboard or ChArUco board spanning the usable field of view and depth range. Record RMS and per-view reprojection errors, camera matrix, distortion, and intrinsic standard deviations.

Do not accept a calibration only because the optimizer returned a solution. Define an application-specific maximum reprojection error and verify measurements on independent validation images.

## 3. Flat-field study

Acquire repeated uniform frames under the intended production illumination, lens aperture, exposure, and gain. Compute the normalized field and its coefficient of variation. Retain the field as a versioned artifact and verify it against a fresh uniform capture before production use.

## 4. Metrology study

Choose reference artifacts that span the actual inspection tolerance range. Measure them repeatedly. Record nominal value, observed value, bias, repeatability, and uncertainty. Validate the measurement resolution and calibration scale independently of the defect detector.

## 5. Gage R&R study

Use a balanced crossed design with representative parts, multiple operators, and repeated measurements. Run the repository's screening calculation, then compare the result with the site's approved MSA procedure before release.

## 6. Inspection benchmark

Create a locked evaluation set containing both conforming and known nonconforming parts. Do not tune thresholds against the locked set. Record:

- TP, FP, TN, FN;
- false-accept rate;
- false-reject rate;
- precision, recall, F1, balanced accuracy;
- latency mean/P95/P99;
- cycle-time budget;
- confidence intervals for critical rates.

Report subgroup performance by lighting condition, lot, fixture, operator, camera, and part family when those factors matter.

## 7. Golden-reference governance

Golden samples must have:

- immutable content hash;
- recipe version;
- approval identity;
- creation timestamp;
- documented reason for approval.

A recipe or golden-image change should trigger a controlled validation cycle rather than silent replacement.

## 8. Stability / drift

Monitor a stable reference part or calibrated artifact over time. Record a time-ordered measurement stream. Use the repository's drift detector as a lightweight screening alarm, but use the site's approved SPC/control-chart rules for release decisions.

## 9. Release gate

A production release should not be approved from software tests alone. It should have evidence for acquisition stability, calibration quality, measurement capability, defect detection performance, cycle time, and long-term drift under representative operating conditions.
