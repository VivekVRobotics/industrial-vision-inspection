# Research Notes

## OpenCV calibration and registration

OpenCV provides camera calibration workflows that estimate camera intrinsics, distortion, and reprojection diagnostics. The repository stores these diagnostics rather than reducing calibration to one matrix. OpenCV also exposes ECC-based image registration and multiscale registration, which motivates the registration module and its correlation quality gate.

Reference: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
Reference: https://docs.opencv.org/4.x/dc/d6b/group__video__track.html

## ChArUco calibration

OpenCV's ArUco/ChArUco documentation recommends ChArUco corners for calibration because they provide accurate corner localization and remain usable when some board observations are unavailable. The package therefore exposes ChArUco calibration as a first-class API and stores intrinsic standard deviations and per-view errors.

Reference: https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html

## Industrial inspection scope

MVTec HALCON's current operator organization separates image acquisition, calibration, inspection, matching, metrology, morphology, segmentation, and 3D procedures. This supports the repository's layered architecture rather than treating inspection as one segmentation function.

Reference: https://www.mvtec.com/doc/halcon/2605/en/toc_system.html

## Measurement systems

NIST treats repeatability, reproducibility, stability, bias, resolution, linearity, and drift as separate measurement-process topics. ISO 5725 also distinguishes trueness and precision. The repository consequently separates classification benchmarks from measurement-system analysis and uncertainty.

Reference: https://www.nist.gov/programs-projects/gage-repeatability-reproducibility-and-stability-study
Reference: https://www.iso.org/standard/69420.html

## Uncertainty

NIST's uncertainty guidance covers the GUM framework, first-order propagation, and Monte Carlo methods. The repository exposes both a transparent first-order path and a Monte Carlo path, while explicitly avoiding a claim of full compliance from a small scalar model.

Reference: https://www.nist.gov/publications/uncertainty-measurement
Reference: https://www.itl.nist.gov/div898/handbook/mpc/section5/mpc55.htm

## Engineering principle

The code should make assumptions visible: calibration quality thresholds, image-quality gates, recipe versions, reference-image identity, uncertainty sources, benchmark labels, and study design are explicit inputs or persisted evidence. The repository should remain classical-vision first until real datasets justify learned components.
