# Industrial Vision Inspection

An explainable, recipe-driven machine-vision inspection toolkit for industrial surface defects, geometric measurement, acquisition-quality gating, camera calibration, registration, golden-reference comparison, and inspection-performance evaluation.

This repository is deliberately **classical-vision first**: every acceptance decision can be traced to acquisition quality, preprocessing, segmentation, morphology, geometric measurements, and explicit recipe thresholds. The result is inspectable and debuggable, while leaving a clear seam for learned-vision experiments later.

## Architecture

```text
Camera / Image
      │
      ▼
┌──────────────────────┐
│ Acquisition QA       │ exposure / contrast / sharpness / clipping
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Registration / ROI   │ ECC / perspective rectification / ROI
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Preprocessing        │ grayscale / blur / illumination correction
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Segmentation         │ fixed / Otsu / adaptive threshold
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Morphology           │ opening / closing
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Region extraction    │ contours / connected components
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Metrology            │ geometry / shape / pixel-to-mm
└──────────┬───────────┘
           ├───────────────┐
           ▼               ▼
   Acceptance rules   Golden reference
           │          residual comparison
           └───────┬───────┘
                   ▼
             PASS / FAIL
             + evidence
```

## Implemented engineering layers

### Inspection recipe
`InspectionConfig` makes the pipeline explicit and auditable. It controls segmentation mode, illumination model, morphology, defect-size limits, shape limits, ROI, border handling, image-quality gating, and final acceptance rules.

### Acquisition-quality gating
Every inspection can measure mean intensity, contrast, Laplacian sharpness, and saturated-pixel fraction before treating a frame as valid evidence. A poor frame can be rejected rather than silently becoming a false PASS.

### Registration and rectification
`register_ecc()` provides intensity-based translation, Euclidean, affine, or homography registration. `rectify_perspective()` maps four ordered corners into a known measurement plane. Registration is an important seam between part pose variation and defect logic.

### Illumination normalization
Dark defects are enhanced with black-hat morphology and light defects with top-hat morphology. This is an explicit preprocessing stage rather than an implicit threshold-tuning trick.

### Segmentation
The pipeline supports fixed, Otsu, and adaptive Gaussian thresholding. OpenCV documents adaptive thresholding for locally varying intensity and connected-component labeling for region statistics; these primitives form the explainable segmentation layer here. citeturn516940search6turn516940search0

### Morphology
Optional opening and closing operations remove small artifacts and bridge/fill local structure before measurement. This follows standard morphology usage in image analysis. citeturn640270search2

### Metrology
Detected regions expose:

- pixel area and perimeter;
- centroid;
- bounding box;
- width/height and aspect ratio;
- circularity;
- extent;
- solidity;
- optional physical area/perimeter through `PixelScale`.

Region-property-based measurement is a standard inspection pattern; scikit-image documents `regionprops` and related tables for geometric region measurements. citeturn640270search0turn640270search4

### Camera calibration
`CameraCalibration` wraps OpenCV intrinsic calibration, distortion correction, persistence, and image-size validation. `PixelScale` provides a simple calibrated planar scale for converting pixel geometry to physical units.

### Golden-reference inspection
`compare_to_reference()` computes a blurred absolute residual between a golden image and an inspected image, thresholds the residual, filters connected components by area, and returns a residual mask plus change statistics. This is useful for stable, repeatable part appearance after registration.

### Inspection performance
`evaluation.py` provides part-level confusion matrices and derives precision, recall, specificity, F1, false-accept, and false-reject rates. This keeps **algorithm behavior** separate from **line-level inspection performance**.

### Visualization
`annotate_result()` and `save_annotated()` make rejection evidence reviewable by an operator or engineer rather than returning only a binary decision.

## Why lighting matters

Machine-vision accuracy is not only an algorithm problem. Lighting geometry, direction, color/wavelength, reflection, glare, and shadows strongly affect image repeatability. Industrial guidance treats lighting selection as a first-class design decision because a poor acquisition can overwhelm downstream image processing. citeturn516940search1

The code therefore treats acquisition quality and illumination normalization as explicit stages of the inspection system.

## Project structure

```text
industrial-vision-inspection/
├── .github/workflows/ci.yml
├── src/vision_inspection/
│   ├── __init__.py
│   ├── calibration.py
│   ├── cli.py
│   ├── evaluation.py
│   ├── inspector.py
│   ├── metrology.py
│   ├── preprocessing.py
│   ├── quality.py
│   ├── reference.py
│   ├── registration.py
│   └── visualization.py
├── tests/
│   ├── test_advanced.py
│   ├── test_components.py
│   └── test_inspector.py
├── pyproject.toml
└── README.md
```

## Run locally

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

python -m pip install -e .
python -m pip install pytest pytest-cov ruff

ruff check src tests
pytest --cov=vision_inspection --cov-report=term-missing -q
python -m compileall -q src tests
```

## CLI

```bash
inspect-image part.png \
  --segmentation otsu \
  --polarity dark \
  --min-area 25 \
  --max-area 100000 \
  --max-defects 0 \
  --annotated result.png
```

The CLI emits structured JSON containing acquisition quality, reject reasons, defect count, defect fraction, and defect geometry. Exit code `0` means PASS; exit code `1` means FAIL.

## Verification philosophy

The tests use synthetic gradients, controlled defect geometry, ROIs, shape constraints, acquisition failures, calibration persistence, registration, reference residuals, and evaluation labels. This is intentional: an industrial inspection repository needs regression tests around **failure modes and decision boundaries**, not only one happy-path image.

GitHub Actions runs Python 3.10, 3.11, and 3.12 with Ruff, coverage-backed pytest, and bytecode compilation.

## Remaining production-depth roadmap

The foundation is now strong enough that the next work should focus on validation against real manufacturing data rather than adding arbitrary algorithms:

1. camera drivers, triggers, exposure and lighting control;
2. flat-field/reference-image calibration and drift monitoring;
3. checkerboard/Charuco calibration workflows and reprojection-error reporting;
4. homography/telecentric metrology with uncertainty estimates;
5. golden-sample libraries and versioned inspection recipes;
6. gauge repeatability and reproducibility (GR&R-style) experiments;
7. labeled benchmark datasets with false-accept / false-reject targets;
8. learned defect models behind the same deterministic recipe/evidence boundary;
9. PLC/MES integration and cycle-time telemetry;
10. long-run process-capability monitoring and model/recipe change control.
