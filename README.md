# Industrial Vision Inspection

An explainable, recipe-driven machine-vision inspection toolkit for industrial surface defects, geometric measurement, acquisition-quality gating, camera calibration, and inspection-performance evaluation.

This repository is deliberately **classical-vision first**: every acceptance decision can be traced to image quality, preprocessing, segmentation, morphology, geometric measurements, and explicit recipe thresholds. That makes failures inspectable and makes the system useful as a foundation for later learned-vision experiments rather than hiding the process inside a black-box classifier.

## Architecture

```text
Camera / Image
      │
      ▼
┌──────────────────────┐
│ Image-quality gate   │  exposure / contrast / sharpness / clipping
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ ROI + preprocessing  │  grayscale / blur / illumination correction
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Segmentation         │  fixed / Otsu / adaptive threshold
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Morphology           │  opening / closing
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Region extraction    │  contours / geometry
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Metrology            │  area / perimeter / bbox / shape / mm conversion
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Acceptance recipe    │  defect count / area / shape / border rules
└──────────┬───────────┘
           ▼
     PASS / FAIL + evidence
```

## Implemented engineering layers

### Inspection recipe
`InspectionConfig` makes the pipeline explicit and auditable. It controls segmentation mode, illumination model, morphology, defect-size limits, shape limits, ROI, border handling, quality gating, and the final acceptance envelope.

### Acquisition-quality gating
Every inspection can measure mean intensity, contrast, Laplacian sharpness, and saturated-pixel fraction before accepting the image as valid evidence. A poor frame can be rejected instead of being misclassified as a good part.

### Illumination normalization
The preprocessing layer supports dark-defect enhancement with black-hat morphology and light-defect enhancement with white/top-hat morphology. This reduces dependence on perfectly uniform background illumination and creates a cleaner segmentation signal.

### Segmentation
The pipeline supports:

- fixed thresholding;
- Otsu automatic thresholding;
- adaptive Gaussian thresholding.

OpenCV documents adaptive thresholding for locally varying intensity and connected-component analysis for labeled region statistics; these primitives are used here as explainable building blocks. citeturn516940search6turn516940search0

### Morphology
Optional opening and closing operations provide explicit removal/fill behavior before region measurement.

### Metrology
Detected regions expose:

- pixel area and perimeter;
- centroid;
- bounding box;
- width/height and aspect ratio;
- circularity;
- extent;
- solidity;
- optional physical area/perimeter through a pixel scale.

Region-property-based measurement is a standard image-analysis pattern; scikit-image documents regionprops for these kinds of geometric measurements. citeturn640270search0turn640270search4

### Camera calibration
`CameraCalibration` wraps OpenCV intrinsic calibration, distortion correction, persistence, and image-size validation. `PixelScale` converts pixel lengths/areas into physical units for planar inspection.

### Inspection performance
`evaluation.py` provides part-level confusion matrices and derives precision, recall, specificity, F1, false-accept, and false-reject rates. This separates **algorithm correctness** from **inspection-system performance**.

## Why lighting matters

Machine-vision accuracy is not only an algorithm problem. Lighting geometry, direction, color/wavelength, reflection, glare, and shadows directly affect image repeatability. Industrial guidance commonly treats lighting selection as a first-class design decision rather than something to solve after segmentation. citeturn516940search1

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
│   └── visualization.py
├── tests/
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

The tests use synthetic images with controlled gradients, defect geometry, ROIs, shape constraints, acquisition failures, calibration persistence, and evaluation labels. This is intentional: an industrial inspection repository needs regression tests around **failure modes and decision boundaries**, not only one happy-path image.

The GitHub Actions matrix runs Python 3.10, 3.11, and 3.12 with Ruff, coverage-backed pytest, and bytecode compilation.

## Roadmap

The next production-depth layers are deliberately separate from the current classical baseline:

1. camera drivers and trigger/exposure control;
2. flat-field and reference-image calibration;
3. geometric camera calibration using checkerboards/Charuco;
4. perspective rectification and homography-based metrology;
5. template/pose registration so part movement does not create false defects;
6. measurement tolerances and gauge-repeatability studies;
7. golden-sample / process-drift monitoring;
8. dataset-backed learned defect models with classical pre/post filters;
9. PLC/MES result interfaces and cycle-time telemetry;
10. benchmark datasets with false-accept / false-reject targets and latency budgets.
