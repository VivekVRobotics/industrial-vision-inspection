# Industrial Vision Inspection

An explainable, recipe-driven machine-vision inspection toolkit for industrial surface defects, geometric measurement, acquisition quality, calibration, measurement-system analysis, process drift, and inspection-performance benchmarking.

This repository is deliberately **classical-vision first**: every acceptance decision can be traced to image quality, preprocessing, segmentation, morphology, metrology, and explicit recipe thresholds. That keeps failure analysis transparent and provides a controlled foundation for later learned-vision experiments.

## System architecture

```text
Camera / trigger
      │
      ▼
Frame + acquisition metadata
      │
      ▼
Image-quality gate
      │
      ▼
Calibration / flat-field / registration
      │
      ▼
ROI + preprocessing
      │
      ▼
Segmentation + morphology
      │
      ▼
Region extraction + metrology
      │
      ├───────────────┐
      ▼               ▼
Rule-based QA     Golden reference
      │               │
      └───────┬───────┘
              ▼
       PASS / FAIL + evidence
              │
      ┌───────┴────────┐
      ▼                ▼
MSA / GR&R        Drift / benchmark
```

## Implemented engineering layers

### Camera acquisition and triggering
`Camera` defines a transport-independent acquisition boundary. `TriggeredSequenceCamera` gives deterministic trigger/frame/timestamp behavior for tests and HIL-style control logic. This mirrors modern industrial acquisition systems where triggered capture is a normal operating mode. citeturn400041search3turn400041search12

### Acquisition-quality gating
Images can be rejected as invalid evidence when exposure, contrast, sharpness, or clipping metrics fall outside the recipe. This prevents the system from confusing an unusable frame with a defect-free part.

### Flat-field calibration
`build_flat_field()` and `apply_flat_field()` model multiplicative illumination variation from uniform reference frames. The intent is to stabilize the image before defect segmentation rather than forcing segmentation to compensate for illumination artifacts.

### Camera calibration
`CameraCalibration` supports intrinsic calibration, distortion correction, persistence, and image-size validation. `PixelScale` supports planar physical measurements.

`calibrate_charuco()` adds a ChArUco workflow with intrinsic standard deviations and per-view reprojection errors. Current OpenCV documentation recommends ChArUco corners for calibration because they are more accurate than raw marker corners and can tolerate partial board views. citeturn382627search0turn382627search7

### Metrology uncertainty
`propagate_length_uncertainty()` explicitly propagates pixel-location and pixel-scale uncertainty into a standard uncertainty and an expanded uncertainty. It is intentionally a transparent first-order model rather than a claim of full ISO/GUM compliance.

### Registration / pose normalization
ECC-based registration and perspective rectification reduce false defects caused by small part pose changes.

### Illumination normalization and segmentation
The preprocessing layer supports dark-defect black-hat enhancement, light-defect top-hat enhancement, fixed thresholding, Otsu thresholding, and adaptive Gaussian thresholding, followed by explicit opening/closing morphology.

### Geometric metrology
Detected regions expose area, perimeter, centroid, bounding box, aspect ratio, circularity, extent, solidity, and optional calibrated physical dimensions.

### Golden-sample versioning
`GoldenSample` stores a SHA-256 digest of the approved image bytes together with recipe version and creation timestamp. This makes golden references traceable instead of silently replacing them in-place.

### Reference-difference inspection
Golden-reference residuals provide a second, explainable inspection path for appearance changes that may not fit a fixed defect geometry rule.

### Measurement-system analysis
`crossed_grr()` estimates repeatability, reproducibility, part-to-part variation, total Gage R&R variation, study variation, and number of distinct categories from a balanced `[part, operator, repeat]` study. NIST treats repeatability, reproducibility, stability, bias, and drift as core measurement-process characterization concerns. citeturn400041search1turn400041search8

The repository deliberately labels this as a screening implementation: a production metrology validation still needs an approved study design, representative parts, controlled operators/conditions, and an appropriate uncertainty budget. Crossed Gage R&R studies are commonly structured with every operator measuring every part repeatedly. citeturn400041search6turn400041search10

### Process drift
`process_drift()` provides a lightweight rolling z-score alarm for measurement streams so calibration/inspection outputs can be monitored over time.

### Inspection-performance benchmarks
`InspectionBenchmark` separates false accepts, false rejects, precision, recall, F1, and latency. This is the layer where a detector becomes an inspection-system measurement rather than just an algorithm score.

## Project structure

```text
industrial-vision-inspection/
├── .github/workflows/ci.yml
├── src/vision_inspection/
│   ├── acquisition.py
│   ├── benchmarks.py
│   ├── calibration.py
│   ├── cli.py
│   ├── evaluation.py
│   ├── flat_field.py
│   ├── golden.py
│   ├── inspector.py
│   ├── measurement_system.py
│   ├── metrology.py
│   ├── preprocessing.py
│   ├── quality.py
│   ├── reference.py
│   ├── registration.py
│   ├── uncertainty.py
│   └── visualization.py
├── tests/
│   ├── test_components.py
│   ├── test_industrial_validation.py
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

The CLI emits structured JSON containing image-quality status, reject reasons, defect count, defect fraction, and defect geometry. Exit code `0` means PASS; exit code `1` means FAIL.

## Verification philosophy

Tests use synthetic images and controlled numeric datasets covering segmentation boundaries, acquisition failures, flat-field behavior, calibration persistence, uncertainty propagation, reference residuals, registration, measurement-system analysis, drift, golden-reference traceability, and benchmark metrics.

The CI matrix runs Python 3.10, 3.11, and 3.12 with Ruff, coverage-backed pytest, and bytecode compilation.

## Production-depth roadmap

1. hardware-specific camera adapters (GenICam/vendor SDK) and trigger/encoder synchronization;
2. validated flat-field and exposure calibration procedures;
3. production ChArUco/checkerboard calibration capture tooling;
4. calibrated multi-plane metrology and full uncertainty budgets;
5. versioned recipe/golden approval workflow and audit trail;
6. formal crossed/expanded Gage R&R and long-term stability studies;
7. process control charts and alarm policies beyond rolling z-scores;
8. benchmark datasets with false-accept / false-reject targets, latency budgets, and confidence intervals;
9. PLC/MES result interfaces, cycle-time telemetry, and traceability;
10. learned defect models evaluated against the classical baseline under identical test protocols.
