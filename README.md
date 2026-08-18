# Industrial Vision Inspection

A traceable, explainable machine-vision inspection toolkit for industrial surface defects, metrology, calibration, acquisition quality, measurement-system analysis, process drift, and statistical benchmarking.

The project is **classical-vision first**. Inspection decisions remain decomposable into acquisition quality, calibration/preprocessing, segmentation, morphology, geometry, and explicit acceptance rules. Learned vision can be added later and benchmarked against the same locked evaluation protocol.

## Architecture

```text
Camera / trigger
      |
      v
Frame + acquisition metadata
      |
      v
Acquisition-quality gate
      |
      v
Calibration / flat-field / registration
      |
      v
ROI + illumination normalization
      |
      v
Segmentation + morphology
      |
      v
Region metrology
      |
      +--------------------+
      |                    |
      v                    v
Rule-based acceptance   Golden reference
      |                    |
      +---------+----------+
                v
        PASS / FAIL + evidence
                |
       +--------+---------+
       v                  v
Measurement system    Benchmark / drift
```

## What makes this different from a demo detector

The package treats an inspection decision as a **measurement and evidence problem**.

An `InspectionResult` records:

- recipe version and recipe SHA-256;
- source-image SHA-256;
- image-quality metrics and failure reasons;
- diagnostic candidate filtering;
- defect count and area fraction;
- geometry and calibrated physical measurements;
- processing time.

That makes a decision reproducible and auditable instead of reducing it to a boolean.

## Production-oriented modules

### Acquisition
`acquisition.py` defines a transport-independent camera protocol plus a deterministic triggered source for integration tests. Frames carry trigger, timestamp, exposure, gain, and frame identity.

The production boundary is intentionally vendor-neutral. Hardware SDKs, PLC synchronization, encoder capture, and GenICam/vendor transports belong in adapters rather than the numerical inspection core.

### Image-quality gating
`quality.py` measures mean, standard deviation, 1st/99th percentiles, Laplacian sharpness, and saturation. A poor frame can be rejected as invalid evidence before a defect-free decision is allowed.

### Camera calibration
`calibration.py` supports:

- pinhole intrinsics;
- distortion;
- extended calibration diagnostics;
- per-view reprojection errors;
- intrinsic parameter standard deviations;
- quality gates;
- persistent calibration archives;
- ChArUco calibration;
- planar pixel-scale conversion.

OpenCV documents standard calibration and ChArUco workflows, including calibration error diagnostics. See `docs/research-notes.md` for references.

### Flat-field correction
`flat_field.py` builds a normalized illumination field from repeated uniform captures, supports robust median aggregation, reports field non-uniformity, smooths the calibration field, and applies multiplicative correction to 2D/3D images.

### Registration
`registration.py` supports translation, Euclidean, affine, and homography ECC alignment, optional image pyramids, minimum-correlation gates, and four-corner perspective rectification.

### Preprocessing and segmentation
`preprocessing.py` supports:

- grayscale conversion;
- Gaussian or median denoising;
- black-hat / top-hat illumination enhancement;
- optional CLAHE;
- fixed thresholding;
- Otsu thresholding;
- adaptive Gaussian thresholding;
- explicit segmentation polarity;
- opening/closing morphology with controlled iteration count.

### Metrology
`metrology.py` reports:

- area/perimeter;
- centroid/bounding box;
- aspect ratio;
- circularity;
- extent;
- solidity;
- equivalent diameter;
- minimum-area rectangle dimensions and angle;
- compactness;
- optional physical dimensions.

### Reference inspection
`reference.py` compares a registered part against a golden image and reports thresholded localized residuals plus mean/P95/max absolute difference statistics.

### Golden-sample governance
`golden.py` records image hashes, recipe versions, approval metadata, creation time, and a registry digest. Stored registries are integrity-checked when loaded.

### Uncertainty
`uncertainty.py` exposes:

- first-order root-sum-square propagation;
- Monte Carlo propagation;
- standard and expanded uncertainty;
- lower/upper reporting bounds;
- explicit method labels.

The implementation deliberately does **not** claim a complete uncertainty budget. Distortion residuals, fixture motion, target uncertainty, segmentation bias, temperature, and correlated inputs remain application-specific contributors.

### Measurement-system analysis
`measurement_system.py` provides a balanced crossed Gage R&R screening model with:

- repeatability;
- operator/reproducibility variation;
- interaction;
- part-to-part variation;
- total Gage R&R;
- percent Gage R&R of study variation;
- NDC screening;
- rolling drift detection;
- simple stability trend estimation.

The module is explicitly a screening implementation; formal release should follow the site's approved MSA procedure.

### Benchmarking
`benchmarks.py` and `evaluation.py` separate classification performance from latency/system performance. They provide:

- TP/FP/TN/FN;
- precision, recall, specificity;
- false-accept and false-reject rates;
- F1, balanced accuracy, Matthews correlation;
- Wilson confidence intervals;
- mean/P95/P99 latency;
- throughput estimate.

## Project layout

```text
industrial-vision-inspection/
├── .github/workflows/ci.yml
├── docs/
│   ├── architecture.md
│   ├── module-reference.md
│   ├── research-notes.md
│   ├── uncertainty.md
│   └── validation-protocol.md
├── src/vision_inspection/
│   ├── __init__.py
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
│   ├── test_advanced.py
│   ├── test_components.py
│   ├── test_industrial_validation.py
│   └── test_inspector.py
├── pyproject.toml
└── README.md
```

## Local development

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\\Scripts\\activate

python -m pip install -e .
python -m pip install pytest pytest-cov ruff

ruff check src tests
pytest --cov=vision_inspection --cov-report=term-missing -q
python -m compileall -q src tests
```

## CLI example

```bash
inspect-image part.png \\
  --recipe-version 0.4.0 \\
  --segmentation otsu \\
  --segmentation-polarity positive \\
  --polarity dark \\
  --min-area 25 \\
  --max-area 100000 \\
  --max-defects 0 \\
  --annotated result.png
```

The CLI emits structured JSON with recipe/image identity, image-quality evidence, reject/diagnostic reasons, processing time, and defect geometry. Exit code `0` is PASS; `1` is FAIL.

## Validation protocol

Software tests are only one layer. A production release should also demonstrate:

1. acquisition timing and trigger integrity;
2. calibration quality across representative poses;
3. flat-field stability under production illumination;
4. metrology bias/repeatability and uncertainty;
5. crossed Gage R&R or the organization's approved MSA procedure;
6. locked-set false-accept/false-reject performance with confidence intervals;
7. cycle-time and tail-latency compliance;
8. golden-reference and recipe governance;
9. long-term stability/drift monitoring.

See `docs/validation-protocol.md`.

## Research basis

The architecture is informed by current OpenCV calibration/registration documentation, MVTec HALCON's separation of acquisition/calibration/inspection/metrology/matching concerns, NIST measurement-system guidance, and ISO 5725's distinction between trueness and precision. The detailed source list is maintained in `docs/research-notes.md`.

## Current scope and limits

This repository does not pretend to be a production camera SDK, PLC driver, certified metrology system, or complete MSA package. Those are integration and validation boundaries. The core objective is to provide a rigorous, testable reference implementation on which those systems can be built and benchmarked.
