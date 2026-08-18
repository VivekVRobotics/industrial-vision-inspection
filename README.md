# Industrial Vision Inspection

A configurable, explainable computer-vision pipeline for detecting localized surface defects in industrial images.

## What this project demonstrates

- Image acquisition from disk with explicit failure handling.
- Grayscale conversion and Gaussian denoising.
- Threshold-based defect segmentation.
- Connected-component analysis for defect measurements.
- Configurable ROI and defect-area limits.
- Structured inspection results with pass/fail status, defect count, area, and bounding boxes.
- JSON command-line output suitable for a manufacturing/test pipeline.
- Automated tests using synthetic inspection images.
- GitHub Actions CI on pushes and pull requests.

## Pipeline

```text
Image
  │
  ├── grayscale
  ├── Gaussian blur
  ├── threshold
  ├── connected components
  ├── area filtering
  └── inspection result
       ├── PASS / FAIL
       ├── defect count
       ├── defect fraction
       └── defect bounding boxes
```

## Run locally

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -e . pytest
pytest -q
```

Inspect an image:

```bash
inspect-image path/to/part.png --threshold 80 --min-area 25 --max-area 100000
```

The CLI exits with code `0` for PASS and `1` for FAIL, making it usable from a larger automation or quality-control process.

## Design notes

This is intentionally a classical-vision baseline rather than an opaque ML classifier. The decision path is inspectable and tunable, which makes it useful for understanding industrial inspection fundamentals: segmentation, ROI selection, morphology/connected components, measurement, and deterministic acceptance criteria.

A future production layer could add camera acquisition, illumination normalization, calibration, annotated result images, PLC/MES integration, and a labeled dataset for learned defect classification.
