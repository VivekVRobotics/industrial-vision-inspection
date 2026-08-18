# Architecture

## Design objective

The repository models industrial inspection as a deterministic measurement pipeline rather than a single image classifier.

```text
Camera / trigger
      |
      v
Frame + metadata
      |
      v
Acquisition-quality gate
      |
      +--> reject unusable evidence
      |
      v
Calibration / flat-field / registration
      |
      v
ROI + preprocessing
      |
      v
Segmentation + morphology
      |
      v
Region metrology
      |
      +--> geometric evidence
      |
      v
Acceptance rules
      |
      +-------------------+
      |                   |
      v                   v
PASS / FAIL         Golden reference
      |                   |
      +---------+---------+
                |
                v
       Traceable result
                |
        +-------+-------+
        |               |
        v               v
       MSA          Drift/benchmark
```

## Stable boundaries

### Acquisition boundary
`Camera` is a transport-independent protocol. Vendor SDKs, GenICam adapters, PLC trigger logic, and encoder synchronization belong outside the core inspection package. `Frame` carries immutable image data and acquisition metadata.

### Inspection boundary
`inspect_array()` is the primary deterministic API. `inspect_image()` adds file I/O only. This separation keeps unit tests independent of filesystem behavior.

### Metrology boundary
`measure_regions()` turns segmentation masks into explicit geometric measurements. Physical conversion is performed only when the caller supplies a validated `PixelScale`.

### Validation boundary
Classification metrics, benchmark latency, Gage R&R, uncertainty, stability, and drift operate on numerical evidence and are independent from the image-processing implementation.

## Traceability model

Every `InspectionResult` records:

- recipe version;
- recipe SHA-256;
- image SHA-256;
- processing time;
- image-quality measurements;
- acceptance reject reasons;
- diagnostic filtering reasons;
- geometric measurements.

This allows a downstream system to correlate a decision with the exact software recipe and image bytes used to generate it.

## Determinism

The core modules avoid global mutable state. Calibration artifacts, golden references, and test camera sources are explicit inputs. Randomized analyses use explicit seeds where applicable.

## Production adapter rule

Vendor camera access, PLC/MES transport, database persistence, and UI behavior should consume the public API rather than reimplementing segmentation or acceptance logic. This preserves one source of truth for the inspection decision.
