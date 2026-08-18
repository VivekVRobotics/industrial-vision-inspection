"""Command-line entry point for traceable inspection runs.

The CLI is intentionally a thin adapter around the public inspection API. It
serializes machine-readable evidence and optionally writes an operator-review
annotation; it does not implement image-processing logic itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .inspector import InspectionConfig, inspect_image
from .preprocessing import PreprocessConfig
from .quality import QualityConfig
from .visualization import save_annotated


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a deterministic industrial CV inspection recipe.")
    parser.add_argument("image", help="input image path")
    parser.add_argument("--recipe-version", default="0.4.0")
    parser.add_argument("--segmentation", choices=("fixed", "otsu", "adaptive"), default="otsu")
    parser.add_argument("--threshold", type=int, default=80)
    parser.add_argument("--segmentation-polarity", choices=("positive", "negative"), default="positive")
    parser.add_argument("--polarity", choices=("dark", "light"), default="dark", help="defect polarity for illumination normalization")
    parser.add_argument("--min-area", type=float, default=25)
    parser.add_argument("--max-area", type=float, default=100000)
    parser.add_argument("--max-defects", type=int, default=0)
    parser.add_argument("--adaptive-block", type=int, default=31)
    parser.add_argument("--adaptive-c", type=float, default=3.0)
    parser.add_argument("--opening", type=int, default=0)
    parser.add_argument("--closing", type=int, default=0)
    parser.add_argument("--morphology-iterations", type=int, default=1)
    parser.add_argument("--background-kernel", type=int, default=51)
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--min-circularity", type=float)
    parser.add_argument("--max-aspect-ratio", type=float)
    parser.add_argument("--min-solidity", type=float)
    parser.add_argument("--allow-border-touch", action="store_true")
    parser.add_argument("--allow-bad-quality", action="store_true")
    parser.add_argument("--annotated", type=Path, help="optional evidence image path")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse command-line options, execute inspection, and print JSON evidence."""
    args = _build_parser().parse_args(argv)
    config = InspectionConfig(
        version=args.recipe_version,
        preprocess=PreprocessConfig(
            blur_kernel=args.blur_kernel,
            background_kernel=args.background_kernel,
            polarity=args.polarity,
        ),
        quality=QualityConfig(),
        segmentation_mode=args.segmentation,
        threshold=args.threshold,
        segmentation_polarity=args.segmentation_polarity,
        adaptive_block=args.adaptive_block,
        adaptive_c=args.adaptive_c,
        opening=args.opening,
        closing=args.closing,
        morphology_iterations=args.morphology_iterations,
        min_area_px=args.min_area,
        max_area_px=args.max_area,
        max_defects=args.max_defects,
        min_circularity=args.min_circularity,
        max_aspect_ratio=args.max_aspect_ratio,
        min_solidity=args.min_solidity,
        reject_border_touching=not args.allow_border_touch,
        reject_bad_image_quality=not args.allow_bad_quality,
    )
    result = inspect_image(args.image, config)

    if args.annotated:
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(args.image)
        save_annotated(image, result, args.annotated)

    payload = {
        "passed": result.passed,
        "recipe_version": result.recipe_version,
        "recipe_sha256": result.recipe_sha256,
        "image_sha256": result.image_sha256,
        "processing_ms": result.processing_ms,
        "reject_reasons": result.reject_reasons,
        "diagnostic_reasons": result.diagnostic_reasons,
        "image_shape": result.image_shape,
        "quality": {
            "passed": result.quality.passed,
            "mean": result.quality.mean,
            "std": result.quality.std,
            "p1": result.quality.p1,
            "p99": result.quality.p99,
            "sharpness": result.quality.sharpness,
            "saturated_fraction": result.quality.saturated_fraction,
            "failures": result.quality.failures,
        },
        "defect_count": result.defect_count,
        "defect_fraction": result.defect_fraction,
        "defects": [
            {
                "area_px": d.measurement.area_px,
                "area_physical": d.measurement.area_physical,
                "perimeter_px": d.measurement.perimeter_px,
                "perimeter_physical": d.measurement.perimeter_physical,
                "centroid_px": d.measurement.centroid_px,
                "bbox_px": d.measurement.bbox_px,
                "width_px": d.measurement.width_px,
                "height_px": d.measurement.height_px,
                "equivalent_diameter_px": d.measurement.equivalent_diameter_px,
                "equivalent_diameter_physical": d.measurement.equivalent_diameter_physical,
                "aspect_ratio": d.measurement.aspect_ratio,
                "circularity": d.measurement.circularity,
                "extent": d.measurement.extent,
                "solidity": d.measurement.solidity,
                "min_rect_angle_deg": d.measurement.min_rect_angle_deg,
            }
            for d in result.defects
        ],
    }
    print(json.dumps(payload, indent=2, default=list))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
