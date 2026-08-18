"""Command-line interface for the industrial inspection pipeline."""

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
    parser = argparse.ArgumentParser(description="Inspect an industrial image with an explainable CV recipe.")
    parser.add_argument("image", help="path to the image")
    parser.add_argument("--segmentation", choices=("fixed", "otsu", "adaptive"), default="otsu")
    parser.add_argument("--polarity", choices=("dark", "light"), default="dark")
    parser.add_argument("--threshold", type=int, default=80)
    parser.add_argument("--min-area", type=float, default=25)
    parser.add_argument("--max-area", type=float, default=100000)
    parser.add_argument("--max-defects", type=int, default=0)
    parser.add_argument("--adaptive-block", type=int, default=31)
    parser.add_argument("--adaptive-c", type=float, default=3.0)
    parser.add_argument("--opening", type=int, default=0)
    parser.add_argument("--closing", type=int, default=0)
    parser.add_argument("--background-kernel", type=int, default=51)
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--min-circularity", type=float)
    parser.add_argument("--max-aspect-ratio", type=float)
    parser.add_argument("--min-solidity", type=float)
    parser.add_argument("--allow-border-touch", action="store_true")
    parser.add_argument("--allow-bad-quality", action="store_true")
    parser.add_argument("--annotated", type=Path, help="optional output path for an annotated image")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = InspectionConfig(
        preprocess=PreprocessConfig(
            blur_kernel=args.blur_kernel,
            background_kernel=args.background_kernel,
            polarity=args.polarity,
        ),
        quality=QualityConfig(),
        segmentation_mode=args.segmentation,
        threshold=args.threshold,
        adaptive_block=args.adaptive_block,
        adaptive_c=args.adaptive_c,
        opening=args.opening,
        closing=args.closing,
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
        "reject_reasons": result.reject_reasons,
        "image_shape": result.image_shape,
        "quality": {
            "passed": result.quality.passed,
            "mean": result.quality.mean,
            "std": result.quality.std,
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
                "centroid_px": d.measurement.centroid_px,
                "bbox_px": d.measurement.bbox_px,
                "width_px": d.measurement.width_px,
                "height_px": d.measurement.height_px,
                "aspect_ratio": d.measurement.aspect_ratio,
                "circularity": d.measurement.circularity,
                "extent": d.measurement.extent,
                "solidity": d.measurement.solidity,
            }
            for d in result.defects
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
