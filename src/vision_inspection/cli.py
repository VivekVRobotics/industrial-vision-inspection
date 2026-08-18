"""Command-line interface for the inspection pipeline."""

import argparse
import json

from .inspector import InspectionConfig, inspect_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect an industrial image for localized dark defects.")
    parser.add_argument("image", help="path to the image")
    parser.add_argument("--threshold", type=int, default=80)
    parser.add_argument("--min-area", type=int, default=25)
    parser.add_argument("--max-area", type=int, default=100000)
    args = parser.parse_args()

    result = inspect_image(
        args.image,
        InspectionConfig(threshold=args.threshold, min_area=args.min_area, max_area=args.max_area),
    )
    print(
        json.dumps(
            {
                "passed": result.passed,
                "defect_count": result.defect_count,
                "defect_fraction": result.defect_fraction,
                "defects": [
                    {"area": d.area, "bbox": d.bbox}
                    for d in result.defects
                ],
            },
            indent=2,
        )
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
