"""Versioned golden-sample metadata and inspection drift baselines."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class GoldenSample:
    """Immutable metadata for a validated golden image/recipe pairing."""

    sample_id: str
    image_sha256: str
    recipe_version: str
    created_utc: str
    notes: str = ""

    @classmethod
    def create(cls, sample_id: str, image_bytes: bytes, recipe_version: str, notes: str = "") -> "GoldenSample":
        if not sample_id or not recipe_version:
            raise ValueError("sample_id and recipe_version must be non-empty")
        digest = hashlib.sha256(image_bytes).hexdigest()
        return cls(sample_id, digest, recipe_version, datetime.now(timezone.utc).isoformat(), notes)


def save_golden_registry(samples: list[GoldenSample], path: str | Path) -> None:
    """Persist golden metadata as deterministic JSON."""
    records = [asdict(sample) for sample in samples]
    Path(path).write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")


def load_golden_registry(path: str | Path) -> list[GoldenSample]:
    """Load validated golden metadata."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("golden registry must contain a list")
    return [GoldenSample(**record) for record in records]
