"""Versioned golden-sample metadata and integrity checks.

Golden images are controlled inspection artifacts. This module records the
content hash, recipe identity, provenance, and approval metadata so a change to
either the image or the recipe becomes detectable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class GoldenSample:
    """Immutable metadata for an approved image/recipe pairing."""

    sample_id: str
    image_sha256: str
    recipe_version: str
    created_utc: str
    notes: str = ""
    approved_by: str = ""

    @classmethod
    def create(
        cls,
        sample_id: str,
        image_bytes: bytes,
        recipe_version: str,
        *,
        notes: str = "",
        approved_by: str = "",
    ) -> "GoldenSample":
        if not sample_id or not recipe_version:
            raise ValueError("sample_id and recipe_version must be non-empty")
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise ValueError("image_bytes must be non-empty bytes")
        return cls(
            sample_id=sample_id,
            image_sha256=hashlib.sha256(image_bytes).hexdigest(),
            recipe_version=recipe_version,
            created_utc=datetime.now(timezone.utc).isoformat(),
            notes=notes,
            approved_by=approved_by,
        )

    def verify_bytes(self, image_bytes: bytes) -> bool:
        """Return True when image bytes match the recorded golden digest."""
        return hashlib.sha256(image_bytes).hexdigest() == self.image_sha256


def registry_digest(samples: list[GoldenSample]) -> str:
    """Return a deterministic digest of registry contents for audit trails."""
    records = [asdict(sample) for sample in samples]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_golden_registry(samples: list[GoldenSample], path: str | Path) -> None:
    """Persist golden metadata and a registry digest as deterministic JSON."""
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("sample_id values must be unique")
    records = [asdict(sample) for sample in samples]
    document = {"samples": records, "registry_sha256": registry_digest(samples)}
    Path(path).write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")


def load_golden_registry(path: str | Path) -> list[GoldenSample]:
    """Load and integrity-check a golden registry."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(document, list):
        records = document
        expected_digest = None
    elif isinstance(document, dict):
        records = document.get("samples")
        expected_digest = document.get("registry_sha256")
    else:
        raise ValueError("golden registry must contain a sample list or registry document")
    if not isinstance(records, list):
        raise ValueError("golden registry samples must be a list")
    samples = [GoldenSample(**record) for record in records]
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("golden registry contains duplicate sample_id values")
    if expected_digest is not None and expected_digest != registry_digest(samples):
        raise ValueError("golden registry integrity digest does not match contents")
    return samples
