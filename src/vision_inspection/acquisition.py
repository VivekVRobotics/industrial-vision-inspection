"""Camera acquisition contracts and deterministic mock camera for inspection tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Frame:
    """Image plus acquisition metadata needed for traceability."""

    image: np.ndarray
    frame_id: int
    timestamp: float
    trigger_id: int | None = None
    exposure_us: float | None = None
    gain_db: float | None = None


class Camera(Protocol):
    """Minimal transport-independent camera contract."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def trigger(self, trigger_id: int | None = None) -> Frame: ...


class TriggeredSequenceCamera:
    """Deterministic camera for repeatable acquisition and control-flow tests."""

    def __init__(self, frames: list[np.ndarray], *, timestamps: list[float] | None = None):
        if not frames:
            raise ValueError("frames must not be empty")
        self._frames = [np.asarray(frame).copy() for frame in frames]
        if any(frame.ndim not in {2, 3} for frame in self._frames):
            raise ValueError("frames must be 2D or 3D images")
        self._timestamps = list(timestamps) if timestamps is not None else [float(i) for i in range(len(frames))]
        if len(self._timestamps) != len(self._frames) or any(not np.isfinite(t) for t in self._timestamps):
            raise ValueError("timestamps must match frames and be finite")
        if any(self._timestamps[i] <= self._timestamps[i - 1] for i in range(1, len(self._timestamps))):
            raise ValueError("timestamps must be strictly increasing")
        self._running = False
        self._index = 0

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def trigger(self, trigger_id: int | None = None) -> Frame:
        if not self._running:
            raise RuntimeError("camera acquisition is not started")
        if self._index >= len(self._frames):
            raise RuntimeError("frame sequence exhausted")
        idx = self._index
        self._index += 1
        return Frame(
            image=self._frames[idx].copy(),
            frame_id=idx,
            timestamp=self._timestamps[idx],
            trigger_id=trigger_id,
        )
