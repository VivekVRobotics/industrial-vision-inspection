"""Transport-independent industrial camera contracts.

The module deliberately stops at the camera boundary. A production adapter can
wrap a GenICam/vendor SDK without coupling the inspection engine to a transport
library. ``Frame`` carries the minimum metadata required to correlate an image
with a trigger, exposure settings, and a monotonically ordered acquisition
stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Frame:
    """Immutable image packet produced by a camera.

    ``timestamp`` is expressed in the camera/adapter clock domain. If a system
    combines camera and PLC clocks, clock synchronization belongs in the
    acquisition adapter rather than being silently inferred here.
    """

    image: np.ndarray
    frame_id: int
    timestamp: float
    trigger_id: int | None = None
    exposure_us: float | None = None
    gain_db: float | None = None

    def __post_init__(self) -> None:
        image = np.asarray(self.image)
        if image.ndim not in {2, 3} or image.size == 0:
            raise ValueError("frame image must be a non-empty 2D/3D array")
        if self.frame_id < 0 or not np.isfinite(self.timestamp):
            raise ValueError("frame_id must be non-negative and timestamp finite")
        if self.exposure_us is not None and (not np.isfinite(self.exposure_us) or self.exposure_us <= 0):
            raise ValueError("exposure_us must be positive and finite")
        if self.gain_db is not None and not np.isfinite(self.gain_db):
            raise ValueError("gain_db must be finite")
        object.__setattr__(self, "image", image.copy())


class Camera(Protocol):
    """Minimal adapter boundary for triggered industrial cameras."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def trigger(self, trigger_id: int | None = None) -> Frame: ...


@dataclass(frozen=True)
class AcquisitionStats:
    """Snapshot of deterministic sequence-camera state for test assertions."""

    emitted_frames: int
    remaining_frames: int
    running: bool


class TriggeredSequenceCamera:
    """Deterministic triggered source used for integration and HIL-style tests.

    The source preserves frame order, rejects use before ``start()``, rejects
    sequence exhaustion, and optionally exposes acquisition metadata. It is not
    a physical-camera simulator: exposure, trigger latency, transport jitter,
    and sensor timing must be modeled by a hardware adapter or dedicated test
    fixture when those effects matter.
    """

    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        timestamps: list[float] | None = None,
        exposures_us: list[float | None] | None = None,
        gains_db: list[float | None] | None = None,
    ) -> None:
        if not frames:
            raise ValueError("frames must not be empty")
        self._frames = [np.asarray(frame).copy() for frame in frames]
        if any(frame.ndim not in {2, 3} or frame.size == 0 for frame in self._frames):
            raise ValueError("frames must be non-empty 2D or 3D images")
        shape = self._frames[0].shape
        if any(frame.shape != shape for frame in self._frames):
            raise ValueError("all frames must have the same shape")
        self._timestamps = list(timestamps) if timestamps is not None else [float(i) for i in range(len(frames))]
        if len(self._timestamps) != len(self._frames) or any(not np.isfinite(t) for t in self._timestamps):
            raise ValueError("timestamps must match frames and be finite")
        if any(self._timestamps[i] <= self._timestamps[i - 1] for i in range(1, len(self._timestamps))):
            raise ValueError("timestamps must be strictly increasing")
        self._exposures = [None] * len(frames) if exposures_us is None else list(exposures_us)
        self._gains = [None] * len(frames) if gains_db is None else list(gains_db)
        if len(self._exposures) != len(frames) or len(self._gains) != len(frames):
            raise ValueError("acquisition metadata lists must match frame count")
        self._running = False
        self._index = 0

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def reset(self) -> None:
        """Return the deterministic source to its initial frame."""
        self._index = 0
        self._running = False

    @property
    def stats(self) -> AcquisitionStats:
        """Return non-mutating acquisition counters for diagnostics/tests."""
        return AcquisitionStats(self._index, len(self._frames) - self._index, self._running)

    def trigger(self, trigger_id: int | None = None) -> Frame:
        if not self._running:
            raise RuntimeError("camera acquisition is not started")
        if self._index >= len(self._frames):
            raise RuntimeError("frame sequence exhausted")
        idx = self._index
        self._index += 1
        return Frame(
            image=self._frames[idx],
            frame_id=idx,
            timestamp=self._timestamps[idx],
            trigger_id=trigger_id,
            exposure_us=self._exposures[idx],
            gain_db=self._gains[idx],
        )
