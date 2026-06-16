from __future__ import annotations

import enum
from dataclasses import dataclass, field


class MatchType(enum.Enum):
    BYTE = "byte"
    TRANSCODE = "transcode"
    NO_MATCH = "no_match"


class Subtype(enum.Enum):
    STRAIGHT = "straight"
    SUBSET = "subset"
    INTERLEAVED = "interleaved"


class SegmentStatus(enum.Enum):
    MATCH = "match"
    A_ONLY = "a_only"
    B_ONLY = "b_only"


@dataclass
class Segment:
    a_start: float | None
    a_end: float | None
    b_start: float | None
    b_end: float | None
    status: SegmentStatus
    micro: bool = False  # True for divergences < ~5s (shown in muted color)

    def to_dict(self) -> dict:
        d = {
            "a_start": self.a_start,
            "a_end": self.a_end,
            "b_start": self.b_start,
            "b_end": self.b_end,
            "status": self.status.value,
        }
        if self.micro:
            d["micro"] = True
        return d


@dataclass
class ComparisonResult:
    match_type: MatchType
    subtype: Subtype | None
    video_segments: list[Segment] = field(default_factory=list)
    audio_segments: list[Segment] = field(default_factory=list)
    video_available: bool = True
    audio_available: bool = True
    a_duration: float = 0.0
    b_duration: float = 0.0
    subset_direction: str | None = None  # "a_in_b" or "b_in_a"
    rate_factor: float | None = None  # playback rate of B relative to A (e.g., 1.0417 for PAL speedup)

    def to_dict(self) -> dict:
        result = {
            "match_type": self.match_type.value,
            "subtype": self.subtype.value if self.subtype else None,
            "a_duration": self.a_duration,
            "b_duration": self.b_duration,
            "video_segments": [s.to_dict() for s in self.video_segments],
            "audio_segments": [s.to_dict() for s in self.audio_segments],
            "video_available": self.video_available,
            "audio_available": self.audio_available,
        }
        if self.subset_direction:
            result["subset_direction"] = self.subset_direction
        if self.rate_factor is not None:
            result["rate_factor"] = self.rate_factor
        return result
