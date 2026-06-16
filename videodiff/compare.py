from __future__ import annotations

import numpy as np
from scipy.signal import correlate

from .fingerprint import Fingerprints
from .models import (
    ComparisonResult,
    MatchType,
    Segment,
    SegmentStatus,
    Subtype,
)


CHANNEL_WEIGHTS = [2.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]
MAX_SCORE = sum(CHANNEL_WEIGHTS)


def _normalized_derivative(signal: np.ndarray) -> np.ndarray:
    """First derivative, zero-mean unit-variance normalized."""
    d = np.diff(signal.astype(float))
    std = d.std()
    if std < 1e-10:
        return np.zeros_like(d)
    return (d - d.mean()) / std


def _get_derivatives(fp: Fingerprints) -> list[np.ndarray]:
    """Get normalized derivatives of all visual signals.

    Order matches CHANNEL_WEIGHTS: brightness, edges, spatial_h, spatial_v,
    contrast, saturation, hue_sin, hue_cos.
    """
    return [
        _normalized_derivative(fp.brightness),
        _normalized_derivative(fp.edges),
        _normalized_derivative(fp.spatial_h),
        _normalized_derivative(fp.spatial_v),
        _normalized_derivative(fp.contrast),
        _normalized_derivative(fp.saturation),
        _normalized_derivative(fp.hue_sin),
        _normalized_derivative(fp.hue_cos),
    ]


def _combined_score(
    a_derivs: list[np.ndarray],
    b_derivs: list[np.ndarray],
    a_start: int,
    b_start: int,
    length: int,
) -> float:
    """Weighted normalized cross-correlation of all signals over a window."""
    score = 0.0
    for a_d, b_d, w in zip(a_derivs, b_derivs, CHANNEL_WEIGHTS):
        a_chunk = a_d[a_start:a_start + length]
        b_chunk = b_d[b_start:b_start + length]
        if len(a_chunk) < length or len(b_chunk) < length:
            return -999.0
        a_std = a_chunk.std()
        b_std = b_chunk.std()
        if a_std < 1e-10 or b_std < 1e-10:
            continue
        a_n = (a_chunk - a_chunk.mean()) / a_std
        b_n = (b_chunk - b_chunk.mean()) / b_std
        score += w * np.dot(a_n, b_n) / length
    return score


def _find_best_offset_crosscorr(
    a_derivs: list[np.ndarray],
    b_derivs: list[np.ndarray],
    a_start: int,
    b_start: int,
    chunk_len: int,
) -> int:
    """Find the lag that best aligns a_derivs[a_start:] with b_derivs[b_start:]
    using FFT-based cross-correlation over chunk_len samples.

    Returns the lag in samples (a_pos = b_pos + lag).
    """
    combined_corr = None
    for a_d, b_d, w in zip(a_derivs, b_derivs, CHANNEL_WEIGHTS):
        a_chunk = a_d[a_start:a_start + chunk_len].copy()
        b_chunk = b_d[b_start:b_start + chunk_len].copy()
        a_std = a_chunk.std()
        b_std = b_chunk.std()
        if a_std < 1e-10 or b_std < 1e-10:
            continue
        a_chunk = (a_chunk - a_chunk.mean()) / a_std
        b_chunk = (b_chunk - b_chunk.mean()) / b_std
        c = w * correlate(a_chunk, b_chunk, mode="full")
        if combined_corr is None:
            combined_corr = c
        else:
            combined_corr += c

    if combined_corr is None:
        return 0

    lags = np.arange(-(chunk_len - 1), chunk_len)
    return int(lags[np.argmax(combined_corr)])


def _find_segments(
    a_derivs: list[np.ndarray],
    b_derivs: list[np.ndarray],
    sample_rate: float,
    window_seconds: float = 20.0,
    step_seconds: float = 10.0,
    match_threshold: float = 0.35 * MAX_SCORE,
    min_segment_seconds: float = 10.0,
    offset_tolerance_seconds: float = 5.0,
) -> list[dict]:
    """Iteratively find matching segments between A and B.

    Walks through both videos with advancing cursors. At each step,
    cross-correlates remaining content to find the best offset, then
    walks forward collecting matching windows. Segments are assumed
    to be in-order and non-overlapping.
    """
    a_len = len(a_derivs[0])
    b_len = len(b_derivs[0])
    window = int(window_seconds * sample_rate)
    step = int(step_seconds * sample_rate)
    min_segment_samples = int(min_segment_seconds * sample_rate)
    offset_tolerance_samples = int(offset_tolerance_seconds * sample_rate)
    # Cross-correlate using up to 5 minutes of content from each cursor
    max_xcorr_chunk = int(300 * sample_rate)

    segments = []
    a_cursor = 0
    b_cursor = 0

    for _ in range(50):  # safety limit on iterations
        if b_cursor >= b_len - window or a_cursor >= a_len - window:
            break

        chunk_len = min(max_xcorr_chunk, a_len - a_cursor, b_len - b_cursor)
        if chunk_len < window:
            break

        offset = _find_best_offset_crosscorr(
            a_derivs, b_derivs, a_cursor, b_cursor, chunk_len,
        )

        # Walk forward at this offset, collecting matching points
        points = []
        for b_rel in range(0, chunk_len - window, step):
            a_rel = b_rel + offset
            if a_rel < 0 or a_rel + window > chunk_len:
                continue
            sc = _combined_score(
                a_derivs, b_derivs,
                a_cursor + a_rel, b_cursor + b_rel, window,
            )
            if sc >= match_threshold:
                points.append((b_rel, a_rel, sc))

        if not points:
            # No match. Advance B cursor and retry.
            b_cursor += int(30 * sample_rate)
            continue

        # Find longest contiguous run of matching points with consistent offset
        runs = []
        current_run = [points[0]]
        for i in range(1, len(points)):
            prev_b, prev_a = current_run[-1][0], current_run[-1][1]
            curr_b, curr_a = points[i][0], points[i][1]
            offset_drift = abs((curr_a - prev_a) - (curr_b - prev_b))
            if curr_b - prev_b <= step * 2 and offset_drift <= offset_tolerance_samples:
                current_run.append(points[i])
            else:
                runs.append(current_run)
                current_run = [points[i]]
        runs.append(current_run)

        best_run = max(runs, key=len)
        run_b_start = best_run[0][0]
        run_b_end = best_run[-1][0] + window
        run_a_start = best_run[0][1]
        run_a_end = best_run[-1][1] + window
        run_dur_samples = run_b_end - run_b_start

        if run_dur_samples < min_segment_samples:
            b_cursor += int(30 * sample_rate)
            continue

        abs_b_start = (b_cursor + run_b_start) / sample_rate
        abs_a_start = (a_cursor + run_a_start) / sample_rate
        # Force equal A and B durations -- content matches at a constant offset
        run_dur_seconds = run_dur_samples / sample_rate
        abs_b_end = abs_b_start + run_dur_seconds
        abs_a_end = abs_a_start + run_dur_seconds
        med_offset = float(np.median([p[1] - p[0] for p in best_run])) / sample_rate
        avg_score = float(np.mean([p[2] for p in best_run]))

        segments.append({
            "b_start": abs_b_start,
            "b_end": abs_b_end,
            "a_start": abs_a_start,
            "a_end": abs_a_end,
            "offset": med_offset,
            "score": avg_score,
        })

        # Advance cursors past this match
        a_cursor += run_a_end
        b_cursor += run_b_end

    # Merge adjacent segments with similar offsets (artifacts of chunked xcorr)
    return _merge_raw_segments(segments)


def _merge_raw_segments(segments: list[dict], max_gap_s: float = 60.0, offset_tol_s: float = 15.0) -> list[dict]:
    """Merge consecutive raw match segments that have small gaps between them.

    Two adjacent matches are merged if:
    - The gap in both A and B is under max_gap_s
    - The offset difference is under offset_tol_s
    This absorbs boundary artifacts from the chunked cross-correlation.
    """
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        b_gap = seg["b_start"] - prev["b_end"]
        a_gap = seg["a_start"] - prev["a_end"]
        offset_diff = abs(seg["offset"] - prev["offset"])
        if b_gap <= max_gap_s and a_gap <= max_gap_s and offset_diff <= offset_tol_s:
            # Merge, keeping A/B durations balanced
            prev["a_end"] = seg["a_end"]
            total_a_dur = prev["a_end"] - prev["a_start"]
            prev["b_end"] = prev["b_start"] + total_a_dur
            prev["score"] = (prev["score"] + seg["score"]) / 2
        else:
            merged.append(seg.copy())
    return merged


def _detect_rate_factor(
    segments: list[dict],
    min_segments: int = 3,
    min_r_squared: float = 0.9,
    rate_range: tuple[float, float] = (0.93, 1.07),
) -> float | None:
    """Detect a constant playback rate mismatch from fragmented raw segments.

    Fits a linear regression to (b_midpoint, offset) pairs. If the fit is
    clean (R-squared > threshold) and the implied rate factor is within
    the plausible range, returns it. Otherwise returns None.
    """
    if len(segments) < min_segments:
        return None

    b_mids = np.array([(s["b_start"] + s["b_end"]) / 2.0 for s in segments])
    offsets = np.array([s["offset"] for s in segments])

    b_mean = b_mids.mean()
    off_mean = offsets.mean()
    ss_bb = np.sum((b_mids - b_mean) ** 2)
    if ss_bb < 1e-10:
        return None

    slope = np.sum((b_mids - b_mean) * (offsets - off_mean)) / ss_bb

    # R-squared
    predicted = slope * b_mids + (off_mean - slope * b_mean)
    ss_res = np.sum((offsets - predicted) ** 2)
    ss_tot = np.sum((offsets - off_mean) ** 2)
    if ss_tot < 1e-10:
        return None
    r_squared = 1.0 - ss_res / ss_tot

    if r_squared < min_r_squared:
        return None

    # slope = 1/r - 1, so r = 1 / (1 + slope)
    rate_factor = 1.0 / (1.0 + slope)

    if not (rate_range[0] <= rate_factor <= rate_range[1]):
        return None
    if abs(rate_factor - 1.0) < 0.005:
        return None

    return rate_factor


def _resample_derivs(
    derivs: list[np.ndarray],
    rate_factor: float,
) -> list[np.ndarray]:
    """Stretch/compress derivative arrays by a rate factor.

    Used to align B's derivatives with A's timeline when B plays at
    a different rate. Resampling is via linear interpolation.
    """
    resampled = []
    for d in derivs:
        old_len = len(d)
        new_len = int(round(old_len * rate_factor))
        if new_len < 2 or old_len < 2:
            resampled.append(d.copy())
            continue
        old_x = np.arange(old_len)
        new_x = np.linspace(0, old_len - 1, new_len)
        resampled.append(np.interp(new_x, old_x, d))
    return resampled


def _unmap_resampled_segments(
    segments: list[dict],
    rate_factor: float,
) -> list[dict]:
    """Convert B-side timestamps from resampled time back to real time.

    In the resampled domain, B indices correspond to A-rate time.
    Real B time = resampled B time / rate_factor.
    """
    result = []
    for seg in segments:
        real_b_start = seg["b_start"] / rate_factor
        real_b_end = seg["b_end"] / rate_factor
        result.append({
            "a_start": seg["a_start"],
            "a_end": seg["a_end"],
            "b_start": real_b_start,
            "b_end": real_b_end,
            "offset": seg["a_start"] - real_b_start,
            "score": seg["score"],
        })
    return result


def _score_result(segments: list[dict]) -> tuple[float, float, int]:
    """Quality summary of raw segment list for comparison gating."""
    if not segments:
        return (0.0, 0.0, 0)
    total_dur = sum(s["a_end"] - s["a_start"] for s in segments)
    avg_score = sum(s["score"] for s in segments) / len(segments)
    return (avg_score, total_dur, len(segments))


def _flip_segments(segments: list[dict]) -> list[dict]:
    """Flip reverse-pass raw segments back to A/B coordinate space.

    When we run _find_segments(b_derivs, a_derivs), the result uses "A" to
    mean B and "B" to mean A.  This swaps them back.
    """
    return [
        {
            "a_start": seg["b_start"],
            "a_end": seg["b_end"],
            "b_start": seg["a_start"],
            "b_end": seg["a_end"],
            "offset": -seg["offset"],
            "score": seg["score"],
        }
        for seg in segments
    ]


def _merge_bidirectional_segments(
    forward: list[dict],
    reverse: list[dict],
    overlap_threshold: float = 0.5,
    offset_tolerance_s: float = 15.0,
) -> list[dict]:
    """Merge forward-pass and reverse-pass raw segments.

    Segments found by both passes (overlapping in A-time with compatible
    offsets) are kept once, preferring the higher-scoring version.
    Segments found only by the reverse pass are inserted -- these are
    the matches the forward pass missed due to cursor-advancement bias.
    """
    if not reverse:
        return forward
    if not forward:
        return reverse

    reverse = sorted(reverse, key=lambda s: s["a_start"])

    claimed = [False] * len(reverse)
    result = []

    for f_seg in forward:
        best_idx = -1
        best_overlap = 0.0

        for ri, r_seg in enumerate(reverse):
            if claimed[ri]:
                continue
            if r_seg["a_end"] <= f_seg["a_start"]:
                continue
            if r_seg["a_start"] >= f_seg["a_end"]:
                break

            a_overlap = min(f_seg["a_end"], r_seg["a_end"]) - max(f_seg["a_start"], r_seg["a_start"])
            b_overlap = min(f_seg["b_end"], r_seg["b_end"]) - max(f_seg["b_start"], r_seg["b_start"])
            offset_diff = abs(f_seg["offset"] - r_seg["offset"])

            if a_overlap <= 0 or b_overlap <= 0:
                continue
            if offset_diff > offset_tolerance_s:
                continue

            shorter_dur = min(
                f_seg["a_end"] - f_seg["a_start"],
                r_seg["a_end"] - r_seg["a_start"],
            )
            if shorter_dur > 0 and a_overlap / shorter_dur >= overlap_threshold:
                if a_overlap > best_overlap:
                    best_overlap = a_overlap
                    best_idx = ri

        # Keep forward segment by default
        chosen = f_seg.copy()

        if best_idx >= 0:
            r_seg = reverse[best_idx]
            # Prefer reverse if it scored meaningfully higher
            if r_seg["score"] > f_seg["score"] + 0.1:
                chosen = r_seg.copy()
            claimed[best_idx] = True

        result.append(chosen)

    # Insert unclaimed reverse segments (the wins from the reverse pass)
    for ri, r_seg in enumerate(reverse):
        if not claimed[ri]:
            result.append(r_seg.copy())

    # Sort by a_start and enforce b_start monotonicity.
    # The downstream pipeline assumes both timelines advance together.
    result.sort(key=lambda s: s["a_start"])
    monotonic = []
    prev_b_end = -1.0
    for seg in result:
        if seg["b_start"] >= prev_b_end - 0.5:
            monotonic.append(seg)
            prev_b_end = seg["b_end"]
        # else: drop -- this reverse-inserted segment would break ordering

    return _merge_raw_segments(monotonic)


def _find_segments_short(
    a_derivs: list[np.ndarray],
    b_derivs: list[np.ndarray],
    sample_rate: float,
    a_duration: float,
    b_duration: float,
) -> list[dict]:
    """Simple full cross-correlation for short videos."""
    a_len = len(a_derivs[0])
    b_len = len(b_derivs[0])
    if a_len < 2 or b_len < 2:
        return []

    offset = _find_best_offset_crosscorr(a_derivs, b_derivs, 0, 0, min(a_len, b_len))

    # Check if the full content matches at this offset
    match_len = min(a_len, b_len) - abs(offset)
    if match_len < 2:
        return []

    a_start = max(0, offset)
    b_start = max(0, -offset)
    score = _combined_score(a_derivs, b_derivs, a_start, b_start, match_len)

    if score < 0.20 * MAX_SCORE:
        return []

    return [{
        "b_start": b_start / sample_rate,
        "b_end": (b_start + match_len) / sample_rate,
        "a_start": a_start / sample_rate,
        "a_end": (a_start + match_len) / sample_rate,
        "offset": offset / sample_rate,
        "score": score,
    }]


def _split_on_offset_changes(
    seg: Segment,
    a_derivs: list[np.ndarray],
    b_derivs: list[np.ndarray],
    sample_rate: float,
    micro_threshold: float = 15.0,
    window_seconds: float = 30.0,
    step_seconds: float = 10.0,
    rate_factor: float | None = None,
) -> list[Segment]:
    """Split a match segment where the internal offset changes.

    Slides a cross-correlation window along the matched content. When the
    best local offset shifts (indicating a small insertion/deletion), splits
    the segment there, inserting a micro A_ONLY or B_ONLY gap.

    When rate_factor is set, expected linear drift from the rate mismatch
    is subtracted before checking for shifts, so only genuine content
    changes trigger splits.

    Returns a list of segments replacing the original.
    """
    if seg.status != SegmentStatus.MATCH:
        return [seg]

    a_dur = seg.a_end - seg.a_start
    if a_dur < window_seconds * 2:
        return [seg]

    window = int(window_seconds * sample_rate)
    step = int(step_seconds * sample_rate)
    search_range = int(10 * sample_rate)

    current_offset = seg.b_start - seg.a_start

    # For rate-aware drift subtraction, track the "base offset" -- the
    # offset with linear rate drift removed.  At each measurement point
    # we compute what the offset should be given the rate factor and this
    # base, and only flag deviations from that expectation.
    base_offset = current_offset

    # Collect (a_time, detected_offset) pairs
    offset_changes = []

    a_pos = seg.a_start + step_seconds
    while a_pos + window_seconds < seg.a_end:
        a_idx = int(a_pos * sample_rate)
        expected_b_idx = int((a_pos + current_offset) * sample_rate)

        best_score = -999.0
        best_b_idx = expected_b_idx

        for delta in range(-search_range, search_range + 1):
            trial_b = expected_b_idx + delta
            if trial_b < 0 or trial_b + window > len(b_derivs[0]):
                continue
            if a_idx + window > len(a_derivs[0]):
                continue
            sc = _combined_score(a_derivs, b_derivs, a_idx, trial_b, window)
            if sc > best_score:
                best_score = sc
                best_b_idx = trial_b

        new_offset = best_b_idx / sample_rate - a_pos

        # Compute shift, subtracting expected rate drift if applicable
        if rate_factor is not None and rate_factor != 1.0:
            drift_since_start = (a_pos - seg.a_start) * (1.0 / rate_factor - 1.0)
            expected_offset = base_offset + drift_since_start
            shift = new_offset - expected_offset
        else:
            shift = new_offset - current_offset

        if abs(shift) > 0.5:
            offset_changes.append((a_pos, current_offset, new_offset, shift))
            current_offset = new_offset
            # Reset base after a genuine shift
            if rate_factor is not None and rate_factor != 1.0:
                base_offset = new_offset - (a_pos - seg.a_start) * (1.0 / rate_factor - 1.0)
        else:
            current_offset = new_offset

        a_pos += step_seconds

    if not offset_changes:
        return [seg]

    # Build split segments
    result = []
    a_cursor = seg.a_start
    b_cursor = seg.b_start

    for split_a, old_offset, new_offset, shift in offset_changes:
        # Match up to the split point
        match_dur = split_a - a_cursor
        if match_dur > 1.0:
            if rate_factor is not None and rate_factor != 1.0:
                b_match_dur = match_dur / rate_factor
            else:
                b_match_dur = match_dur
            result.append(Segment(
                a_start=a_cursor,
                a_end=split_a,
                b_start=b_cursor,
                b_end=b_cursor + b_match_dur,
                status=SegmentStatus.MATCH,
            ))

        # Insert micro gap for the offset shift
        a_after_split = split_a
        if rate_factor is not None and rate_factor != 1.0:
            b_after_split = b_cursor + match_dur / rate_factor
        else:
            b_after_split = b_cursor + match_dur

        if shift > 0:
            # B has extra content (B is ahead -> B_ONLY gap)
            gap_dur = abs(shift)
            micro = gap_dur < micro_threshold
            result.append(Segment(
                a_start=None, a_end=None,
                b_start=b_after_split,
                b_end=b_after_split + gap_dur,
                status=SegmentStatus.B_ONLY,
                micro=micro,
            ))
            b_cursor = b_after_split + gap_dur
        else:
            # A has extra content (A is ahead -> A_ONLY gap)
            gap_dur = abs(shift)
            micro = gap_dur < micro_threshold
            result.append(Segment(
                a_start=a_after_split,
                a_end=a_after_split + gap_dur,
                b_start=None, b_end=None,
                status=SegmentStatus.A_ONLY,
                micro=micro,
            ))
            b_cursor = b_after_split
            a_after_split += gap_dur

        a_cursor = a_after_split
        b_cursor = max(b_cursor, a_cursor + new_offset)

    # Final match from last split to segment end
    remaining_a = seg.a_end - a_cursor
    if remaining_a > 1.0:
        if rate_factor is not None and rate_factor != 1.0:
            b_remaining = remaining_a / rate_factor
        else:
            b_remaining = remaining_a
        result.append(Segment(
            a_start=a_cursor,
            a_end=seg.a_end,
            b_start=b_cursor,
            b_end=b_cursor + b_remaining,
            status=SegmentStatus.MATCH,
        ))

    return result


def _refine_boundaries(
    segments: list[Segment],
    a_derivs: list[np.ndarray],
    b_derivs: list[np.ndarray],
    sample_rate: float,
    granularity: float,
) -> list[Segment]:
    """Refine the boundary between each adjacent MATCH and gap segment.

    For each transition (MATCH->gap or gap->MATCH), uses a small
    cross-correlation window to binary-search for the exact point where
    the match quality drops, down to granularity precision.
    """
    if len(segments) < 2:
        return segments

    window = int(max(granularity * 2, 6.0) * sample_rate)
    min_step = max(1, int(granularity * sample_rate / 2))

    result = list(segments)  # work on a copy

    for idx in range(len(result) - 1):
        seg = result[idx]
        nxt = result[idx + 1]

        # Refine the end of a MATCH that borders a gap.
        # Only refine if the current boundary region scores poorly,
        # indicating the coarse pass overshot.
        if seg.status == SegmentStatus.MATCH and nxt.status != SegmentStatus.MATCH:
            offset = seg.b_start - seg.a_start
            # Check if the current boundary area actually needs refinement
            boundary_ok = _boundary_scores_well(
                a_derivs, b_derivs, sample_rate,
                seg.a_end - 10, offset, window,
            )
            if boundary_ok:
                continue
            refined_a = _find_match_boundary(
                a_derivs, b_derivs, sample_rate,
                a_start=seg.a_start, a_end=seg.a_end,
                offset=offset, window=window, min_step=min_step,
                search_end=True,
            )
            if refined_a is not None and abs(refined_a - seg.a_end) > 0.5:
                delta = refined_a - seg.a_end
                result[idx] = Segment(
                    a_start=seg.a_start, a_end=refined_a,
                    b_start=seg.b_start, b_end=seg.b_start + (refined_a - seg.a_start),
                    status=SegmentStatus.MATCH, micro=seg.micro,
                )
                # Adjust the gap start
                if nxt.status == SegmentStatus.A_ONLY:
                    result[idx + 1] = Segment(
                        a_start=refined_a, a_end=nxt.a_end,
                        b_start=None, b_end=None,
                        status=SegmentStatus.A_ONLY, micro=nxt.micro,
                    )
                elif nxt.status == SegmentStatus.B_ONLY:
                    new_b_start = seg.b_start + (refined_a - seg.a_start)
                    result[idx + 1] = Segment(
                        a_start=None, a_end=None,
                        b_start=new_b_start, b_end=nxt.b_end,
                        status=SegmentStatus.B_ONLY, micro=nxt.micro,
                    )

        # Refine the start of a MATCH that follows a gap
        if seg.status != SegmentStatus.MATCH and nxt.status == SegmentStatus.MATCH:
            offset = nxt.b_start - nxt.a_start
            boundary_ok = _boundary_scores_well(
                a_derivs, b_derivs, sample_rate,
                nxt.a_start, offset, window,
            )
            if boundary_ok:
                continue
            refined_a = _find_match_boundary(
                a_derivs, b_derivs, sample_rate,
                a_start=nxt.a_start, a_end=nxt.a_end,
                offset=offset, window=window, min_step=min_step,
                search_end=False,
            )
            if refined_a is not None and abs(refined_a - nxt.a_start) > 0.5:
                result[idx + 1] = Segment(
                    a_start=refined_a, a_end=nxt.a_end,
                    b_start=nxt.b_start + (refined_a - nxt.a_start),
                    b_end=nxt.b_end,
                    status=SegmentStatus.MATCH, micro=nxt.micro,
                )
                # Adjust the gap end
                if seg.status == SegmentStatus.A_ONLY:
                    result[idx] = Segment(
                        a_start=seg.a_start, a_end=refined_a,
                        b_start=None, b_end=None,
                        status=SegmentStatus.A_ONLY, micro=seg.micro,
                    )
                elif seg.status == SegmentStatus.B_ONLY:
                    new_b_end = nxt.b_start + (refined_a - nxt.a_start)
                    result[idx] = Segment(
                        a_start=None, a_end=None,
                        b_start=seg.b_start, b_end=new_b_end,
                        status=SegmentStatus.B_ONLY, micro=seg.micro,
                    )

    # Filter out any segments that collapsed to zero or negative duration
    filtered = []
    for seg in result:
        if seg.status == SegmentStatus.MATCH:
            if seg.a_end - seg.a_start > 0.5:
                filtered.append(seg)
        elif seg.status == SegmentStatus.A_ONLY:
            if seg.a_end - seg.a_start > 0.5:
                filtered.append(seg)
        elif seg.status == SegmentStatus.B_ONLY:
            if seg.b_end - seg.b_start > 0.5:
                filtered.append(seg)

    return filtered


def _boundary_scores_well(
    a_derivs: list[np.ndarray],
    b_derivs: list[np.ndarray],
    sample_rate: float,
    a_time: float,
    offset: float,
    window: int,
    threshold: float = 0.20 * MAX_SCORE,
) -> bool:
    """Check if the match quality near a boundary is already reasonable."""
    a_idx = int(a_time * sample_rate)
    b_idx = int((a_time + offset) * sample_rate)
    if a_idx < 0 or b_idx < 0:
        return False
    if a_idx + window > len(a_derivs[0]) or b_idx + window > len(b_derivs[0]):
        return True  # can't check, assume it's fine
    sc = _combined_score(a_derivs, b_derivs, a_idx, b_idx, window)
    return sc >= threshold


def _find_match_boundary(
    a_derivs: list[np.ndarray],
    b_derivs: list[np.ndarray],
    sample_rate: float,
    a_start: float,
    a_end: float,
    offset: float,
    window: int,
    min_step: int,
    search_end: bool,
    match_threshold: float = 0.27 * MAX_SCORE,
) -> float | None:
    """Binary search for where match quality transitions.

    If search_end=True, finds the latest point in [a_start, a_end] where
    the match is still good (refining the end of a match).
    If search_end=False, finds the earliest point where match becomes good
    (refining the start of a match).
    """
    lo = int(a_start * sample_rate)
    hi = int(a_end * sample_rate)

    if hi - lo < window * 2:
        return None

    def is_good(a_idx: int) -> bool:
        b_idx = int((a_idx / sample_rate + offset) * sample_rate)
        if a_idx < 0 or b_idx < 0:
            return False
        if a_idx + window > len(a_derivs[0]) or b_idx + window > len(b_derivs[0]):
            return False
        sc = _combined_score(a_derivs, b_derivs, a_idx, b_idx, window)
        return sc >= match_threshold

    if search_end:
        # Find rightmost good position
        best = lo
        while hi - lo > min_step:
            mid = (lo + hi) // 2
            if is_good(mid):
                best = mid
                lo = mid + min_step
            else:
                hi = mid
        return (best + window) / sample_rate
    else:
        # Find leftmost good position
        best = hi
        while hi - lo > min_step:
            mid = (lo + hi) // 2
            if is_good(mid):
                best = mid
                hi = mid
            else:
                lo = mid + min_step
        return best / sample_rate


def _rebuild_gaps(
    segments: list[Segment],
    a_duration: float,
    b_duration: float,
) -> list[Segment]:
    """Rebuild gap segments from match segments.

    Discards existing gap segments and re-derives A_ONLY/B_ONLY from the
    spaces between matches. Preserves the micro flag from any gap that
    previously existed at approximately the same location.
    """
    # Collect micro gap locations for re-tagging
    micro_regions = []
    for seg in segments:
        if seg.micro:
            if seg.status == SegmentStatus.A_ONLY:
                micro_regions.append(("a", seg.a_start, seg.a_end))
            elif seg.status == SegmentStatus.B_ONLY:
                micro_regions.append(("b", seg.b_start, seg.b_end))

    matches = [s for s in segments if s.status == SegmentStatus.MATCH]

    def is_micro(side: str, start: float, end: float) -> bool:
        dur = end - start
        if dur >= 15.0:
            return False
        for m_side, m_start, m_end in micro_regions:
            if m_side == side and abs(m_start - start) < 30 and abs(m_end - end) < 30:
                return True
        return dur < 15.0  # new gaps under 15s also count as micro

    result = []
    prev_a = 0.0
    prev_b = 0.0

    for seg in matches:
        # A gap before this match
        if seg.a_start - prev_a > 0.5:
            micro = is_micro("a", prev_a, seg.a_start)
            result.append(Segment(
                a_start=prev_a, a_end=seg.a_start,
                b_start=None, b_end=None,
                status=SegmentStatus.A_ONLY, micro=micro,
            ))
        # B gap before this match
        if seg.b_start - prev_b > 0.5:
            micro = is_micro("b", prev_b, seg.b_start)
            result.append(Segment(
                a_start=None, a_end=None,
                b_start=prev_b, b_end=seg.b_start,
                status=SegmentStatus.B_ONLY, micro=micro,
            ))

        result.append(seg)
        prev_a = seg.a_end
        prev_b = seg.b_end

    # Trailing gaps
    if a_duration - prev_a > 0.5:
        result.append(Segment(
            a_start=prev_a, a_end=a_duration,
            b_start=None, b_end=None,
            status=SegmentStatus.A_ONLY,
        ))
    if b_duration - prev_b > 0.5:
        result.append(Segment(
            a_start=None, a_end=None,
            b_start=prev_b, b_end=b_duration,
            status=SegmentStatus.B_ONLY,
        ))

    return result


def _segments_to_model(
    segments: list[dict],
    a_duration: float,
    b_duration: float,
    granularity: float = 2.0,
    rate_factor: float | None = None,
) -> list[Segment]:
    """Convert raw segment dicts to Segment model objects, filling in gaps."""
    result = []
    prev_a_end = 0.0
    prev_b_end = 0.0

    for seg in segments:
        bs, be = seg["b_start"], seg["b_end"]
        a_s, ae = seg["a_start"], seg["a_end"]

        # Gap in A before this match
        if a_s - prev_a_end > 1.0:
            result.append(Segment(
                a_start=prev_a_end, a_end=a_s,
                b_start=None, b_end=None,
                status=SegmentStatus.A_ONLY,
            ))
        # Gap in B before this match
        if bs - prev_b_end > 1.0:
            result.append(Segment(
                a_start=None, a_end=None,
                b_start=prev_b_end, b_end=bs,
                status=SegmentStatus.B_ONLY,
            ))

        result.append(Segment(
            a_start=a_s, a_end=ae,
            b_start=bs, b_end=be,
            status=SegmentStatus.MATCH,
        ))
        prev_a_end = ae
        prev_b_end = be

    # Trailing content
    if a_duration - prev_a_end > 1.0:
        result.append(Segment(
            a_start=prev_a_end, a_end=a_duration,
            b_start=None, b_end=None,
            status=SegmentStatus.A_ONLY,
        ))
    if b_duration - prev_b_end > 1.0:
        result.append(Segment(
            a_start=None, a_end=None,
            b_start=prev_b_end, b_end=b_duration,
            status=SegmentStatus.B_ONLY,
        ))

    result = _absorb_small_gaps(result, granularity, rate_factor=rate_factor)
    return _drop_tiny_gaps(result, granularity)


def _drop_tiny_gaps(segments: list[Segment], granularity: float) -> list[Segment]:
    """Drop gap segments shorter than 2x granularity.

    Unlike the old approach, this does NOT extend adjacent matches -- that
    would create duration mismatches. It simply removes the tiny gap.
    """
    min_dur = granularity
    return [
        seg for seg in segments
        if seg.status == SegmentStatus.MATCH
        or (seg.status == SegmentStatus.A_ONLY and seg.a_end - seg.a_start >= min_dur)
        or (seg.status == SegmentStatus.B_ONLY and seg.b_end - seg.b_start >= min_dur)
    ]


def _absorb_small_gaps(
    segments: list[Segment],
    granularity: float,
    rate_factor: float | None = None,
) -> list[Segment]:
    """Merge adjacent MATCH segments that have only small gaps between them.

    When rate_factor is None, A and B durations are forced equal.
    When rate_factor is set, A and B durations are allowed to differ
    by the expected rate ratio.
    """
    min_gap = max(granularity * 6, 10.0)

    if len(segments) < 3:
        return segments

    result = []
    i = 0
    while i < len(segments):
        seg = segments[i]

        if seg.status != SegmentStatus.MATCH:
            result.append(seg)
            i += 1
            continue

        # We have a MATCH. Look ahead for gaps followed by another MATCH.
        merged = seg
        j = i + 1
        while j < len(segments):
            # Peek at consecutive gap segments until next MATCH
            k = j
            a_gap_total = 0.0
            b_gap_total = 0.0
            while k < len(segments) and segments[k].status != SegmentStatus.MATCH:
                gap_seg = segments[k]
                if gap_seg.status == SegmentStatus.A_ONLY:
                    a_gap_total += gap_seg.a_end - gap_seg.a_start
                else:
                    b_gap_total += gap_seg.b_end - gap_seg.b_start
                k += 1

            if k >= len(segments) or segments[k].status != SegmentStatus.MATCH:
                break

            # Absorb if:
            # 1. All gaps are individually small, OR
            # 2. Gaps are "paired" (both A and B have gaps of similar size),
            #    indicating matching content the xcorr briefly lost
            all_small = a_gap_total < min_gap and b_gap_total < min_gap

            paired = False
            if a_gap_total > 0 and b_gap_total > 0:
                gap_ratio = min(a_gap_total, b_gap_total) / max(a_gap_total, b_gap_total)
                paired = gap_ratio > 0.5

            if not all_small and not paired:
                break

            # Verify merging maintains duration balance
            next_match = segments[k]
            merged_a_dur = next_match.a_end - merged.a_start
            merged_b_dur = next_match.b_end - merged.b_start
            if merged_a_dur > 0 and merged_b_dur > 0:
                dur_ratio = min(merged_a_dur, merged_b_dur) / max(merged_a_dur, merged_b_dur)
                if rate_factor is not None:
                    expected_ratio = min(1.0, 1.0 / rate_factor) / max(1.0, 1.0 / rate_factor)
                    if abs(dur_ratio - expected_ratio) > 0.05:
                        break
                else:
                    if dur_ratio < 0.95:
                        break

            if rate_factor is not None:
                # Keep actual B endpoints -- durations differ by the rate factor
                new_b_end = next_match.b_end
            else:
                # Force balanced durations using A as authoritative
                new_b_end = merged.b_start + (next_match.a_end - merged.a_start)

            merged = Segment(
                a_start=merged.a_start,
                a_end=next_match.a_end,
                b_start=merged.b_start,
                b_end=new_b_end,
                status=SegmentStatus.MATCH,
            )
            j = k + 1

        result.append(merged)
        i = j

    return result


def _chunk_hamming_distance(a_ints: list[int], b_ints: list[int]) -> float:
    """Hamming distance between two Chromaprint integer lists, normalized to [0, 1].

    Returns 0.0 for identical audio, 1.0 for completely different.
    Compares the shorter list pairwise; if one list is empty, returns 1.0.
    """
    if not a_ints or not b_ints:
        return 1.0
    total_bits = 0
    differing_bits = 0
    for a_val, b_val in zip(a_ints, b_ints):
        total_bits += 32
        differing_bits += bin(a_val ^ b_val).count("1")
    if total_bits == 0:
        return 1.0
    return differing_bits / total_bits


def _compare_audio(
    fp_a: Fingerprints,
    fp_b: Fingerprints,
    video_segments: list[Segment],
    granularity: float,
    match_threshold: float = 0.25,
    rate_factor: float | None = None,
) -> list[Segment]:
    """Compare audio within video-matched regions.

    For each video MATCH segment, compares the corresponding audio
    fingerprint chunks between A and B.  Produces audio segments
    showing where audio agrees or diverges within the matched regions.

    When rate_factor is set, B spans a different number of chunks than A
    within a match segment, so chunk indices are mapped proportionally.

    match_threshold: maximum normalized Hamming distance to consider
    audio as matching (0.25 = up to 25% of bits differ).
    """
    if not fp_a.audio_chunks or not fp_b.audio_chunks:
        return []

    a_chunks = fp_a.audio_chunks
    b_chunks = fp_b.audio_chunks

    audio_segments = []

    for seg in video_segments:
        if seg.status != SegmentStatus.MATCH:
            continue

        # Map segment time ranges to chunk indices
        a_chunk_start = int(seg.a_start / granularity)
        a_chunk_end = int(seg.a_end / granularity)
        b_chunk_start = int(seg.b_start / granularity)
        b_chunk_end = int(seg.b_end / granularity)

        a_span = a_chunk_end - a_chunk_start
        b_span = b_chunk_end - b_chunk_start
        b_total_dur = seg.b_end - seg.b_start

        # Walk through corresponding chunks and group into runs
        # of matching or non-matching audio
        run_status = None  # True = match, False = mismatch
        run_start_a = seg.a_start
        run_start_b = seg.b_start

        for i in range(a_span):
            a_idx = a_chunk_start + i

            # Proportional B chunk mapping for rate-shifted content
            if rate_factor is not None and rate_factor != 1.0 and a_span > 0:
                b_idx = b_chunk_start + int(i * b_span / a_span)
            else:
                b_idx = b_chunk_start + i

            if a_idx >= len(a_chunks) or b_idx >= len(b_chunks):
                break

            dist = _chunk_hamming_distance(a_chunks[a_idx], b_chunks[b_idx])
            is_match = dist <= match_threshold

            if run_status is None:
                run_status = is_match
                continue

            if is_match != run_status:
                # Emit the completed run with proportional B timestamps
                frac = i / a_span if a_span > 0 else 0
                run_end_a = seg.a_start + i * granularity
                run_end_b = seg.b_start + frac * b_total_dur
                dur = run_end_a - run_start_a

                if dur > 0.5:
                    if run_status:
                        audio_segments.append(Segment(
                            a_start=run_start_a, a_end=run_end_a,
                            b_start=run_start_b, b_end=run_end_b,
                            status=SegmentStatus.MATCH,
                        ))
                    else:
                        # Audio diverges -- emit both A_ONLY and B_ONLY
                        audio_segments.append(Segment(
                            a_start=run_start_a, a_end=run_end_a,
                            b_start=None, b_end=None,
                            status=SegmentStatus.A_ONLY,
                        ))
                        audio_segments.append(Segment(
                            a_start=None, a_end=None,
                            b_start=run_start_b, b_end=run_end_b,
                            status=SegmentStatus.B_ONLY,
                        ))

                run_start_a = run_end_a
                run_start_b = run_end_b
                run_status = is_match

        # Emit final run
        if run_status is not None:
            dur = seg.a_end - run_start_a
            if dur > 0.5:
                if run_status:
                    audio_segments.append(Segment(
                        a_start=run_start_a, a_end=seg.a_end,
                        b_start=run_start_b, b_end=seg.b_end,
                        status=SegmentStatus.MATCH,
                    ))
                else:
                    audio_segments.append(Segment(
                        a_start=run_start_a, a_end=seg.a_end,
                        b_start=None, b_end=None,
                        status=SegmentStatus.A_ONLY,
                    ))
                    audio_segments.append(Segment(
                        a_start=None, a_end=None,
                        b_start=run_start_b, b_end=seg.b_end,
                        status=SegmentStatus.B_ONLY,
                    ))

    return audio_segments


def classify_segments(segments: list[Segment]) -> tuple[Subtype | None, str | None]:
    match_segs = [s for s in segments if s.status == SegmentStatus.MATCH]
    a_only = [s for s in segments if s.status == SegmentStatus.A_ONLY]
    b_only = [s for s in segments if s.status == SegmentStatus.B_ONLY]

    if not match_segs:
        return None, None
    if not a_only and not b_only:
        return Subtype.STRAIGHT, None
    if not a_only and b_only:
        return Subtype.SUBSET, "a_in_b"
    if a_only and not b_only:
        return Subtype.SUBSET, "b_in_a"
    if len(match_segs) >= 2:
        return Subtype.INTERLEAVED, None
    if len(match_segs) == 1:
        if not a_only:
            return Subtype.SUBSET, "a_in_b"
        if not b_only:
            return Subtype.SUBSET, "b_in_a"
    return Subtype.INTERLEAVED, None


def compare(
    fp_a: Fingerprints,
    fp_b: Fingerprints,
    file_hash_a: str,
    file_hash_b: str,
    granularity: float = 2.0,
) -> ComparisonResult:
    video_available = fp_a.has_video and fp_b.has_video
    audio_available = bool(fp_a.audio_chunks) and bool(fp_b.audio_chunks)

    # Byte-identical check
    if file_hash_a == file_hash_b:
        return ComparisonResult(
            match_type=MatchType.BYTE,
            subtype=Subtype.STRAIGHT,
            a_duration=fp_a.duration,
            b_duration=fp_b.duration,
            video_available=video_available,
            audio_available=audio_available,
        )

    # Primary: multi-signal visual comparison
    video_segments = []
    rate_factor = None
    is_short = True
    if video_available and len(fp_a.brightness) > 0 and len(fp_b.brightness) > 0:
        a_derivs = _get_derivatives(fp_a)
        b_derivs = _get_derivatives(fp_b)

        # For short videos (< 60s), use a single full cross-correlation
        min_samples = min(len(a_derivs[0]), len(b_derivs[0]))
        is_short = min_samples < int(60 * fp_a.sample_rate)

        if is_short:
            raw_segments = _find_segments_short(
                a_derivs, b_derivs,
                sample_rate=fp_a.sample_rate,
                a_duration=fp_a.duration,
                b_duration=fp_b.duration,
            )
        else:
            forward_segments = _find_segments(
                a_derivs, b_derivs,
                sample_rate=fp_a.sample_rate,
                min_segment_seconds=max(granularity, 10.0),
            )
            reverse_raw = _find_segments(
                b_derivs, a_derivs,
                sample_rate=fp_a.sample_rate,
                min_segment_seconds=max(granularity, 10.0),
            )
            raw_segments = _merge_bidirectional_segments(
                forward_segments, _flip_segments(reverse_raw),
            )

            # Detect constant playback rate mismatch (e.g., PAL speedup)
            rate_factor = _detect_rate_factor(raw_segments)

            if rate_factor is not None:
                # Re-run with resampled B derivatives
                resampled_b = _resample_derivs(b_derivs, rate_factor)

                rs_forward = _find_segments(
                    a_derivs, resampled_b,
                    sample_rate=fp_a.sample_rate,
                    min_segment_seconds=max(granularity, 10.0),
                )
                rs_reverse_raw = _find_segments(
                    resampled_b, a_derivs,
                    sample_rate=fp_a.sample_rate,
                    min_segment_seconds=max(granularity, 10.0),
                )
                rs_raw = _merge_bidirectional_segments(
                    rs_forward, _flip_segments(rs_reverse_raw),
                )
                rs_raw = _unmap_resampled_segments(rs_raw, rate_factor)

                # Quality gate: keep resampled result only if it improves
                orig_score, orig_dur, orig_n = _score_result(raw_segments)
                rs_score, rs_dur, rs_n = _score_result(rs_raw)

                use_resampled = False
                if rs_dur > orig_dur * 1.1:
                    use_resampled = True
                elif rs_dur >= orig_dur * 0.95 and rs_n < orig_n:
                    use_resampled = True
                elif rs_dur >= orig_dur * 0.95 and rs_score > orig_score + 0.1:
                    use_resampled = True

                if use_resampled:
                    raw_segments = rs_raw
                else:
                    rate_factor = None

        video_segments = _segments_to_model(
            raw_segments, fp_a.duration, fp_b.duration, granularity,
            rate_factor=rate_factor,
        )

        # Passes 2 and 3 only for longer videos where there's enough signal
        if not is_short:
            # Pass 2: Split match segments where internal offset changes
            refined = []
            for seg in video_segments:
                if seg.status == SegmentStatus.MATCH:
                    refined.extend(_split_on_offset_changes(
                        seg, a_derivs, b_derivs, fp_a.sample_rate,
                        rate_factor=rate_factor,
                    ))
                else:
                    refined.append(seg)
            video_segments = refined

            # Pass 3: Refine segment boundaries to granularity precision
            video_segments = _refine_boundaries(
                video_segments, a_derivs, b_derivs,
                fp_a.sample_rate, granularity,
            )

            # Rebuild gaps from matches to ensure consistency after refinement
            video_segments = _rebuild_gaps(
                video_segments, fp_a.duration, fp_b.duration,
            )

    # Use video as primary (it now works across resolutions)
    primary_segments = video_segments

    # Audio comparison within video-matched regions
    audio_segments = []
    if audio_available and primary_segments:
        audio_segments = _compare_audio(
            fp_a, fp_b, primary_segments, granularity,
            rate_factor=rate_factor,
        )

    if not primary_segments:
        return ComparisonResult(
            match_type=MatchType.NO_MATCH,
            subtype=None,
            video_segments=video_segments,
            audio_segments=audio_segments,
            a_duration=fp_a.duration,
            b_duration=fp_b.duration,
            video_available=video_available,
            audio_available=audio_available,
        )

    subtype, direction = classify_segments(primary_segments)
    match_type = MatchType.TRANSCODE if subtype else MatchType.NO_MATCH

    return ComparisonResult(
        match_type=match_type,
        subtype=subtype,
        video_segments=video_segments,
        audio_segments=audio_segments,
        a_duration=fp_a.duration,
        b_duration=fp_b.duration,
        video_available=video_available,
        audio_available=audio_available,
        subset_direction=direction,
        rate_factor=rate_factor,
    )
