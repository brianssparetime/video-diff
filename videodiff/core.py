from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np

from .compare import compare
from .fingerprint import Fingerprints, file_partial_hash, fingerprint_video
from .models import ComparisonResult

CACHE_DIR = "/tmp/videodiff_cache"


def check_dependencies(ffmpeg_path: str, fpcalc_path: str) -> None:
    missing = []
    if not shutil.which(ffmpeg_path):
        missing.append(f"ffmpeg not found at '{ffmpeg_path}'")
    ffprobe = ffmpeg_path.replace("ffmpeg", "ffprobe")
    if not shutil.which(ffprobe):
        missing.append(f"ffprobe not found at '{ffprobe}'")
    if not shutil.which(fpcalc_path):
        missing.append(f"fpcalc not found at '{fpcalc_path}'")
    if missing:
        for m in missing:
            print(f"Error: {m}", file=sys.stderr)
        sys.exit(2)


CACHE_VERSION = 4  # bump when fingerprint format changes

def _cache_key(file_hash: str, granularity: float) -> str:
    return f"{file_hash}_g{granularity}_v{CACHE_VERSION}"


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, key + ".npz")


def _save_fingerprint(key: str, fp: Fingerprints) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(key)
    # Store arrays in npz, metadata in a sidecar json
    np.savez_compressed(
        path,
        brightness=fp.brightness,
        edges=fp.edges,
        contrast=fp.contrast,
        saturation=fp.saturation,
        hue_sin=fp.hue_sin,
        hue_cos=fp.hue_cos,
        spatial_h=fp.spatial_h,
        spatial_v=fp.spatial_v,
    )
    meta_path = path.replace(".npz", ".json")
    meta = {
        "sample_rate": fp.sample_rate,
        "duration": fp.duration,
        "has_video": fp.has_video,
        "has_audio": fp.has_audio,
        "audio_chunks": fp.audio_chunks,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def _load_fingerprint(key: str) -> Fingerprints | None:
    path = _cache_path(key)
    meta_path = path.replace(".npz", ".json")
    if not os.path.exists(path) or not os.path.exists(meta_path):
        return None
    try:
        data = np.load(path)
        with open(meta_path) as f:
            meta = json.load(f)
        return Fingerprints(
            brightness=data["brightness"],
            edges=data["edges"],
            contrast=data["contrast"],
            saturation=data["saturation"],
            hue_sin=data["hue_sin"],
            hue_cos=data["hue_cos"],
            spatial_h=data["spatial_h"],
            spatial_v=data["spatial_v"],
            sample_rate=meta["sample_rate"],
            duration=meta["duration"],
            has_video=meta["has_video"],
            has_audio=meta["has_audio"],
            audio_chunks=meta["audio_chunks"],
        )
    except Exception:
        return None


def run_comparison(
    path_a: str,
    path_b: str,
    granularity: float = 2.0,
    ffmpeg_path: str = "ffmpeg",
    fpcalc_path: str = "fpcalc",
    progress_callback=None,
    force: bool = False,
) -> ComparisonResult:
    """Run a full comparison between two video files.

    progress_callback, if provided, is called with (stage, detail) where:
      stage: "hashing", "fingerprint_a", "fingerprint_b", "comparing", "done"
      detail: dict with stage-specific info (e.g., percent complete)

    If force=True, ignore cached fingerprints and re-analyze from scratch.
    """
    check_dependencies(ffmpeg_path, fpcalc_path)

    if progress_callback:
        progress_callback("hashing", {"percent": 0})

    hash_a = file_partial_hash(path_a)
    hash_b = file_partial_hash(path_b)

    if progress_callback:
        progress_callback("hashing", {"percent": 100})

    def make_fp_callback(stage_name):
        def cb(current_ms, total_ms):
            pct = min(100, int(current_ms / total_ms * 100)) if total_ms > 0 else 0
            if progress_callback:
                progress_callback(stage_name, {"percent": pct})
        return cb

    # Fingerprint A (with cache)
    key_a = _cache_key(hash_a, granularity)
    fp_a = None if force else _load_fingerprint(key_a)
    cached_a = fp_a is not None

    if fp_a is None:
        if progress_callback:
            progress_callback("fingerprint_a", {"percent": 0})
        fp_a = fingerprint_video(
            path_a, granularity, ffmpeg_path, fpcalc_path,
            progress_callback=make_fp_callback("fingerprint_a"),
        )
        _save_fingerprint(key_a, fp_a)
    else:
        if progress_callback:
            progress_callback("fingerprint_a", {"percent": 100, "cached": True})

    # Fingerprint B (with cache)
    key_b = _cache_key(hash_b, granularity)
    fp_b = None if force else _load_fingerprint(key_b)
    cached_b = fp_b is not None

    if fp_b is None:
        if progress_callback:
            progress_callback("fingerprint_b", {"percent": 0})
        fp_b = fingerprint_video(
            path_b, granularity, ffmpeg_path, fpcalc_path,
            progress_callback=make_fp_callback("fingerprint_b"),
        )
        _save_fingerprint(key_b, fp_b)
    else:
        if progress_callback:
            progress_callback("fingerprint_b", {"percent": 100, "cached": True})

    if progress_callback:
        progress_callback("comparing", {"percent": 0})

    result = compare(fp_a, fp_b, hash_a, hash_b, granularity=granularity)

    if progress_callback:
        progress_callback("done", {"percent": 100})

    return result
