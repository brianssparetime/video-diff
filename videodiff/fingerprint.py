from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class VideoInfo:
    duration: float
    has_video: bool
    has_audio: bool
    width: int = 0
    height: int = 0


@dataclass
class Fingerprints:
    brightness: np.ndarray   # mean grayscale brightness per sample
    edges: np.ndarray        # mean Sobel gradient magnitude per sample
    contrast: np.ndarray     # grayscale standard deviation per sample
    saturation: np.ndarray   # mean HSV saturation per sample
    hue_sin: np.ndarray      # mean sin(hue) per sample
    hue_cos: np.ndarray      # mean cos(hue) per sample
    spatial_h: np.ndarray    # horizontal gradient count (dHash-like) per sample
    spatial_v: np.ndarray    # vertical gradient count (dHash-like) per sample
    sample_rate: float       # samples per second
    duration: float
    has_video: bool
    has_audio: bool
    audio_chunks: list[list[int]] = field(default_factory=list)


def file_partial_hash(path: str) -> str:
    """Fast content key: file size + hash of first and last 64KB."""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        head = f.read(65536)
        h.update(head)
        if size > 65536:
            f.seek(max(65536, size - 65536))
            h.update(f.read(65536))
    return f"{size:x}_{h.hexdigest()[:32]}"


def probe_video(path: str, ffmpeg_path: str = "ffmpeg") -> VideoInfo:
    ffprobe = ffmpeg_path.replace("ffmpeg", "ffprobe")
    cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    duration = float(info["format"]["duration"])
    has_video = False
    has_audio = False
    width, height = 0, 0
    for s in info.get("streams", []):
        if s["codec_type"] == "video" and not has_video:
            has_video = True
            width = int(s.get("width", 0))
            height = int(s.get("height", 0))
        elif s["codec_type"] == "audio":
            has_audio = True
    return VideoInfo(
        duration=duration, has_video=has_video, has_audio=has_audio,
        width=width, height=height,
    )


_EMPTY_FEATURES = {
    "brightness": np.array([]), "edges": np.array([]),
    "contrast": np.array([]), "saturation": np.array([]),
    "hue_sin": np.array([]), "hue_cos": np.array([]),
    "spatial_h": np.array([]), "spatial_v": np.array([]),
}


def extract_visual_features(
    path: str,
    info: VideoInfo,
    sample_rate_hz: float = 4.0,
    ffmpeg_path: str = "ffmpeg",
    progress_callback=None,
) -> dict[str, np.ndarray]:
    """Extract visual feature signals via ffmpeg pipe.

    Decodes the video sequentially at the target sample rate, avoiding
    the repeated keyframe-seeking overhead of per-frame random access.

    Returns a dict of numpy arrays keyed by channel name.
    progress_callback, if provided, is called with (current_ms, total_ms).
    """
    if info.width <= 0 or info.height <= 0:
        return dict(_EMPTY_FEATURES)

    frame_size = info.width * info.height * 3
    duration_ms = info.duration * 1000

    cmd = [
        ffmpeg_path, "-i", path, "-an",
        "-vf", f"fps={sample_rate_hz}",
        "-pix_fmt", "bgr24", "-f", "rawvideo",
        "-v", "error", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    brightness, edges, contrast, saturation = [], [], [], []
    hue_sin, hue_cos, spatial_h, spatial_v = [], [], [], []
    frame_idx = 0
    last_report_ms = 0.0

    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (info.height, info.width, 3),
            )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            brightness.append(float(np.mean(gray)))
            contrast.append(float(np.std(gray)))

            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges.append(float(np.mean(np.sqrt(sx**2 + sy**2))))

            saturation.append(float(np.mean(hsv[:, :, 1])))
            hue_rad = hsv[:, :, 0].astype(np.float64) * (2.0 * np.pi / 180.0)
            hue_sin.append(float(np.mean(np.sin(hue_rad))))
            hue_cos.append(float(np.mean(np.cos(hue_rad))))

            small_h = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
            spatial_h.append(int(np.sum(small_h[:, 1:] > small_h[:, :-1])))

            small_v = cv2.resize(gray, (8, 9), interpolation=cv2.INTER_AREA)
            spatial_v.append(int(np.sum(small_v[1:, :] > small_v[:-1, :])))

            frame_idx += 1
            current_ms = frame_idx * (1000.0 / sample_rate_hz)

            if progress_callback and current_ms - last_report_ms >= 5000:
                progress_callback(current_ms, duration_ms)
                last_report_ms = current_ms
    finally:
        proc.stdout.close()
        proc.wait()
        proc.stderr.close()

    if progress_callback:
        progress_callback(duration_ms, duration_ms)
    return {
        "brightness": np.array(brightness),
        "edges": np.array(edges),
        "contrast": np.array(contrast),
        "saturation": np.array(saturation),
        "hue_sin": np.array(hue_sin),
        "hue_cos": np.array(hue_cos),
        "spatial_h": np.array(spatial_h, dtype=np.float64),
        "spatial_v": np.array(spatial_v, dtype=np.float64),
    }


def extract_audio_chunks(
    path: str,
    granularity: float,
    duration: float,
    ffmpeg_path: str = "ffmpeg",
    fpcalc_path: str = "fpcalc",
) -> list[list[int]]:
    """Extract audio fingerprint chunks via fpcalc."""
    wav_path = tempfile.mktemp(prefix="videodiff_audio_", suffix=".wav")
    try:
        subprocess.run(
            [ffmpeg_path, "-i", path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "44100", "-ac", "1", "-y", wav_path],
            capture_output=True, check=True,
        )
    except subprocess.CalledProcessError:
        return []

    try:
        result = subprocess.run(
            [fpcalc_path, "-raw", "-json",
             "-length", str(int(min(duration, 7200))), wav_path],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return []

    fp_array = data.get("fingerprint", [])
    fp_duration = data.get("duration", duration)
    if not fp_array:
        return []

    ints_per_second = len(fp_array) / fp_duration if fp_duration > 0 else 2.0
    ints_per_chunk = max(1, int(ints_per_second * granularity))
    num_chunks = int(duration / granularity)

    chunks = []
    for i in range(num_chunks):
        start_idx = int(i * ints_per_chunk)
        end_idx = int((i + 1) * ints_per_chunk)
        chunk = fp_array[start_idx:end_idx]
        chunks.append(chunk if chunk else [])
    return chunks


def fingerprint_video(
    path: str,
    granularity: float = 2.0,
    ffmpeg_path: str = "ffmpeg",
    fpcalc_path: str = "fpcalc",
    progress_callback=None,
) -> Fingerprints:
    info = probe_video(path, ffmpeg_path)

    # Visual features sampled at 4Hz.
    # Higher rate gives finer offset resolution for cross-correlation,
    # reducing drift in long matched segments.
    sample_rate = 4.0

    features = _EMPTY_FEATURES
    if info.has_video:
        features = extract_visual_features(
            path, info, sample_rate, ffmpeg_path, progress_callback,
        )

    audio_chunks = []
    if info.has_audio:
        audio_chunks = extract_audio_chunks(
            path, granularity, info.duration, ffmpeg_path, fpcalc_path,
        )

    return Fingerprints(
        brightness=features["brightness"],
        edges=features["edges"],
        contrast=features["contrast"],
        saturation=features["saturation"],
        hue_sin=features["hue_sin"],
        hue_cos=features["hue_cos"],
        spatial_h=features["spatial_h"],
        spatial_v=features["spatial_v"],
        sample_rate=sample_rate,
        duration=info.duration,
        has_video=info.has_video,
        has_audio=info.has_audio,
        audio_chunks=audio_chunks,
    )
