# videodiff

videodiff compares two video files and reports where their content matches and where it differs. It answers the questions you cannot resolve by checking file size or a checksum: are these the same program, is one a trimmed or extended version of the other, and exactly which spans of footage are shared.

Comparison works even when the two files share no bytes. Transcoding, re-encoding, a resolution change, a different container, an added or removed intro, commercial breaks cut out, or a constant playback-rate shift all leave the underlying footage recognizable. videodiff aligns the two videos by what they show and sound like, not by how they were encoded.

The result is a timeline of segments. Each segment is labeled as shared between both files, present only in A, or present only in B. Video and audio are reported separately, so a file with identical pictures but a replaced soundtrack reads as a video match with an audio divergence.

## What it tells you

- **Match type**: byte-identical, a transcode of the same content, or no match.
- **Relationship**: the two files are the same throughout (straight), one is wholly contained in the other (subset), or shared and unique spans alternate (interleaved).
- **Segments**: the precise time ranges, in each file, where content is shared or unique.
- **Rate shift**: a constant speed difference between the two, such as the 4 percent PAL speedup, reported as a single rate factor.

## Requirements

- Python 3.10 or newer.
- `ffmpeg` and `ffprobe` on the path, for decoding and probing.
- `fpcalc` (Chromaprint) on the path, for audio fingerprinting.

Python dependencies (opencv, numpy, scipy, bottle) install from `pyproject.toml`.

## Install

```
pip install -e .
```

## Command line

Compare two files and print the result as JSON:

```
videodiff A.mp4 B.mkv
```

Options:

- `-g`, `--granularity` — comparison resolution in seconds (default: 2.0). Smaller values find shorter divergences at the cost of time.
- `--pretty` — indent the JSON output.
- `--ffmpeg-path`, `--fpcalc-path` — point at non-default binary locations.

The exit code is 0 when the files match (byte-identical or transcode) and 1 when they do not.

## Web viewer

The web interface adds the scrub viewer, the main tool for checking a result. Start it with:

```
videodiff-server
```

Then open `http://127.0.0.1:9080`. Enter two file paths and run the comparison. The segment timeline appears as colored bars for video and audio.

Scrubbing across the timeline shows the actual decoded frame at that timestamp in each file, side by side, with the matching frame in the other file located through the computed mapping. The viewer shows ground truth. If two frames that should align do not, the mapping is wrong and you see it directly, rather than having it smoothed over.

## How it works

Comparison runs in three stages: fingerprint each file, align the fingerprints, then classify the alignment into segments.

**Fingerprinting** reduces each video to a set of compact per-sample signals rather than comparing pixels. ffmpeg decodes the video sequentially at four frames per second, and each sampled frame yields eight visual measurements: brightness, contrast, edge strength, color saturation, hue, and two coarse spatial-layout hashes. Audio runs through fpcalc to produce Chromaprint fingerprints, chunked at the chosen granularity. Fingerprints are cached in `/tmp`, keyed by a fast partial hash of the file, so re-running a comparison reanalyzes nothing.

**Alignment** matches the two fingerprints by shape, not absolute value. Each signal is converted to its normalized first derivative, which survives brightness and contrast changes from transcoding. The aligner cross-correlates the derivatives to find the time offset that best lines the two videos up, then walks forward confirming the match window by window. Running the search in both directions and merging the results catches matches that a single forward pass would skip. A linear fit across the matched offsets detects a constant rate shift; when found, B is resampled to A's timeline and the search repeats, kept only if the alignment improves.

**Classification** turns the matched spans into the final segment list. Long matches are split where their internal offset jumps, marking a short insertion or deletion. Boundaries between shared and unique spans are refined toward granularity precision by a focused cross-correlation search. The spaces left between matches become the A-only and B-only segments. Audio is then compared within each shared video span, by Hamming distance over the fingerprint chunks, so a divergent soundtrack surfaces on its own timeline.
