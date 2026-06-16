from __future__ import annotations

import argparse
import json
import os
import sys

from .core import run_comparison
from .models import MatchType


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two video files and report differences.",
    )
    parser.add_argument("file_a", help="Path to the first video file")
    parser.add_argument("file_b", help="Path to the second video file")
    parser.add_argument(
        "-g", "--granularity",
        type=float,
        default=2.0,
        help="Comparison granularity in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default="ffmpeg",
        help="Path to ffmpeg binary (default: ffmpeg)",
    )
    parser.add_argument(
        "--fpcalc-path",
        default="fpcalc",
        help="Path to fpcalc binary (default: fpcalc)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )

    args = parser.parse_args()

    for path in (args.file_a, args.file_b):
        if not os.path.isfile(path):
            print(f"Error: file not found: {path}", file=sys.stderr)
            return 2

    result = run_comparison(
        path_a=args.file_a,
        path_b=args.file_b,
        granularity=args.granularity,
        ffmpeg_path=args.ffmpeg_path,
        fpcalc_path=args.fpcalc_path,
    )

    indent = 2 if args.pretty else None
    print(json.dumps(result.to_dict(), indent=indent))

    if result.match_type in (MatchType.BYTE, MatchType.TRANSCODE):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
