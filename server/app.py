from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
import threading
import uuid

from bottle import Bottle, request, response, static_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from videodiff.core import run_comparison

app = Bottle()

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(SERVER_DIR, "static")
VIEWS_DIR = os.path.join(SERVER_DIR, "views")

# In-memory job store
jobs: dict[str, dict] = {}


@functools.lru_cache(maxsize=100)
def _extract_frame(file_path: str, t: float) -> bytes:
    """Extract a single frame as JPEG bytes, 480px wide."""
    cmd = [
        "ffmpeg",
        "-ss", str(t),
        "-i", file_path,
        "-frames:v", "1",
        "-vf", "scale=480:-2",
        "-f", "image2",
        "-c:v", "mjpeg",
        "-q:v", "5",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=10)
    if result.returncode != 0 or not result.stdout:
        return b""
    return result.stdout


@app.route("/")
def index():
    with open(os.path.join(VIEWS_DIR, "index.html")) as f:
        return f.read()


@app.route("/static/<filepath:path>")
def serve_static(filepath):
    return static_file(filepath, root=STATIC_DIR)


def _run_job(job_id: str, file_a: str, file_b: str, granularity: float, force: bool = False):
    """Run comparison in a background thread, updating job status."""
    job = jobs[job_id]

    def progress_cb(stage, detail):
        job["stage"] = stage
        job["detail"] = detail

    try:
        result = run_comparison(
            path_a=file_a,
            path_b=file_b,
            granularity=granularity,
            progress_callback=progress_cb,
            force=force,
        )
        job["status"] = "done"
        job["result"] = result.to_dict()
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@app.route("/api/compare", method="POST")
def api_compare():
    response.content_type = "application/json"
    data = request.json
    if not data or "file_a" not in data or "file_b" not in data:
        response.status = 400
        return json.dumps({"error": "file_a and file_b are required"})

    file_a = data["file_a"]
    file_b = data["file_b"]

    for path in (file_a, file_b):
        if not os.path.isfile(path):
            response.status = 400
            return json.dumps({"error": f"File not found: {path}"})

    granularity = float(data.get("granularity", 2.0))
    force = bool(data.get("force", False))

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "running",
        "stage": "starting",
        "detail": {},
    }

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, file_a, file_b, granularity, force),
        daemon=True,
    )
    thread.start()

    return json.dumps({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    response.content_type = "application/json"
    job = jobs.get(job_id)
    if not job:
        response.status = 404
        return json.dumps({"error": "Job not found"})

    resp = {
        "status": job["status"],
        "stage": job.get("stage", ""),
        "detail": job.get("detail", {}),
    }

    if job["status"] == "done":
        resp["result"] = job["result"]
        del jobs[job_id]
    elif job["status"] == "error":
        resp["error"] = job.get("error", "Unknown error")
        del jobs[job_id]

    return json.dumps(resp)


@app.route("/api/frame")
def api_frame():
    file_path = request.query.get("file", "")
    t_str = request.query.get("t", "0")

    if not file_path or not os.path.isfile(file_path):
        response.status = 400
        return b""

    try:
        t = round(float(t_str), 1)
    except ValueError:
        response.status = 400
        return b""

    data = _extract_frame(file_path, t)
    if not data:
        response.status = 500
        return b""

    response.content_type = "image/jpeg"
    return data


def main():
    parser = argparse.ArgumentParser(description="Video Diff web UI server")
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=9080,
        help="Port to listen on (default: 9080)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    # Import all videodiff modules so Bottle's reloader watches them for changes.
    # Static files (HTML, JS, CSS) are fetched fresh by the browser each request.
    import videodiff.cli
    import videodiff.compare
    import videodiff.core
    import videodiff.fingerprint
    import videodiff.models

    app.run(host=args.host, port=args.port, debug=True, reloader=True)


if __name__ == "__main__":
    main()
