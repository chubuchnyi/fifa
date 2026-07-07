"""poseannot — pose annotator (browser UI, FastAPI backend).

v0 — read-only: navigate frames, view 2D pose overlay on video, view
matching 3D SMPL-X pose. No editing yet — that's v1.

Entrypoint:
    .venv/bin/uvicorn poseannot.app:app --host 0.0.0.0 --port 8000
"""
