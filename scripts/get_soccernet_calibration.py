"""Download the OPEN SoccerNet camera-calibration dataset (``calibration-2023``) — the B1 asset.

Unlike the broadcast *video* (NDA-gated, needs a signed-NDA password), the calibration challenge's
**frames + per-image pitch-line GT are openly downloadable with no password** — verified against
soccer-net.org/data and the SoccerNet/sn-calibration README on 2026-06-23 (their own getting
-started calls ``downloadDataTask(task="calibration-2023", ...)`` with no ``password=`` argument).
This is the landscape-16:9 real-broadcast set with pitch calibration GT that B1 needs to score our
``FieldCalibrator`` on *independent* footage (``samples/video/clip.mp4`` has faint markings → zero
landmarks). No 3D-pose GT here — this asset is for calibration only.

What it does (idempotent):
  1. ``pip install SoccerNet`` if the package is missing.
  2. ``downloadDataTask(task="calibration-2023", split=[...])`` into ``--root``.
  3. Unzip each ``<split>.zip`` and report the directory that actually holds the ``<id>.jpg`` +
     ``<id>.json`` pairs — the ``--frames-dir`` to hand the calibration bake-off.

Then score it (needs CUDA + the wired calibrator):
  PYTHONPATH=src python scripts/run_calib_eval.py --dataset soccernet \
      --frames-dir <printed frames dir> --limit 200 \
      --backend pitch3d.adapters.models.pnlcalib_backend:make

Run on the pod (persistent volume):
  PATH=/workspace/.venv/bin:$PATH python scripts/get_soccernet_calibration.py \
      --root /workspace/SoccerNet --splits test valid
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


def _ensure_soccernet() -> None:
    """Import the SoccerNet package, pip-installing it into the active venv on first use."""
    try:
        import SoccerNet  # noqa: F401
    except ImportError:
        print("[1/3] installing SoccerNet pip package", file=sys.stderr)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-U", "SoccerNet"], check=True
        )


def _download(root: Path, splits: list[str]) -> None:
    """Pull the calibration-2023 split zips into ``root`` (open data — no NDA password)."""
    from SoccerNet.Downloader import SoccerNetDownloader as SNdl

    print(f"[2/3] downloadDataTask(calibration-2023, {splits}) -> {root}", file=sys.stderr)
    dl = SNdl(LocalDirectory=str(root))
    dl.downloadDataTask(task="calibration-2023", split=splits)


def _unzip(task_dir: Path, splits: list[str]) -> dict[str, Path]:
    """Unzip each ``<split>.zip`` under ``task_dir``; return ``{split: dir-with-JSONs}``."""
    frames_dirs: dict[str, Path] = {}
    for split in splits:
        zip_path = task_dir / f"{split}.zip"
        dest = task_dir / split
        if zip_path.exists() and not dest.exists():
            print(f"[3/3] unzip {zip_path.name} -> {dest}", file=sys.stderr)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
        # The zip may extract flat into <dest> or nest a folder; locate where the JSONs landed.
        candidates = [p.parent for p in dest.rglob("*.json")] if dest.exists() else []
        if candidates:
            frames_dirs[split] = max(set(candidates), key=candidates.count)
    return frames_dirs


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Download open SoccerNet calibration-2023 (B1 asset).")
    p.add_argument("--root", default="/workspace/SoccerNet", help="download + unzip destination")
    p.add_argument("--splits", nargs="+", default=["test", "valid"],
                   choices=["train", "valid", "test", "challenge"],
                   help="splits to fetch (default: test valid)")
    args = p.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    _ensure_soccernet()
    _download(root, args.splits)
    frames_dirs = _unzip(root / "calibration-2023", args.splits)

    print("\n== ready ==")
    if not frames_dirs:
        print("no frames found — check the download under", root / "calibration-2023")
        return
    for split, fdir in frames_dirs.items():
        n = sum(1 for _ in fdir.glob("*.json"))
        print(f"  {split}: {n} annotated frames at {fdir}")
    any_dir = next(iter(frames_dirs.values()))
    print("\nscore it:")
    print("  PYTHONPATH=src python scripts/run_calib_eval.py --dataset soccernet \\")
    print(f"      --frames-dir {any_dir} --limit 200 \\")
    print("      --backend pitch3d.adapters.models.pnlcalib_backend:make")


if __name__ == "__main__":
    main()
