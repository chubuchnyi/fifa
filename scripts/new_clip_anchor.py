"""A video file in, a camlab anchor out. One command, nobody in the room.

Everything between "here is an mp4" and "camlab has a first camera", in the order the steps have
to happen, each one because a measurement said so:

1. **Measure the crop.** `adapters.io.framing.measure_framing` finds the widest 16:9 window centred
   on the grass. On a portrait phone clip this is the difference between a solve and nothing:
   measured here, `14604680` goes from 45 % grass to 93 %, and its optical axis moves to
   ``cy = -145`` — outside the image, which camlab derives correctly and pitch3d historically did
   not. Returns ``None`` when no row reaches the grass floor, and then the full frame is used and
   the calibration is left to refuse on its own terms.

2. **Find a frame worth solving.** PnLCalib returns *named* landmarks, and four well-spread ones
   are a homography — but which frame has them is not a property anyone can guess. Measured on
   `14604680`: frame 0 gives **1** keypoint, frame 630 gives **19**. Picking frame 0 by convention
   cost two rounds before this probe existed. So sample the source, count, and take the best.

3. **Ingest into camlab** with that crop and a window centred on that frame, then write its default
   start camera.

4. **Make the anchor** from the named landmarks and let camlab's own auto-fit finish it and its own
   paint score it.

camlab is driven, never modified: its `ingest` and `start_camera.py` are called in its own venv,
and the anchor lands in `camera_manual.json`, the store a human's drag already writes.

Run::

    PNLCALIB_REPO=~/repos/PnLCalib \\
    PNLCALIB_WEIGHTS_KP=models/pnlcalib/SV_kp \\
    PNLCALIB_WEIGHTS_LINES=models/pnlcalib/SV_lines \\
    .venv/bin/python scripts/new_clip_anchor.py --video samples/video/<clip>.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pitch3d.adapters.io.framing import measure_framing  # noqa: E402

#: Sample the source every this many frames when looking for a solvable one. ~3 s of CPU each, so
#: a 700-frame clip costs about half a minute — cheap against the round it saves.
PROBE_STRIDE = 90
#: A homography needs four correspondences, and four that sit on top of each other are one. This is
#: the diagonal of the landmarks' bounding box, in px, below which a frame is not worth ingesting:
#: `demo_14604680` f55 gave 7 keypoints inside a small patch and the focal pinned at its 300 px
#: search floor, which is a bound being hit and therefore a finding, not a camera.
MIN_SPREAD_PX = 300.0
MIN_KEYPOINTS = 6


def probe(video: Path, crop: tuple[int, int, int, int] | None, stride: int,
          device: str) -> list[tuple[int, int, float]]:
    """``(frame, n_keypoints, spread_px)`` for every sampled frame, best first."""
    from pitch3d.adapters.models.pnlcalib_backend import make

    backend = make()
    backend.device = device
    s = backend._load()  # noqa: SLF001
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out: list[tuple[int, int, float]] = []
    for f in range(0, max(total, 1), stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        if crop is not None:
            cw, ch, cx0, cy0 = crop
            frame = frame[cy0:cy0 + ch, cx0:cx0 + cw]
        kp, _lines, w, h = backend._infer_frame(s, frame)  # noqa: SLF001
        spread = 0.0
        if len(kp) >= 2:
            p = np.array([[d["x"] * w, d["y"] * h] for d in kp.values()])
            spread = float(np.hypot(*(p.max(0) - p.min(0))))
        out.append((f, len(kp), spread))
        print(f"    frame {f:5d}   keypoints {len(kp):3d}   spread {spread:6.0f} px")
    cap.release()
    return sorted(out, key=lambda r: (-r[1], -r[2]))


def camlab(args: list[str], camlab_dir: Path) -> str:
    """Run something in camlab's own venv. It is a tool here, not a library to import."""
    r = subprocess.run([str(camlab_dir / ".venv/bin/python"), *args], cwd=camlab_dir,
                       capture_output=True, text=True, timeout=1800, check=False)
    if r.returncode != 0:
        raise SystemExit(f"camlab failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--clip-id", default=None, help="defaults to the file stem")
    ap.add_argument("--frames", type=int, default=60, help="how many frames to ingest")
    ap.add_argument("--stride", type=int, default=PROBE_STRIDE)
    ap.add_argument("--camlab-dir", type=Path, default=Path("/home/chubuchnyi/camlab"))
    ap.add_argument("--server", default="http://127.0.0.1:8899")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-crop", action="store_true", help="ingest the full frame")
    args = ap.parse_args()

    video = args.video.resolve()
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in video.stem)
    clip = args.clip_id or safe[:40]
    if not video.exists():
        raise SystemExit(f"{video} does not exist")

    print(f"[1/4] measuring the crop for {video.name}")
    crop = None
    if not args.no_crop:
        fr = measure_framing(str(video), first=0, last=None)
        if fr is None:
            print("      no row reaches the grass floor — keeping the full frame")
        else:
            crop = fr.rect
            print(f"      crop {crop[0]}x{crop[1]}+{crop[2]}+{crop[3]}   "
                  f"grass {fr.grass_before:.0%} -> {fr.grass_after:.0%}")

    print(f"[2/4] looking for a frame with enough named landmarks (every {args.stride})")
    ranked = probe(video, crop, args.stride, args.device)
    if not ranked or ranked[0][1] < MIN_KEYPOINTS or ranked[0][2] < MIN_SPREAD_PX:
        best = ranked[0] if ranked else (0, 0, 0.0)
        print(f"      best is frame {best[0]} with {best[1]} landmarks over {best[2]:.0f} px — "
              f"below {MIN_KEYPOINTS} / {MIN_SPREAD_PX:.0f} px, so no anchor is offered.")
        print("      This clip needs a hand-aimed frame in camlab's viewer, or a denser probe "
              f"(--stride {max(args.stride // 3, 5)}).")
        return 2
    target, n_kp, spread = ranked[0]
    print(f"      frame {target}: {n_kp} landmarks over {spread:.0f} px")

    first = max(target - args.frames // 2, 0)
    local = target - first
    print(f"[3/4] ingesting {args.frames} frames from {first} into camlab as '{clip}'")
    crop_arg = "None" if crop is None else str(tuple(crop))
    camlab(["-c", "import sys; sys.path.insert(0,'src')\n"
            "from camlab.io.ingest import ingest\n"
            "from pathlib import Path\n"
            f"i=ingest(Path(r'{video}'), '{clip}', first={first}, n_frames={args.frames}, "
            f"crop={crop_arg})\n"
            "print('      ', i.clip_id, i.width, 'x', i.height, 'first', i.first_frame)"],
           args.camlab_dir)
    camlab(["scripts/start_camera.py", clip], args.camlab_dir)

    print(f"[4/4] anchor on local frame {local} (source frame {target})")
    r = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "anchor_from_pnlcalib.py"),
         "--clip", clip, "--frame", str(local), "--which", "camera_start.json",
         "--server", args.server, "--run-dir", str(args.camlab_dir / "runs"),
         "--device", args.device],
        capture_output=True, text=True, timeout=1800, check=False)
    print(r.stdout.rstrip() or r.stderr.rstrip())
    if r.returncode != 0:
        return r.returncode

    print(f"\n  done. Judge it by eye: {args.server}/  →  clip '{clip}' →  camera_start.json "
          f"→ frame {local}")
    print("  then press 'solve this clip' there; camlab carries it to the rest of the frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
