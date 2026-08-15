"""One frame in, one camlab anchor out — from named pitch landmarks, with nobody in the room.

The automatic first camera, doing it the way that needs no vocabulary, no prompt and no review.
PnLCalib's two HRNet heads return landmarks that are **already named**: id → world comes from the
table shipped with the weights, so four of them are a homography and there is nothing to search.
That is the whole difference from `write_camlab_anchor.py`, which asks a model to name camlab's
detected segments and then enumerates every label-consistent assignment.

**One frame per clip is the entire cost.** Measured here 2026-08-10: ~4.2 s/frame on CPU (2.0 s
keypoint head + 2.1 s line head at 8 threads; 16 threads is 3x worse, not better). So the GPU that
this backend is usually fenced behind is not needed for an anchor, and the fence is really only
about licence — PnLCalib is GPL-2.0 and its weights are SoccerNet research-only, so it is imported
by dotted path from a checkout outside this repo and never vendored.

camlab is not modified and not imported: the frame, the principal point and the verdict all come
over its HTTP API, and the anchor is written into `camera_manual.json`, the store `solve/hand.py`
already reads and already refuses to rank by source.

Run (camlab's server up, weights on disk)::

    PNLCALIB_REPO=~/repos/PnLCalib \\
    PNLCALIB_WEIGHTS_KP=models/pnlcalib/SV_kp \\
    PNLCALIB_WEIGHTS_LINES=models/pnlcalib/SV_lines \\
    .venv/bin/python scripts/anchor_from_pnlcalib.py --clip demo_14604680 --frame 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_camlab_anchor import get_json, refine, write_manual  # noqa: E402

from pitch3d.adapters.models.calibration import solve_homography_ransac  # noqa: E402
from pitch3d.core.scene.plane_camera import _decompose, _k_inv, _measure_focal  # noqa: E402

#: RANSAC inlier threshold in **world metres**, not pixels — ADR-0012's reason: a pitch homography
#: is heteroscedastic, so no single pixel scale is an inlier scale. Same value the calibrator uses.
RANSAC_THRESHOLD_M = 1.0
RANSAC_ITERS = 400

#: The focal is searched over these bounds (`plane_camera.FOCAL_BOUNDS`). A result sitting ON one
#: of them is not a lens, it is the search giving up — *a bound that is being hit is a finding, not
#: a setting* (camlab, `inherited-claims.md`). Seen on the uncropped `14604731` clip: 9 landmarks
#: clustered in the stands gave focal 300 px at z = 0.03 m, a "camera" lying on the grass, and
#: without this gate the page presented it as a result and offered to ship it.
FOCAL_BOUNDS = (300.0, 20000.0)
#: A broadcast or phone camera is above the pitch and off it. Deliberately wide — camlab's own
#: bootstrap gate (5–45 m) rejects a phone held at 1.5 m, which its `pitch-level-clips` branch
#: records as rejecting the *true* camera outright.
HEIGHT_BOUNDS = (1.0, 80.0)
#: camlab calls a frame solved under 20 px. Above it the anchor is still written — the eye is the
#: judge and a bad anchor loses to the seed on the paint anyway — but the caller is told plainly.
BAND_PX = 20.0


def landmarks(clip_frames_dir: Path, frame: int, device: str) -> tuple[np.ndarray, np.ndarray,
                                                                      np.ndarray, dict]:
    """Run PnLCalib on ONE frame and return ``(image_uv, world_xy, confidence, line_obs)``.

    Goes through the shipped adapter rather than a second copy of the inference: it owns the
    id→world table import (so the mapping always matches the weights actually loaded) and the
    letterbox fix, without which a portrait clip reaches the heads squashed 0.5x by 0.28x.
    """
    from pitch3d.adapters.models.pnlcalib_backend import make

    backend = make()
    backend.device = device
    s = backend._load()  # noqa: SLF001 - one frame, deliberately not the whole-clip iterator
    path = clip_frames_dir / f"{frame:06d}.jpg"
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise SystemExit(f"could not read {path}")
    kp_dict, lines_dict, w_orig, h_orig = backend._infer_frame(s, bgr)  # noqa: SLF001

    uv, world, conf = [], [], []
    for kid, d in kp_dict.items():
        n_main = 57
        wp = s["kw"][kid - 1] if kid <= n_main else s["ka"][kid - 1 - n_main]
        uv.append([d["x"] * w_orig, d["y"] * h_orig])
        world.append([float(wp[0]), -float(wp[1])])  # #118: template Y negated into our Z-up world
        conf.append(float(d.get("p", 1.0)))
    l_uv, l_abc, l_conf = backend._line_observations(lines_dict, w_orig, h_orig)  # noqa: SLF001
    return (np.asarray(uv, float).reshape(-1, 2), np.asarray(world, float).reshape(-1, 2),
            np.asarray(conf, float).reshape(-1),
            {"uv": l_uv, "abc": l_abc, "conf": l_conf, "w": w_orig, "h": h_orig})


def _restore(store: Path, backup: str | None) -> None:
    """Put camlab's store back exactly as it was. A refusal must leave no trace."""
    if backup is None:
        store.unlink(missing_ok=True)
    else:
        store.write_text(backup)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--which", default="camera_start.json", help="the solve the anchor overlays")
    ap.add_argument("--server", default="http://127.0.0.1:8899")
    ap.add_argument("--run-dir", default="/home/chubuchnyi/camlab/runs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-lines", action="store_true", help="points only, no point-on-line rows")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = f"{args.server}/api/run/{args.clip}"
    cam = get_json(f"{base}/camera?which={args.which}")
    cx, cy = float(cam["cx"]), float(cam["cy"])
    run = Path(args.run_dir) / args.clip

    image_uv, world_xy, conf, lines = landmarks(run / "frames", args.frame, args.device)
    print(f"{args.clip} f{args.frame}: PnLCalib returned {len(image_uv)} named keypoints "
          f"(mean conf {conf.mean():.2f} )" if len(image_uv) else
          f"{args.clip} f{args.frame}: PnLCalib returned NO keypoints")
    print(f"  {len(lines['uv'])} point-on-line observations; principal point ({cx:.1f}, {cy:.1f}) "
          f"from camlab")
    if len(image_uv) < 4:
        print("  fewer than four named landmarks — no homography. Try another frame.")
        return 1

    kw = {} if args.no_lines or not len(lines["uv"]) else {
        "line_uv": lines["uv"], "line_abc": lines["abc"], "line_weights": lines["conf"]}
    h_i2w, inliers = solve_homography_ransac(
        image_uv, world_xy, weights=conf, threshold=RANSAC_THRESHOLD_M,
        max_iters=RANSAC_ITERS, seed=0, **kw)
    print(f"  homography from {int(np.sum(inliers))}/{len(inliers)} inlier landmarks")

    h_w2i = np.linalg.inv(h_i2w)
    focal = _measure_focal([h_w2i], cx, cy)
    rot, t = _decompose(h_w2i, _k_inv(focal, cx, cy))
    centre = -rot.T @ t
    entry = {"focal_px": float(focal), "rotation": cv2.Rodrigues(rot)[0].ravel().tolist(),
             "position": [float(v) for v in centre]}
    print(f"  camera: focal {focal:.0f} px, position {np.round(centre, 2).tolist()} m")

    edge = 0.01 * (FOCAL_BOUNDS[1] - FOCAL_BOUNDS[0])
    if not (FOCAL_BOUNDS[0] + edge < focal < FOCAL_BOUNDS[1] - edge):
        print(f"  REFUSED: the focal came back at {focal:.0f} px, on the edge of the "
              f"{FOCAL_BOUNDS[0]:.0f}–{FOCAL_BOUNDS[1]:.0f} px search range. That is the search "
              f"giving up, not a lens — the landmarks do not pin a camera on this frame.")
        return 3
    if not (HEIGHT_BOUNDS[0] < centre[2] < HEIGHT_BOUNDS[1]):
        print(f"  REFUSED: this puts the camera {centre[2]:.2f} m above the pitch, outside "
              f"{HEIGHT_BOUNDS[0]:.0f}–{HEIGHT_BOUNDS[1]:.0f} m. Nobody filmed from there.")
        return 3

    store = run / "camera_manual.json"
    backup = store.read_text() if store.exists() else None
    write_manual(run, args.which, args.frame, entry)
    raw = get_json(f"{base}/residual/{args.frame}?which={args.which}")
    ref = refine(base, args.which, args.frame)
    got = get_json(f"{base}/residual/{args.frame}?which={args.which}")
    final = json.loads(store.read_text())[args.which][str(args.frame)]
    print(f"  raw aim      : median {raw['median_px']} px on {raw['n_scored']} samples")
    print(f"  after refit  : median {got['median_px']} px, worst line {got['worst_line_px']} px "
          f"on {got['n_scored']} samples  (moved={ref.get('moved')})")
    print(f"  final camera : focal {final['focal_px']:.0f} px, "
          f"position {np.round(final['position'], 2).tolist()} m")

    # camlab's own rule, which this script read and then did not apply: "worst_line_px first,
    # because it is the verdict. A pooled median cannot show a camera sitting on one family of
    # lines while the family parallel to it is metres off." On `MOR_POR_181952` f7 the median was
    # 11.3 px and two of the five scored markings sat at 87 and 70 px — inside the band by the
    # median, nowhere near a camera.
    worst = got["worst_line_px"]
    n_markings = len(got.get("per_line") or {})
    print(f"  support      : {n_markings} markings scored, coverage {got.get('coverage')}")
    if worst is None or worst > BAND_PX:
        print(f"  REFUSED: worst line {worst} px is outside camlab's {BAND_PX:.0f} px band. The "
              f"median ({got['median_px']} px) is not the verdict — a camera can sit on one family "
              f"of lines and be metres off the family parallel to it.")
        _restore(store, backup)
        return 3

    # camlab's auto-fit reports whether it could pair detected segments with projected markings.
    # Worth saying, **not worth refusing on**: it was a hard refusal for one commit, and it threw
    # away `MOR_POR_181952` f59 — a camera at 7.3 px median and 11.04 px worst line, comfortably
    # inside the band. `matched: false` means the refit could not *improve* the aim, not that the
    # aim is wrong, and the verdict here is the paint.
    if not ref.get("matched", True) or ref.get("matched_after", 1) == 0:
        print(f"  note: camlab's auto-fit matched {ref.get('matched_after')} of "
              f"{ref.get('lines')} detected lines (it needs {ref.get('min_matched')}), so it left "
              f"the aim as it was. The paint still scores it, and that is the verdict.")

    if args.dry_run:
        _restore(store, backup)
        print("  store left as it was (dry run)")
    else:
        print(f"  wrote anchor f{args.frame} → {store}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
