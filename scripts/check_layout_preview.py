"""Does the browser's live layout preview agree with what the server commits? (#127)

The pitch-layout drag is previewed in the browser and committed on the server, and the two
compute the same similarity twice — `_layoutDrag` in ``poseannot/static/index.html`` is a port
of :func:`poseannot.camera.plane_similarity`. Nothing in the type system holds them together,
and a divergence has exactly one symptom: the outline jumps at the moment the operator lets go,
which is the defect #127 existed to remove.

So this drives both. It pulls the SHIPPED JavaScript out of ``index.html`` (rather than a
retyped copy), runs it under node, POSTs the pixel it says to POST, and compares:

  * the three numbers the operator reads — metres, degrees, scale;
  * the *maps* — where the preview sends a grid of frame pixels vs where the server's new
    solve does. Comparing the 3x3s entrywise would be meaningless, since a homography is only
    defined up to scale; comparing images is not.

Measured 2026-08-03 on the Colombia clip: 0.0000 px over 5 drags covering both handles and
both gains. Needs a running server (``.venv/bin/uvicorn poseannot.app:app --port 8000``) and
node; it undoes every edit it makes, so it leaves the scene as it found it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests

BASE = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parents[1]

#: Both handles, both gains, and one turn that shrinks rather than grows.
CASES = [
    {"handle": "move", "uv": [1010, 600], "fine": False},
    {"handle": "move", "uv": [1010, 600], "fine": True},
    {"handle": "turn", "uv": [1600, 700], "fine": False},
    {"handle": "turn", "uv": [1600, 700], "fine": True},
    {"handle": "turn", "uv": [1400, 560], "fine": True},
]

HARNESS = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
function grab(name) {
  const i = src.indexOf(`\n    ${name}(`);
  if (i < 0) throw new Error(`method ${name} not found in index.html`);
  let k = src.indexOf("{", i), depth = 0;
  for (; k < src.length; k++) {
    if (src[k] === "{") depth++;
    else if (src[k] === "}") { depth--; if (depth === 0) break; }
  }
  return src.slice(i, k + 1).trim().replace(/,$/, "");
}
const app = eval(`({ ${grab("_mat3")}, ${grab("_mapH")}, ${grab("_layoutDrag")} })`);
const d = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
Object.assign(app, { pitchW2I: d.w2i, pitchI2W: d.i2w, pitchHandles: d.handles });
console.log(JSON.stringify(JSON.parse(process.argv[4]).map((c) => {
  app._dragPitch = c.handle; app._dragPitchUV = c.uv; app._dragPitchFine = !!c.fine;
  const r = app._layoutDrag();
  return { ...c, uv_post: r.uv, moved_m: r.moved, turn_deg: r.turnDeg, scale: r.scale, a: r.a };
})));
"""


def image(h, pts: np.ndarray) -> np.ndarray:
    q = np.column_stack([pts, np.ones(len(pts))]) @ np.asarray(h, dtype=float).T
    return q[:, :2] / q[:, 2:3]


def main() -> int:
    s = requests.Session()
    s.post(f"{BASE}/login", data={"username": "admin", "password": "physics"}, timeout=10)
    base = s.get(f"{BASE}/api/pitch/calibrated/0", timeout=30).json()
    if "w2i" not in base:
        print("server has no w2i in /api/pitch/calibrated — is it running the current code?")
        return 2
    if base["adjusted"]:
        print("the layout is already hand-adjusted; undo it first so this starts from the solve")
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp, "harness.js")
        harness.write_text(HARNESS)
        state = Path(tmp, "state.json")
        state.write_text(json.dumps({k: base[k] for k in ("w2i", "i2w", "handles")}))
        js = json.loads(subprocess.run(
            ["node", str(harness), str(ROOT / "poseannot/static/index.html"),
             str(state), json.dumps(CASES)],
            capture_output=True, text=True, check=True).stdout)

    gx, gy = np.meshgrid(np.linspace(0, 1920, 9), np.linspace(0, 1080, 7))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    i2w = np.asarray(base["i2w"], dtype=float)

    bad = 0
    for c in js:
        r = s.post(f"{BASE}/api/pitch/adjust", timeout=30,
                   json={"frame": 0, "handle": c["handle"], "uv": c["uv_post"]}).json()
        after = s.get(f"{BASE}/api/pitch/calibrated/0", timeout=30).json()
        s.post(f"{BASE}/api/pitch/adjust/undo", timeout=30)

        dm = abs(r["moved_m"] - c["moved_m"])
        dd = abs(r["turn_deg"] - c["turn_deg"])
        ds = abs(r["scale"] - c["scale"])
        err = float(np.abs(image(c["a"], grid)
                           - image(np.asarray(after["w2i"], dtype=float) @ i2w, grid)).max())
        ok = dm < 0.01 and dd < 0.02 and ds < 1e-4 and err < 0.5
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {c['handle']:5s} fine={str(c['fine']):5s} "
              f"preview {c['moved_m']:6.2f} m {c['turn_deg']:+7.2f}° x{c['scale']:.4f}  "
              f"committed {r['moved_m']:6.2f} {r['turn_deg']:+7.2f} x{r['scale']:.4f}  "
              f"jump {err:.4f} px over {len(grid)} pixels")

    if s.get(f"{BASE}/api/pitch/calibrated/0", timeout=30).json()["adjusted"]:
        print("WARNING: the scene was left hand-adjusted — undo it in the UI")
        bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
