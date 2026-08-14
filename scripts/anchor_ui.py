"""Get a camera for a clip, look at it, hand it to camlab. One page, one job, one button.

Replaces `labeller_ui.py`, which was built around the path that came first — a person reviewing a
model's line labels — and kept asking for that after the automatic path made it unnecessary. If a
page offers a button whose failure message is "no label-consistent camera", it is showing its own
internals instead of answering the question the operator has, which is **does this clip have a
camera and does it land on the paint**.

What it does, in the order it happens (`scripts/new_clip_anchor.py` is the same steps as a CLI):

* **add a clip** — measure the crop, find a frame with enough named landmarks, ingest into camlab;
* **find the camera** — PnLCalib's *named* landmarks on that frame → homography → camlab's own
  auto-fit finishes it and camlab's own paint scores it;
* **look at it** — the pitch model projected through the camera, drawn over the frame. That is the
  verdict. The number is only the number;
* **hand it over** — camlab carries the anchor to the rest of the frames.

camlab is not modified. Frames, detected paint, the principal point, the refit and the residual all
come over its HTTP API, and the anchor lands in `camera_manual.json`, the store a human's drag
already writes.

Run::

    cd ~/camlab && .venv/bin/python -m uvicorn camlab.server.app:app --port 8899 &
    PNLCALIB_REPO=~/repos/PnLCalib PNLCALIB_WEIGHTS_KP=models/pnlcalib/SV_kp \\
    PNLCALIB_WEIGHTS_LINES=models/pnlcalib/SV_lines \\
    .venv/bin/python scripts/anchor_ui.py          # then open http://127.0.0.1:8811
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

app = FastAPI(title="anchor")
CFG = SimpleNamespace(camlab="http://127.0.0.1:8899", run_dir=Path("/home/chubuchnyi/camlab/runs"),
                      videos=Path("samples/video"), device="cpu")


def camlab_json(path: str, method: str = "GET", body: dict | None = None) -> dict:
    """Call camlab, and pass **its own sentence** through on failure, not the status code.

    Its routes answer 404 for at least three different things — a frame outside the clip, a camera
    file the clip does not have, a frame that failed to decode — and each says which in the body.
    """
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(  # noqa: S310 - configured host
        f"{CFG.camlab}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:  # noqa: S310
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail", "")
        except (ValueError, OSError):
            detail = ""
        raise HTTPException(502, f"camlab: {detail or e.reason}") from e
    except urllib.error.URLError as e:
        raise HTTPException(502, f"camlab is not answering on {CFG.camlab} — is it running? "
                                 f"({e.reason})") from e


def run(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *cmd], capture_output=True, text=True,
                          timeout=timeout, check=False, cwd=Path(__file__).resolve().parent.parent)


@app.get("/api/state")
def state() -> JSONResponse:
    """Everything the form needs so it cannot offer an impossible request."""
    runs = camlab_json("/api/runs")
    for r in runs:
        d = CFG.run_dir / str(r.get("clip_id", ""))
        cams = sorted(p.name for p in d.glob("camera_*.json")) if d.is_dir() else []
        r["cameras"] = [c for c in cams if c != "camera_manual.json"]
        r["solved"] = any(c in cams for c in ("camera_smooth.json", "camera_fixed.json"))
        man = d / "camera_manual.json"
        r["anchors"] = sum(len(v) for v in json.loads(man.read_text()).values()) \
            if man.exists() else 0
    vids = sorted(p.name for p in CFG.videos.glob("*.mp4")) if CFG.videos.is_dir() else []
    known = {str(r.get("source", "")) for r in runs}
    return JSONResponse({"clips": runs, "videos": [v for v in vids if v not in known]})


@app.post("/api/add")
def add(body: dict) -> JSONResponse:
    """Measure the crop, find a solvable frame, ingest, and anchor it — the whole first pass."""
    video = CFG.videos / str(body["video"])
    if not video.exists():
        raise HTTPException(400, f"{video} not found")
    clip = str(body.get("clip_id") or video.stem)[:40]
    r = run(["scripts/new_clip_anchor.py", "--video", str(video), "--clip-id", clip,
             "--device", CFG.device, "--server", CFG.camlab,
             "--camlab-dir", str(CFG.run_dir.parent)])
    return JSONResponse({"clip": clip, "ok": r.returncode == 0,
                         "log": (r.stdout + r.stderr).strip()})


@app.post("/api/anchor")
def anchor(body: dict) -> JSONResponse:
    """Find the camera for one already-ingested clip: probe its frames, then anchor the best."""
    clip = str(body["clip"])
    which = str(body.get("which") or "camera_start.json")
    d = CFG.run_dir / clip
    n = len(list((d / "frames").glob("*.jpg")))
    if not n:
        raise HTTPException(400, f"{clip} has no decoded frames")

    from anchor_from_pnlcalib import landmarks  # noqa: PLC0415 - heavy, only on this path

    best, rows = None, []
    for f in range(0, n, max(1, n // 8)):
        uv, _world, _conf, _lines = landmarks(d / "frames", f, CFG.device)
        spread = float(np.hypot(*(uv.max(0) - uv.min(0)))) if len(uv) >= 2 else 0.0
        rows.append({"frame": f, "n": len(uv), "spread": round(spread)})
        if best is None or (len(uv), spread) > (best[1], best[2]):
            best = (f, len(uv), spread)
    if best is None or best[1] < 6 or best[2] < 300.0:
        got = f"{best[1]} landmarks over {best[2]:.0f} px" if best else "nothing"
        raise HTTPException(
            400,
            f"No frame of {clip} shows enough of the pitch: the best is frame "
            f"{best[0] if best else 0} with {got}, and a camera needs at least 6 spread over "
            f"300 px. This clip needs aiming by hand in camlab's viewer.")

    r = run(["scripts/anchor_from_pnlcalib.py", "--clip", clip, "--frame", str(best[0]),
             "--which", which, "--server", CFG.camlab, "--run-dir", str(CFG.run_dir),
             "--device", CFG.device])
    if r.returncode != 0:
        raise HTTPException(400, (r.stdout + r.stderr).strip()[-400:])
    res = camlab_json(f"/api/run/{clip}/residual/{best[0]}?which={which}")
    return JSONResponse({"clip": clip, "frame": best[0], "which": which, "probe": rows,
                         "landmarks": best[1], "median_px": res["median_px"],
                         "worst_line_px": res["worst_line_px"], "n_scored": res["n_scored"],
                         "log": r.stdout.strip()})


@app.post("/api/solve")
def solve(body: dict) -> JSONResponse:
    """Hand the anchor to camlab: carry → self-heal → shared centre → smooth, its whole chain."""
    clip, frame = str(body["clip"]), int(body.get("frame", 0))
    which = str(body.get("which") or "camera_start.json")
    out = camlab_json(f"/api/run/{clip}/solve?anchor={frame}&seed={which}", method="POST", body={})
    return JSONResponse(out)


@app.get("/api/overlay.png")
def overlay(clip: str, frame: int, which: str = "camera_start.json") -> Response:
    """The frame with the pitch model projected through the camera, over the paint it must match.

    **This is the verdict, and the number is only the number.** Yellow is the pitch model as this
    camera places it; cyan is the paint camlab actually detected. Where they separate, the camera
    is wrong there — which a median cannot tell you, because a marking pivoted about the middle of
    its overlap reports an offset of zero and is far out at both ends.
    """
    data = camlab_json(f"/api/run/{clip}/lines/{frame}?which={which}")
    url = f"{CFG.camlab}/api/run/{clip}/frame/{frame}"
    try:
        with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310 - configured host
            img = cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
    except urllib.error.URLError as e:
        raise HTTPException(502, f"could not fetch the frame: {e.reason}") from e
    if img is None:
        raise HTTPException(502, "frame did not decode")

    for s in np.asarray(data.get("segments", []), float).reshape(-1, 4):
        cv2.line(img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (0, 200, 255), 6, cv2.LINE_AA)
    for e in data.get("lines", []):
        pts = np.asarray(e.get("model") or [], float).reshape(-1, 2)
        if len(pts) >= 2:
            cv2.polylines(img, [pts.astype(np.int32)], False, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.polylines(img, [pts.astype(np.int32)], False, (255, 230, 0), 3, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "encode failed")
    return Response(buf.tobytes(), media_type="image/png")


PAGE = """
<title>camera for a clip</title>
<style>
:root{--bg:#12141a;--fg:#e6e8ee;--mut:#8b93a7;--acc:#ffd24a;--ok:#5ddc9a;--bad:#ff7b72;
--card:#1b1f28;--line:#2a303c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-sans-serif,system-ui,sans-serif}
header{padding:14px 18px;border-bottom:1px solid var(--line);display:flex;gap:10px;
align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0 14px 0 0}
select,input,button{background:var(--card);color:var(--fg);border:1px solid var(--line);
border-radius:7px;padding:8px 11px;font:inherit}
button{cursor:pointer}button.p{background:var(--acc);color:#141414;border-color:var(--acc);
font-weight:650}button:disabled{opacity:.4;cursor:default}
main{padding:18px;display:grid;gap:16px;grid-template-columns:minmax(0,1fr) 420px}
@media(max-width:1050px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:14px}
.card h2{font-size:12px;margin:0 0 10px;color:var(--mut);font-weight:650;letter-spacing:.07em;
text-transform:uppercase}
#ov{width:100%;border-radius:8px;display:block;background:#000}
.step{display:flex;gap:10px;align-items:flex-start;padding:9px 0;
border-bottom:1px solid var(--line)}
.step:last-child{border-bottom:0}
.step b{display:block}.step .d{color:var(--mut);font-size:13px}
.n{width:24px;height:24px;border-radius:50%;background:#2a303c;color:var(--mut);flex:0 0 24px;
display:grid;place-items:center;font-size:12px;font-weight:700}
.n.on{background:var(--acc);color:#141414}.n.done{background:var(--ok);color:#0d1f16}
.big{font-size:30px;font-weight:700;font-variant-numeric:tabular-nums}
.msg{padding:10px 12px;border-radius:8px;margin-top:10px;font-size:13px;white-space:pre-wrap}
.msg.err{background:#3a1c1c;color:var(--bad)}.msg.ok{background:#14301f;color:var(--ok)}
.msg.warn{background:#3a2f14;color:var(--acc)}
.key{display:flex;gap:16px;font-size:12px;color:var(--mut);margin-top:8px}
.sw{display:inline-block;width:22px;height:3px;vertical-align:middle;margin-right:5px}
pre{white-space:pre-wrap;font-size:12px;color:var(--mut);max-height:230px;overflow:auto;margin:0}
</style>
<header>
  <h1>camera for a clip</h1>
  <select id="clip" style="min-width:230px"></select>
  <button id="find" class="p">find the camera</button>
  <span style="width:14px"></span>
  <select id="video" style="min-width:220px"></select>
  <button id="add">add this video</button>
</header>
<main>
  <div class="card">
    <h2>the frame, and where this camera puts the pitch</h2>
    <img id="ov" alt="nothing yet">
    <div class="key">
      <span><i class="sw" style="background:#00e6ff"></i>pitch model through this camera</span>
      <span><i class="sw" style="background:#ffa000"></i>paint camlab detected</span>
    </div>
    <div id="hint" class="msg warn" style="display:none"></div>
  </div>
  <div>
    <div class="card">
      <h2>result</h2>
      <div id="score" class="mut">press <b>find the camera</b></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>what happens when you press it</h2>
      <div class="step"><span class="n" id="s1">1</span><span><b>look for a frame</b>
        <span class="d">a camera needs ≥6 named pitch landmarks spread over ≥300 px. Frame 0 is
        usually not that frame — on one clip it has 1, frame 630 has 19.</span></span></div>
      <div class="step"><span class="n" id="s2">2</span><span><b>fit the camera</b>
        <span class="d">named landmarks → homography → camlab's own auto-fit finishes it.
        ~4 s per frame on the processor, no graphics card.</span></span></div>
      <div class="step"><span class="n" id="s3">3</span><span><b>you judge it</b>
        <span class="d">the cyan pitch model must lie on the white painted lines. That is the
        verdict; the median in pixels is only the number.</span></span></div>
      <div class="step"><span class="n" id="s4">4</span><span><b>hand it to camlab</b>
        <span class="d">it carries this one camera to every other frame, heals what it loses,
        and smooths the result.</span></span></div>
      <button id="solve" class="p" style="margin-top:12px;width:100%" disabled>
        hand it to camlab (solve the whole clip)</button>
    </div>
    <div class="card" style="margin-top:16px"><h2>log</h2><pre id="log">—</pre></div>
  </div>
</main>
<script>
const $=s=>document.querySelector(s); let S={};
const mark=(i,k)=>{for(let j=1;j<=4;j++){const e=$('#s'+j);
  e.className='n'+(j<i?' done':j===i?' on':'');} if(k==='clear') $('#s1').className='n';};
const note=(t,k)=>{const e=$('#hint'); e.style.display=t?'block':'none';
  e.className='msg '+(k||'warn'); e.textContent=t||'';};
async function jp(u,b){const r=await fetch(u,{method:'POST',
  headers:{'content-type':'application/json'},body:JSON.stringify(b)});
  const j=await r.json().catch(()=>({})); if(!r.ok) throw new Error(j.detail||r.statusText);
  return j;}
async function load(){
  try{const d=await (await fetch('/api/state')).json(); S.clips=d.clips;
    $('#clip').innerHTML=d.clips.map(c=>`<option value="${c.clip_id}">${c.clip_id}`
      +`${c.anchors?' · anchored':''}${c.solved?' · solved':''}</option>`).join('');
    $('#video').innerHTML=d.videos.length
      ? d.videos.map(v=>`<option>${v}</option>`).join('')
      : '<option value="">— every video here is already a clip —</option>';
  }catch(e){note('camlab is not answering on port 8899. Start it and reload.','err');}
}
load();
$('#find').onclick=async()=>{
  const clip=$('#clip').value; $('#find').disabled=true; $('#solve').disabled=true;
  mark(1); note('');
  $('#score').innerHTML='looking for a frame that shows enough of the pitch — about half a minute.';
  $('#log').textContent='';
  try{
    const d=await jp('/api/anchor',{clip});
    S.res=d; mark(3);
    $('#score').innerHTML=`<div class="big">${d.median_px} px</div>
      <div class="mut">median distance from the painted lines, on ${d.n_scored} samples ·
      worst line ${d.worst_line_px} px</div>
      <div class="mut" style="margin-top:8px">frame ${d.frame}, ${d.landmarks} landmarks</div>`;
    $('#ov').src=`/api/overlay.png?clip=${clip}&frame=${d.frame}&which=${d.which}&t=${Date.now()}`;
    $('#log').textContent=d.log||'';
    note('Now look at the picture: the cyan pitch model must sit on the white painted lines. '
      +'If it does, hand it to camlab. If it is off anywhere, do not — say where.','ok');
    $('#solve').disabled=false;
  }catch(e){ mark(1,'clear'); $('#score').textContent='no camera for this clip.';
    note(e.message,'err'); }
  $('#find').disabled=false;
};
$('#add').onclick=async()=>{
  const v=$('#video').value; if(!v){note('nothing to add.','warn');return;}
  $('#add').disabled=true; note('');
  $('#score').innerHTML=`adding <b>${v}</b>: measuring the crop, scanning for a usable frame, `
    +`decoding. A minute or so.`;
  try{ const d=await jp('/api/add',{video:v}); $('#log').textContent=d.log;
    await load(); $('#clip').value=d.clip;
    if(d.ok){ note('added and anchored. Press "find the camera" to see it, or hand it over.','ok');
      mark(3);} else { note('added, but no camera — see the log.','err'); }
  }catch(e){ note(e.message,'err'); }
  $('#add').disabled=false;
};
$('#solve').onclick=async()=>{
  $('#solve').disabled=true; mark(4);
  $('#score').innerHTML+='<div class="mut" style="margin-top:8px">camlab is solving…</div>';
  try{ const d=await jp('/api/solve',{clip:S.res.clip,frame:S.res.frame,which:S.res.which});
    $('#log').textContent=JSON.stringify(d,null,1);
    note('camlab has solved the clip. Open its viewer to scrub through every frame: '
      +'http://127.0.0.1:8899/','ok');
  }catch(e){ note(e.message,'err'); $('#solve').disabled=false; }
};
</script>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(PAGE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8811)
    ap.add_argument("--camlab", default="http://127.0.0.1:8899")
    ap.add_argument("--run-dir", default="/home/chubuchnyi/camlab/runs")
    ap.add_argument("--videos", default="samples/video")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    CFG.camlab = args.camlab.rstrip("/")
    CFG.run_dir = Path(args.run_dir)
    CFG.videos = Path(args.videos)
    CFG.device = args.device

    import uvicorn
    print(f"camera page on http://{args.host}:{args.port}   (camlab: {CFG.camlab})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
