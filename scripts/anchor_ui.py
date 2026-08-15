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
import threading
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

#: The job runs for the best part of a minute — nine frames of inference at ~4 s each — so it runs
#: on a thread and reports where it is. The first version did the work inside the request and put
#: one line of text in a corner panel; the operator's reading of that was "nothing happens", which
#: is the correct reading of a page that shows nothing for fifty seconds.
JOB = SimpleNamespace(running=False, stage="", detail="", step=0, result=None, error=None, what="")


def _set(stage: str, detail: str = "", step: int = 0) -> None:
    JOB.stage, JOB.detail, JOB.step = stage, detail, step


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
    known = {Path(str(r.get("source", ""))).name for r in runs}

    # ONE list, not two. Two look-alike dropdowns side by side, with the primary button acting on
    # the left one, is how "I select a video and press find the camera" becomes "nothing happens".
    items = [{"id": r["clip_id"], "kind": "clip", "frames": r["n_frames"],
              "label": f"{r['clip_id']}  ({r['n_frames']} frames"
                       f"{', anchored' if r['anchors'] else ''}"
                       f"{', solved' if r['solved'] else ''})"}
             for r in runs]
    items += [{"id": v, "kind": "video", "frames": 0,
               "label": f"{v}  (new video — will be added first)"}
              for v in vids if v not in known]
    return JSONResponse({"items": items})


def _job(kind: str, ident: str) -> None:
    """The whole first pass, on a thread, saying where it is at every step."""
    try:
        clip = ident
        if kind == "video":
            _set("adding the video", f"measuring the crop and scanning {ident} for a usable "
                 f"frame — this decodes the clip, so give it a minute", 1)
            clip = str(Path(ident).stem)[:40]
            r = run(["scripts/new_clip_anchor.py", "--video", str(CFG.videos / ident),
                     "--clip-id", clip, "--device", CFG.device, "--server", CFG.camlab,
                     "--camlab-dir", str(CFG.run_dir.parent)])
            if r.returncode != 0:
                JOB.error = (r.stdout + r.stderr).strip()[-600:]
                return
            JOB.result = _score(clip, _anchored_frame(clip), "camera_start.json",
                                (r.stdout + r.stderr).strip())
            return

        d = CFG.run_dir / clip
        frames = sorted((d / "frames").glob("*.jpg"))
        if not frames:
            JOB.error = f"{clip} has no decoded frames."
            return

        from anchor_from_pnlcalib import landmarks  # noqa: PLC0415 - heavy, only on this path

        n = len(frames)
        step = max(1, n // 8)
        checked, best = [], None
        for k, f in enumerate(range(0, n, step), start=1):
            _set("looking for a frame that shows enough of the pitch",
                 f"frame {f} of {n - 1} — {k} of {len(range(0, n, step))} checked, "
                 f"about 4 seconds each", 1)
            uv, _w, _c, _l = landmarks(d / "frames", f, CFG.device)
            spread = float(np.hypot(*(uv.max(0) - uv.min(0)))) if len(uv) >= 2 else 0.0
            checked.append({"frame": f, "n": len(uv), "spread": round(spread)})
            JOB.detail = (f"frame {f} of {n - 1}: {len(uv)} landmarks — best so far "
                          f"{best[1] if best else 0}")
            if best is None or (len(uv), spread) > (best[1], best[2]):
                best = (f, len(uv), spread)

        ranked = sorted(checked, key=lambda c: (-c["n"], -c["spread"]))
        usable = [c for c in ranked if c["n"] >= 6 and c["spread"] >= 300.0]
        if not usable:
            top = ranked[0] if ranked else {"frame": 0, "n": 0, "spread": 0}
            JOB.error = (
                f"No frame of {clip} shows enough of the pitch. The best is frame {top['frame']} "
                f"with {top['n']} landmarks spread over {top['spread']} px; a camera needs at "
                f"least 6 over 300 px.\n\nWhat was found:\n"
                + "\n".join(f"  frame {c['frame']}: {c['n']} landmarks over {c['spread']} px"
                             for c in checked)
                + "\n\nThis clip has to be aimed by hand in camlab's viewer.")
            return

        # Try the best few and let the PAINT choose, instead of trusting the landmark count to
        # predict which frame yields a camera. It does not: on `MOR_POR_181952` every frame
        # returns 12-13 landmarks over ~915 px, and the worst line they produce runs from 88 px
        # (refused) to 11 px (a good anchor). The difference is invisible before the fit.
        tries, best_row = usable[:3], None
        for k, c in enumerate(tries, start=1):
            _set("fitting the camera",
                 f"frame {c['frame']} ({c['n']} landmarks) — candidate {k} of {len(tries)}; "
                 f"camlab scores each one against the paint it detected", 2)
            r = run(["scripts/anchor_from_pnlcalib.py", "--clip", clip, "--frame",
                     str(c["frame"]), "--which", "camera_start.json", "--server", CFG.camlab,
                     "--run-dir", str(CFG.run_dir), "--device", CFG.device])
            if r.returncode != 0:
                JOB.detail = f"frame {c['frame']} refused"
                continue
            got = _score(clip, c["frame"], "camera_start.json", r.stdout.strip())
            JOB.detail = f"frame {c['frame']}: worst line {got['worst_line_px']} px"
            if best_row is None or (got["worst_line_px"] or 1e9) < (best_row["worst_line_px"]
                                                                   or 1e9):
                best_row = got
        if best_row is None:
            JOB.error = (
                f"Every candidate frame of {clip} was refused — the landmarks are there but the "
                f"camera they imply does not land on the paint.\n\n"
                + (r.stdout + r.stderr).strip()[-500:])
            return
        if best_row["frame"] != tries[0]["frame"]:
            _set("fitting the camera", f"keeping frame {best_row['frame']}", 2)
            run(["scripts/anchor_from_pnlcalib.py", "--clip", clip, "--frame",
                 str(best_row["frame"]), "--which", "camera_start.json", "--server", CFG.camlab,
                 "--run-dir", str(CFG.run_dir), "--device", CFG.device])
        JOB.result = best_row
        return

    except Exception as exc:  # noqa: BLE001 - a page must say what broke, not go blank
        JOB.error = f"{type(exc).__name__}: {exc}"
    finally:
        JOB.running = False


def _anchored_frame(clip: str) -> int:
    man = CFG.run_dir / clip / "camera_manual.json"
    blob = json.loads(man.read_text()) if man.exists() else {}
    for entries in blob.values():
        for f in entries:
            return int(f)
    return 0


def _score(clip: str, frame: int, which: str, log: str) -> dict:
    res = camlab_json(f"/api/run/{clip}/residual/{frame}?which={which}")
    med = res["median_px"]
    # camlab calls a frame solved under 20 px. Above that the page must not present a green result:
    # the uncropped `14604731` clip came back at 17 px with the focal pinned at its search floor
    # and the camera lying on the grass, and the first version offered to ship it.
    return {"clip": clip, "frame": frame, "which": which, "log": log,
            "median_px": med, "worst_line_px": res["worst_line_px"],
            "n_scored": res["n_scored"],
            "warn": None if (med is not None and med <= 20.0) else
            f"{med} px is outside camlab's 20 px band. Look hard at the overlay before handing "
            f"this over — check the camera is even pointing at the pitch."}


@app.post("/api/go")
def go(body: dict) -> JSONResponse:
    """Start the whole thing for whatever is selected — a clip or a video, one button either way.

    One job at a time, and a second press **attaches** to the one already running rather than
    being refused. The refusal read as "it is not computing" from the outside, which is the worst
    thing a long job can say: it is computing, and the page just would not show it.
    """
    kind, ident = str(body.get("kind", "clip")), str(body["id"])
    if JOB.running:
        return JSONResponse({"started": False, "attached": True, "working_on": JOB.what})
    JOB.running, JOB.result, JOB.error, JOB.what = True, None, None, f"{kind}:{ident}"
    _set("starting", "", 1)
    threading.Thread(target=_job, args=(kind, ident), daemon=True).start()
    return JSONResponse({"started": True})


@app.get("/api/progress")
def progress() -> JSONResponse:
    return JSONResponse({"running": JOB.running, "stage": JOB.stage, "detail": JOB.detail,
                         "step": JOB.step, "result": JOB.result, "error": JOB.error,
                         "what": JOB.what})


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
header{padding:14px 18px;border-bottom:1px solid var(--line);display:flex;gap:12px;
align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0 14px 0 0}
select,button{background:var(--card);color:var(--fg);border:1px solid var(--line);
border-radius:7px;padding:9px 13px;font:inherit}
button{cursor:pointer}
button.p{background:var(--acc);color:#141414;border-color:var(--acc);font-weight:650}
button:disabled{background:#20242d;color:#5b6273;border-color:var(--line);cursor:not-allowed;
font-weight:500}
main{padding:18px;display:grid;gap:16px;grid-template-columns:minmax(0,1fr) 400px}
@media(max-width:1050px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:14px}
.card h2{font-size:12px;margin:0 0 10px;color:var(--mut);font-weight:650;letter-spacing:.07em;
text-transform:uppercase}
#ov{width:100%;border-radius:8px;display:block}
#stage{min-height:320px;display:grid;place-items:center;text-align:center;padding:40px 20px;
border-radius:8px;background:#0e1116}
.spin{width:34px;height:34px;border:3px solid #2a303c;border-top-color:var(--acc);
border-radius:50%;animation:r 900ms linear infinite;margin:0 auto 16px}
@keyframes r{to{transform:rotate(360deg)}}
.stagetxt{font-size:16px;font-weight:600}.stagesub{color:var(--mut);margin-top:7px;font-size:13px}
.step{display:flex;gap:10px;align-items:flex-start;padding:9px 0;
border-bottom:1px solid var(--line)}
.step:last-of-type{border-bottom:0}
.step b{display:block}.step .d{color:var(--mut);font-size:13px}
.n{width:24px;height:24px;border-radius:50%;background:#2a303c;color:var(--mut);flex:0 0 24px;
display:grid;place-items:center;font-size:12px;font-weight:700}
.n.on{background:var(--acc);color:#141414}.n.done{background:var(--ok);color:#0d1f16}
.big{font-size:32px;font-weight:700;font-variant-numeric:tabular-nums}
.msg{padding:11px 13px;border-radius:8px;margin-top:12px;font-size:13px;white-space:pre-wrap}
.msg.err{background:#3a1c1c;color:var(--bad)}.msg.ok{background:#14301f;color:var(--ok)}
.key{display:flex;gap:18px;font-size:12px;color:var(--mut);margin-top:9px}
.sw{display:inline-block;width:22px;height:3px;vertical-align:middle;margin-right:6px}
pre{white-space:pre-wrap;font-size:12px;color:var(--mut);max-height:220px;overflow:auto;margin:0}
</style>
<header>
  <h1>camera for a clip</h1>
  <select id="pick" style="min-width:420px"></select>
  <button id="go" class="p">find the camera</button>
</header>
<main>
  <div class="card">
    <h2 id="lefttitle">the frame, and where this camera puts the pitch</h2>
    <div id="stage"><div><div class="stagetxt">pick a clip and press
      <b>find the camera</b></div>
      <div class="stagesub">about a minute on the processor. A new video is added first.</div>
    </div></div>
    <img id="ov" alt="" style="display:none">
    <div class="key" id="key" style="display:none">
      <span><i class="sw" style="background:#00e6ff"></i>pitch model through this camera</span>
      <span><i class="sw" style="background:#ffa000"></i>paint camlab detected</span>
    </div>
    <div id="hint" class="msg" style="display:none"></div>
  </div>
  <div>
    <div class="card"><h2>result</h2><div id="score" class="mut">nothing yet</div></div>
    <div class="card" style="margin-top:16px">
      <h2>what the button does</h2>
      <div class="step"><span class="n" id="s1">1</span><span><b>look for a frame</b>
        <span class="d">a camera needs ≥6 named pitch landmarks spread over ≥300 px. Frame 0 is
        usually not that frame — on one clip it has 1 and frame 630 has 19.</span></span></div>
      <div class="step"><span class="n" id="s2">2</span><span><b>fit the camera</b>
        <span class="d">named landmarks → homography → camlab's own auto-fit finishes it.
        ~4 s per frame, no graphics card.</span></span></div>
      <div class="step"><span class="n" id="s3">3</span><span><b>you judge it</b>
        <span class="d">the cyan pitch model must lie on the white painted lines. That is the
        verdict; the median in pixels is only the number.</span></span></div>
      <div class="step"><span class="n" id="s4">4</span><span><b>hand it to camlab</b>
        <span class="d">it carries this one camera to every other frame, heals what it loses and
        smooths the result.</span></span></div>
      <button id="solve" style="margin-top:13px;width:100%" disabled>
        hand it to camlab (solve the whole clip)</button>
    </div>
    <div class="card" style="margin-top:16px"><h2>log</h2><pre id="log">—</pre></div>
  </div>
</main>
<script>
const $=s=>document.querySelector(s); let S={}, timer=null;
const steps=(i,done)=>{for(let j=1;j<=4;j++){const e=$('#s'+j);
  e.className='n'+(done&&j<=3?' done':j<i?' done':j===i?' on':'');}};
const note=(t,k)=>{const e=$('#hint'); e.style.display=t?'block':'none';
  e.className='msg '+(k||'ok'); e.textContent=t||'';};
const lock=on=>{$('#go').disabled=on; $('#pick').disabled=on;
  if(on) $('#solve').disabled=true;};
function stage(txt,sub,spin){
  $('#ov').style.display='none'; $('#key').style.display='none';
  $('#stage').style.display='grid';
  $('#stage').innerHTML=`<div>${spin?'<div class="spin"></div>':''}`
    +`<div class="stagetxt">${txt}</div><div class="stagesub">${sub||''}</div></div>`;
}
async function jp(u,b){const r=await fetch(u,{method:'POST',
  headers:{'content-type':'application/json'},body:JSON.stringify(b)});
  const j=await r.json().catch(()=>({})); if(!r.ok) throw new Error(j.detail||r.statusText);
  return j;}
async function load(){
  try{const d=await (await fetch('/api/state')).json(); S.items=d.items;
    $('#pick').innerHTML=d.items.map(i=>
      `<option value="${i.kind}:${i.id}">${i.label}</option>`).join('');
  }catch(e){ stage('camlab is not answering on port 8899',
      'start it, then reload this page', false); lock(true); return; }
  // A job may already be running — from another tab, or from before this reload. Show it instead
  // of an idle page that then refuses the button.
  try{ const p=await (await fetch('/api/progress')).json();
    if(p.running){ lock(true); steps(p.step||1); stage(p.stage,p.detail,true); poll(); }
  }catch(e){}
}
load();
$('#go').onclick=async()=>{
  const [kind,...rest]=$('#pick').value.split(':'); const id=rest.join(':');
  lock(true); note(''); $('#log').textContent=''; $('#score').textContent='working…';
  steps(1); stage('starting…','',true);
  try{ const j=await jp('/api/go',{kind,id});
    if(j.attached) note('one was already running ('+j.working_on+') — showing that one.','ok');
    poll(); }
  catch(e){ lock(false); stage('could not start','',false); note(e.message,'err'); }
};
function poll(){
  clearInterval(timer);
  timer=setInterval(async()=>{
    let p; try{ p=await (await fetch('/api/progress')).json(); }catch(e){ return; }
    if(p.running){ steps(p.step||1); stage(p.stage, p.detail, true); return; }
    clearInterval(timer); lock(false);
    if(p.error){ steps(1); stage('no camera for this clip','',false);
      $('#score').textContent='—'; note(p.error,'err'); return; }
    if(!p.result) return;
    const d=p.result; S.res=d; steps(3,true);
    $('#stage').style.display='none'; $('#ov').style.display='block';
    $('#key').style.display='flex';
    $('#ov').src=`/api/overlay.png?clip=${d.clip}&frame=${d.frame}`
      +`&which=${d.which}&t=${Date.now()}`;
    $('#score').innerHTML=`<div class="big">${d.median_px} px</div>
      <div class="mut">median distance from the painted lines, on ${d.n_scored} samples ·
      worst line ${d.worst_line_px} px</div>
      <div class="mut" style="margin-top:8px">clip <b>${d.clip}</b>, frame ${d.frame}</div>`;
    $('#log').textContent=d.log||'';
    if(d.warn){ note(d.warn,'err'); } else {
      note('Look at the picture: the cyan pitch model must sit on the white painted lines. '
        +'If it does, hand it to camlab. If it is off anywhere, say where.','ok'); }
    $('#solve').disabled=false; await load();
  }, 900);
}
$('#solve').onclick=async()=>{
  lock(true); steps(4);
  stage('camlab is solving the whole clip','carry → self-heal → shared centre → smooth',true);
  try{ const d=await jp('/api/solve',
        {clip:S.res.clip,frame:S.res.frame,which:S.res.which});
    $('#log').textContent=JSON.stringify(d,null,1);
    steps(4,true);
    $('#stage').style.display='none'; $('#ov').style.display='block';
    note('camlab has solved the clip. Open its viewer to scrub every frame: '
      +'http://127.0.0.1:8899/','ok');
  }catch(e){ stage('camlab could not solve it','',false); note(e.message,'err'); }
  lock(false); await load();
};
</script>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    # No-store, because this page is edited and the server restarted while a tab is open, and a
    # cached copy of the previous build is indistinguishable from "it stopped working".
    return HTMLResponse(PAGE, headers={"Cache-Control": "no-store, must-revalidate",
                                       "Pragma": "no-cache"})


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
