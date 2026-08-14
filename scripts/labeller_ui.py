"""Review the labels, make the anchor, let camlab's paint judge it — a page for one question.

ADR-0013 §3: every lab builds the UI its own question needs, and does not share one. This is that
page for **"which pitch marking is this detected segment, and does the camera it implies land on
the paint"**. It is deliberately not part of `poseannot`, which is frozen, and it adds nothing to
camlab, which stays a tool driven over HTTP.

**Why a review page rather than a button.** The measurement in
`findings/labeller-blind-run-2026-08-13.md` is that a model given close-ups gets 7 of 9 labels
right, and that the anchor survives the two it gets wrong. A person looking at nine tiles fixes
those two in seconds — and that correction is not overhead, it is **the label-producing
instrument**, which is the scarcest input this project has. `is-a-model-worth-training.md` names
hand-placed cameras on diverse clips as the prerequisite for self-labelling and could not start
because there was no cheap way to make them. This is the cheap way.

**What it does not do.** It never shows the answer key. On clips camlab has already solved a key
exists on disk, and showing it would turn every future labelling measurement into a lookup. The
feedback here is the same thing that judges the finished solve: camlab's own paint residual.

Run (camlab's server must be up; see `--camlab`)::

    cd ~/camlab && .venv/bin/python -m uvicorn camlab.server.app:app --port 8899 &
    .venv/bin/python scripts/labeller_ui.py            # then open http://127.0.0.1:8811
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_line_labeller import VOCABULARY, crop_tile, draw  # noqa: E402
from write_camlab_anchor import (  # noqa: E402
    DISTANCE_BOUNDS,
    FOCAL_BOUNDS,
    HEIGHT_BOUNDS,
    SHORTLIST,
    solve_anchor,
    write_manual,
)

app = FastAPI(title="labeller")
CFG = SimpleNamespace(camlab="http://127.0.0.1:8899", run_dir=Path("/home/chubuchnyi/camlab/runs"))
#: One prepared frame at a time: the decoded image and camlab's own segments for it. Keyed so a
#: reload does not silently review one frame's tiles against another frame's segments.
STATE: dict = {}


def camlab_json(path: str) -> dict:
    """GET from camlab, and **pass its own explanation through** rather than the status code.

    The first version reported only `HTTP Error 404: Not Found`, which is true and useless:
    camlab's routes answer 404 for at least three different things — a frame outside the clip, a
    camera file that clip does not have, and a frame that failed to decode — and each says which in
    the response body. Swallowing that turned a one-word fix into a debugging session.
    """
    try:
        with urllib.request.urlopen(f"{CFG.camlab}{path}", timeout=300) as r:  # noqa: S310
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail", "")
        except (ValueError, OSError):
            detail = ""
        raise HTTPException(502, f"camlab said: {detail or e.reason} ({path})") from e
    except urllib.error.URLError as e:
        raise HTTPException(502, f"camlab at {CFG.camlab} is not reachable: {e.reason}") from e


def png(img: np.ndarray) -> Response:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "could not encode")
    return Response(buf.tobytes(), media_type="image/png")


@app.get("/api/clips")
def clips() -> JSONResponse:
    """camlab's runs, plus the two things the form needs to stop offering impossible requests.

    `n_frames` bounds the frame box, and the camera files each clip actually has replace a
    hard-coded default — `camera_smooth.json` does not exist for every run, and asking for one that
    is missing is one of the several things camlab answers 404 to.
    """
    runs = camlab_json("/api/runs")
    for r in runs:
        d = CFG.run_dir / str(r.get("clip_id", ""))
        r["cameras"] = sorted(p.name for p in d.glob("camera_*.json")
                              if p.name != "camera_manual.json") if d.is_dir() else []
    return JSONResponse(runs)


@app.post("/api/prepare")
def prepare(body: dict) -> JSONResponse:
    """Pull camlab's own frame and its own segments, and offer whatever labels already exist.

    The segments are **camlab's**, not ours: a label made here lands one-to-one on the line its
    solver will use, so nothing has to be re-detected or agreed between the two repos.
    """
    clip, frame = str(body["clip"]), int(body["frame"])
    which = str(body.get("which", "camera_smooth.json"))
    method = str(body.get("method", "hough"))
    base = f"/api/run/{clip}"
    lines = camlab_json(f"{base}/lines/{frame}?method={method}&which={which}")
    segments = np.asarray(lines["segments"], float).reshape(-1, 4)
    if len(segments) == 0:
        raise HTTPException(400, f"{clip} f{frame}: camlab detected no segments here")

    try:
        url = f"{CFG.camlab}{base}/frame/{frame}"
        with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310 - configured host
            img = cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
    except urllib.error.URLError as e:
        raise HTTPException(502, f"could not fetch the frame: {e}") from e

    STATE.clear()
    STATE.update(clip=clip, frame=frame, which=which, method=method, image=img, segments=segments)

    # A previous run's labels, if any, so the page opens as a REVIEW rather than a blank form.
    proposed: dict[str, str] = {}
    for name in ("labels_crops.json", "labels.json"):
        p = Path("out/labeller") / f"{clip}_f{frame}" / name
        if p.exists():
            raw = json.loads(p.read_text())
            proposed = {str(k): str(v) for k, v in raw.get("labels", raw).items()}
            break
    return JSONResponse({
        "clip": clip, "frame": frame, "which": which,
        "n_segments": len(segments), "vocabulary": VOCABULARY, "proposed": proposed,
        "lengths": [round(float(np.hypot(s[2] - s[0], s[3] - s[1]))) for s in segments],
    })


@app.get("/api/overview.png")
def overview() -> Response:
    if "image" not in STATE:
        raise HTTPException(400, "nothing prepared yet")
    return png(draw(STATE["image"], STATE["segments"]))


@app.get("/api/tile.png")
def tile(i: int) -> Response:
    if "image" not in STATE:
        raise HTTPException(400, "nothing prepared yet")
    if not (1 <= i <= len(STATE["segments"])):
        raise HTTPException(404, f"segment {i} is outside 1..{len(STATE['segments'])}")
    return png(crop_tile(STATE["image"], STATE["segments"][i - 1], i))


@app.post("/api/rank")
def rank(body: dict) -> JSONResponse:
    """Enumerate label-consistent cameras, refit each with camlab's auto-fit, score on its paint.

    Identical code path to `scripts/write_camlab_anchor.py` — `solve_anchor` is shared, not
    reimplemented, so the page and the CLI cannot drift into disagreeing about what a good anchor
    is. It restores camlab's store before returning; committing is a separate, explicit call.
    """
    if "image" not in STATE:
        raise HTTPException(400, "nothing prepared yet")
    labels = {str(k): str(v) for k, v in dict(body.get("labels", {})).items()}
    opts = SimpleNamespace(
        focal_min=FOCAL_BOUNDS[0], focal_max=FOCAL_BOUNDS[1],
        height_min=HEIGHT_BOUNDS[0], height_max=HEIGHT_BOUNDS[1],
        distance_min=DISTANCE_BOUNDS[0], distance_max=DISTANCE_BOUNDS[1],
        realizable_px=float("inf"), shortlist=int(body.get("shortlist", SHORTLIST)))
    res = solve_anchor(CFG.camlab, STATE["clip"], STATE["frame"], STATE["which"], STATE["method"],
                       labels, CFG.run_dir, opts)
    return JSONResponse({
        "pool": res["pool"], "floor": res["floor"], "twins": res["twins"],
        "baseline": {"median_px": res["baseline"]["median_px"],
                     "n_scored": res["baseline"]["n_scored"]},
        "rows": [{
            "median_px": c["median_px"], "worst_line_px": c["worst_line_px"],
            "n_scored": c["n_scored"], "raw_median_px": c["raw_median_px"],
            "focal_px": c["camera"]["focal_px"], "position": c["camera"]["position"],
            "pairs": [[int(s), int(m)] for s, m in zip(c["segments"], c["markings"], strict=True)],
            "camera": c["camera"],
        } for c in res["scored"]],
    })


@app.post("/api/commit")
def commit(body: dict) -> JSONResponse:
    """Write one chosen camera into camlab's `camera_manual.json` — the store a human's drag uses.

    Keyed by the solve it overlays, so it cannot disturb edits made against a different one.
    """
    if "image" not in STATE:
        raise HTTPException(400, "nothing prepared yet")
    cam = dict(body["camera"])
    run = CFG.run_dir / STATE["clip"]
    write_manual(run, STATE["which"], STATE["frame"], cam)
    r = camlab_json(f"/api/run/{STATE['clip']}/residual/{STATE['frame']}?which={STATE['which']}")
    return JSONResponse({"written": str(run / "camera_manual.json"),
                         "median_px": r["median_px"], "worst_line_px": r["worst_line_px"],
                         "n_scored": r["n_scored"],
                         "viewer": f"{CFG.camlab}/#{STATE['clip']}"})


PAGE = """
<title>label review — anchor from named lines</title>
<style>
:root{--bg:#12141a;--fg:#e6e8ee;--mut:#8b93a7;--acc:#ffd24a;--ok:#5ddc9a;--bad:#ff7b72;
--card:#1b1f28;--line:#2a303c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-sans-serif,system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
padding:10px 16px;display:flex;gap:10px;flex-wrap:wrap;align-items:center;z-index:5}
h1{font-size:15px;margin:0 12px 0 0;font-weight:600}
select,input,button{background:var(--card);color:var(--fg);border:1px solid var(--line);
border-radius:6px;padding:6px 9px;font:inherit}
button{cursor:pointer}button.p{background:var(--acc);color:#141414;border-color:var(--acc);
font-weight:600}button:disabled{opacity:.45;cursor:default}
main{padding:16px;display:grid;gap:16px;grid-template-columns:minmax(0,1fr) 440px}
@media(max-width:1100px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h2{font-size:13px;margin:0 0 8px;color:var(--mut);font-weight:600;
text-transform:uppercase;letter-spacing:.06em}
#ov{width:100%;border-radius:8px;display:block}
.tiles{display:grid;gap:10px;grid-template-columns:repeat(auto-fill,minmax(190px,1fr))}
.tile{background:#0e1116;border:1px solid var(--line);border-radius:8px;padding:8px}
.tile img{width:100%;border-radius:5px;display:block;cursor:zoom-in}
.tile select{width:100%;margin-top:6px}
.tile .n{color:var(--mut);font-size:12px;display:flex;justify-content:space-between}
.tile.na{opacity:.55}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:5px 6px;border-bottom:1px solid var(--line);font-size:13px}
th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase}
td:first-child,th:first-child{text-align:left}
tr.best td{color:var(--ok)}tr.weak td{color:var(--mut)}
.msg{padding:8px 10px;border-radius:6px;margin-top:8px;font-size:13px}
.msg.warn{background:#3a2f14;color:var(--acc)}.msg.err{background:#3a1c1c;color:var(--bad)}
.msg.ok{background:#14301f;color:var(--ok)}
.mut{color:var(--mut)}dialog{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:8px;max-width:96vw}dialog img{max-width:90vw;max-height:85vh;display:block}
</style>
<header>
  <h1>label review</h1>
  <select id="clip"></select>
  <label class="mut">frame
    <input id="frame" type="number" value="0" min="0" style="width:78px">
    <span id="frames" class="mut"></span></label>
  <select id="which" style="width:200px"></select>
  <button id="go" class="p">prepare</button>
  <button id="rank">rank anchors</button>
  <span id="status" class="mut"></span>
</header>
<main>
  <div>
    <div class="card"><h2>the frame, with camlab's own segments</h2><img id="ov" alt=""></div>
    <div class="card" style="margin-top:16px"><h2>one tile per segment — is there paint under the
      dashes?</h2><div class="tiles" id="tiles"></div></div>
  </div>
  <div>
    <div class="card"><h2>anchors, judged by camlab's paint</h2>
      <div id="bar" class="mut"></div>
      <table id="tbl"><thead><tr><th>median</th><th>worst</th><th>n</th><th>raw</th>
      <th>focal</th><th>position</th><th></th></tr></thead><tbody></tbody></table>
      <div id="note"></div>
    </div>
  </div>
</main>
<dialog id="zoom"><img id="zoomimg" alt=""></dialog>
<script>
const $=s=>document.querySelector(s), S={vocab:[],n:0,rows:[]};
const say=(t,k)=>{$('#status').textContent=t; $('#status').className=k||'mut';};
async function jp(u,b){
 const r=await fetch(u,{method:'POST',headers:{'content-type':'application/json'},
   body:JSON.stringify(b)});
 if(!r.ok) throw new Error((await r.json()).detail||r.statusText);
 return r.json();}
let CLIPS=[];
function onclip(){
  const c=CLIPS.find(x=>x.clip_id===$('#clip').value); if(!c) return;
  // The frame box is bounded by the clip, and `which` offers only cameras that exist. Both are
  // things camlab answers 404 to, and an unanswerable form is a worse error message than any text.
  $('#frame').max=Math.max(c.n_frames-1,0);
  if(+$('#frame').value>c.n_frames-1) $('#frame').value=0;
  $('#frames').textContent=`of 0..${c.n_frames-1}`;
  const pref=['camera_smooth.json','camera_fixed.json','camera_healed.json'];
  const best=pref.find(p=>c.cameras.includes(p))||c.cameras[0]||'';
  $('#which').innerHTML=c.cameras.map(n=>
    `<option${n===best?' selected':''}>${n}</option>`).join('')
    || '<option value="">— this clip has no solved camera —</option>';
}
(async()=>{try{CLIPS=await (await fetch('/api/clips')).json();
 $('#clip').innerHTML=CLIPS.map(c=>`<option>${c.clip_id}</option>`).join('');
 $('#clip').onchange=onclip; onclip();}
 catch(e){say('camlab is not answering — is its server up on 8899?','msg err');}})();
$('#go').onclick=async()=>{
  say('asking camlab for the frame and its segments…');
  try{
    const d=await jp('/api/prepare',{clip:$('#clip').value,frame:+$('#frame').value,
      which:$('#which').value});
    S.vocab=d.vocabulary; S.n=d.n_segments;
    const t=Date.now();
    $('#ov').src='/api/overview.png?t='+t;
    $('#tiles').innerHTML=Array.from({length:d.n_segments},(_,k)=>{const i=k+1;
      const sel=S.vocab.map(v=>`<option${d.proposed[i]===v?' selected':''}>${v}</option>`).join('');
      return `<div class="tile" id="t${i}"><div class="n"><b>segment ${i}</b>
        <span>${d.lengths[k]} px</span></div>
        <img src="/api/tile.png?i=${i}&t=${t}" data-i="${i}" alt="segment ${i}">
        <select data-i="${i}">${sel}</select></div>`;}).join('');
    $('#tiles').querySelectorAll('select').forEach(s=>{s.onchange=mark; mark.call(s);});
    $('#tiles').querySelectorAll('img').forEach(im=>im.onclick=()=>{
      $('#zoomimg').src=im.src; $('#zoom').showModal();});
    const known=Object.keys(d.proposed).length;
    say(`${d.n_segments} segments` + (known?` · ${known} labels proposed — review them`:
      ' · no proposal on disk, label them'), known?'msg ok':'mut');
    $('#tbl').querySelector('tbody').innerHTML='';
    $('#note').innerHTML=''; $('#bar').textContent='';
  }catch(e){say(e.message,'msg err');}
};
function mark(){const el=$('#t'+this.dataset.i);
  el.classList.toggle('na',this.value==='not_a_marking');}
$('#zoom').onclick=()=>$('#zoom').close();
$('#rank').onclick=async()=>{
  if(!S.n){say('prepare a frame first','msg err');return;}
  const labels={}; $('#tiles').querySelectorAll('select').forEach(s=>labels[s.dataset.i]=s.value);
  say('enumerating, refitting each with camlab\\'s auto-fit, scoring on its paint…');
  try{
    const d=await jp('/api/rank',{labels}); S.rows=d.rows;
    $('#bar').innerHTML=`pool ${d.pool} · the solve on disk scores <b>${d.baseline.median_px}</b> px
      on ${d.baseline.n_scored} samples — that is the bar`;
    $('#tbl').querySelector('tbody').innerHTML=d.rows.map((r,i)=>{
      const weak=r.n_scored<d.floor||r.median_px==null;
      return `<tr class="${i===0&&!weak?'best':''}${weak?'weak':''}">
        <td>${r.median_px==null?'—':r.median_px.toFixed(2)}</td>
        <td>${r.worst_line_px==null?'—':r.worst_line_px.toFixed(1)}</td>
        <td>${r.n_scored}</td><td>${r.raw_median_px==null?'—':r.raw_median_px.toFixed(1)}</td>
        <td>${r.focal_px.toFixed(0)}</td>
        <td>${r.position.map(v=>v.toFixed(1)).join(', ')}</td>
        <td><button data-r="${i}">use</button></td></tr>`;}).join('');
    $('#tbl').querySelectorAll('button[data-r]').forEach(b=>b.onclick=()=>use(+b.dataset.r));
    $('#note').innerHTML = d.twins ?
      `<div class="msg warn">the top two are <b>half-turn twins</b> — the pitch is exactly
       symmetric about its centre, so the paint scores both the same and never will. Pick the one
       whose stand, hoardings and goal look right in the frame above, or flip it later in camlab's
       viewer.</div>` : (d.rows.length?'':
      `<div class="msg err">no label-consistent camera. A homography needs two segments in each
       of the two parallel families; check the labels, or try another frame.</div>`);
    say(`${d.rows.length} ranked`,'msg ok');
  }catch(e){say(e.message,'msg err');}
};
async function use(i){
  try{const d=await jp('/api/commit',{camera:S.rows[i].camera});
    $('#note').innerHTML=`<div class="msg ok">written to camlab · this frame now scores
      <b>${d.median_px}</b> px median, worst line ${d.worst_line_px}, on ${d.n_scored} samples.
      Open <a href="${d.viewer}" target="_blank" style="color:inherit">camlab's viewer</a> to judge
      it by eye and run the clip.</div>`;
  }catch(e){say(e.message,'msg err');}
}
</script>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(PAGE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8811)
    ap.add_argument("--camlab", default="http://127.0.0.1:8899", help="camlab's server")
    ap.add_argument("--run-dir", default="/home/chubuchnyi/camlab/runs")
    args = ap.parse_args()
    CFG.camlab = args.camlab.rstrip("/")
    CFG.run_dir = Path(args.run_dir)

    import uvicorn
    print(f"label review on http://{args.host}:{args.port}  (camlab: {CFG.camlab})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
