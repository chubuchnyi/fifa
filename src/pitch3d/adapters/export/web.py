"""Web viewer export — a dependency-free three.js bundle of the resolved scene (FR-27, M3-7).

Writes a self-contained viewer: ``index.html`` (opens in any browser, no build step and no server —
the scene data is inlined) plus ``scene.json`` (the same payload as a standalone artifact for
programmatic / LLM consumption). The viewer shows the **resolved** subject roots and the ball as
animated, team-coloured markers on a metric pitch, in glTF's **Y-up** convention: it reuses the
glTF path's Z-up→Y-up conversion (:func:`pitch3d.adapters.export.gltf.build_gltf_scene`) so the web
view opens *without scale/coord loss* (AC-6). It is the read-side, no-GPU/no-Blender complement to
the in-process :func:`pitch3d.adapters.render.radar.render_radar` — a 3D "where is everyone" the
operator (and the LLM observation loop, ADR-0008) can open anywhere.

Honest scope (R-6): **markers, not SMPL-X meshes**; the full textured-mesh web viewer rides the
gated ``.glb`` path (``pitch3d[export]``) in a later increment. The viewer loads three.js from a
pinned CDN at view time (the only network dependency); the scene data itself is embedded.
"""

from __future__ import annotations

import json
from pathlib import Path

from ...core.scene.scene import Scene
from ...core.scene.subject import Role

#: Marker colours as glTF/three.js-native ``[0, 1]`` RGB floats.
_GOLD = (1.0, 0.78, 0.16)        # the ball
_REFEREE = (1.0, 0.85, 0.10)     # officials (carry no team colour)
#: Deterministic two-team fallback when a Team carries no measured ``color_rgb``.
_TEAM_FALLBACK = ((0.20, 0.45, 0.95), (0.95, 0.30, 0.25))
_DEFAULT = (0.80, 0.80, 0.80)


def _subject_colors(scene: Scene) -> dict[int, tuple[float, float, float]]:
    """Map each subject ``track_id`` → an ``[0, 1]`` RGB marker colour.

    Prefers the team's measured ``color_rgb`` (the tracker's HSV classifier, FR-6); falls back to a
    deterministic per-team colour when the team carries none, and tints officials apart.
    """
    teams = {t.id: t for t in scene.teams}
    order = {t.id: i for i, t in enumerate(scene.teams)}  # stable fallback index per team
    colors: dict[int, tuple[float, float, float]] = {}
    for subj in scene.subjects:
        if subj.role is Role.REFEREE:
            colors[subj.track_id] = _REFEREE
            continue
        team = teams.get(subj.team_id) if subj.team_id is not None else None
        if team is not None and team.color_rgb is not None:
            c = team.color_rgb
            colors[subj.track_id] = (float(c[0]), float(c[1]), float(c[2]))
        elif subj.team_id is not None:
            colors[subj.track_id] = _TEAM_FALLBACK[order.get(subj.team_id, 0) % 2]
        else:
            colors[subj.track_id] = _DEFAULT
    return colors


def build_viewer_payload(scene: Scene, *, fps: float = 25.0) -> dict:
    """Resolve ``scene`` into the viewer's JSON payload (Y-up tracks + pitch dims + colours).

    Reuses :func:`build_gltf_scene` so the per-node times/translations are byte-for-byte the same
    Y-up samples the glTF export emits — the web view inherits the exact axis/scale conversion, so
    it can never drift from the interchange export (AC-6).
    """
    from .gltf import build_gltf_scene  # lazy: gltf.py imports this module back (THREEJS branch)

    gscene = build_gltf_scene(scene, fps=fps)
    colors = _subject_colors(scene)
    dims = scene.field.dimensions
    nodes: list[dict] = []
    for node in gscene.nodes:
        if node.name == "ball":
            kind, color = "ball", _GOLD
        else:
            kind = "subject"
            color = colors.get(int(node.name.removeprefix("subject_")), _DEFAULT)
        nodes.append(
            {
                "name": node.name,
                "kind": kind,
                "color": [round(float(c), 4) for c in color],
                "times": [round(float(t), 4) for t in node.times.tolist()],
                "positions": [[round(float(v), 4) for v in p] for p in node.translations.tolist()],
            }
        )
    return {
        "generator": "pitch3d",
        "up": "Y",
        "fps": float(fps),
        "pitch": {"length": float(dims.length), "width": float(dims.width)},
        "nodes": nodes,
    }


def write_web_bundle(scene: Scene, out_dir: Path, *, fps: float = 25.0) -> list[str]:
    """Write ``index.html`` + ``scene.json`` into ``out_dir``; return ``[html, json]`` paths.

    The HTML embeds the payload (so it opens straight off the filesystem, no server) and
    ``scene.json`` carries the same data for tools/agents that want the numbers without the page.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_viewer_payload(scene, fps=fps)
    json_path = out_dir / "scene.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html = _VIEWER_HTML.replace("__SCENE_JSON__", json.dumps(payload))
    html_path = out_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")
    return [str(html_path), str(json_path)]


# A self-contained three.js viewer. ``__SCENE_JSON__`` is replaced with the embedded payload so the
# file opens with no server; three.js itself is pinned from a CDN (the only view-time dependency).
_VIEWER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>pitch3d - web viewer</title>
<style>
  html,body{margin:0;height:100%;background:#0c0e0c;color:#cfeacf;
    font:13px system-ui,sans-serif;overflow:hidden}
  #hud{position:fixed;top:8px;left:8px;z-index:10;background:rgba(0,0,0,.55);
    padding:6px 10px;border-radius:6px}
  #hud b{color:#7fd6a0}
</style>
</head>
<body>
<div id="hud"><b>pitch3d</b> viewer &middot; <span id="info">loading</span>
  &middot; drag=orbit &middot; wheel=zoom &middot; space=pause</div>
<script type="importmap">
{ "imports": { "three": "https://unpkg.com/three@0.160.0/build/three.module.js" } }
</script>
<script type="module">
import * as THREE from "three";
const SCENE = __SCENE_JSON__;

const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0c0e0c);
const camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.1, 5000);

const L = SCENE.pitch.length, W = SCENE.pitch.width, L2 = L/2, W2 = W/2;

scene.add(new THREE.HemisphereLight(0xffffff, 0x224422, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.0);
sun.position.set(L, L, W); scene.add(sun);

// pitch: world X = length, world Z = width (the Y-up frame), Y = up.
const pitch = new THREE.Mesh(
  new THREE.PlaneGeometry(L, W),
  new THREE.MeshStandardMaterial({color:0x1f6b35}));
pitch.rotation.x = -Math.PI/2; scene.add(pitch);

const lineMat = new THREE.LineBasicMaterial({color:0xdfe8df});
const segs = [];
const edge=(x0,z0,x1,z1)=>segs.push(x0,0.02,z0, x1,0.02,z1);
edge(-L2,-W2, L2,-W2); edge(L2,-W2, L2,W2); edge(L2,W2, -L2,W2); edge(-L2,W2, -L2,-W2);
edge(0,-W2, 0,W2);  // halfway line
const lg = new THREE.BufferGeometry();
lg.setAttribute("position", new THREE.Float32BufferAttribute(segs,3));
scene.add(new THREE.LineSegments(lg, lineMat));
const circ=[], R=Math.min(9.15, W2*0.5);
for(let i=0;i<=64;i++){const a=i/64*Math.PI*2; circ.push(Math.cos(a)*R,0.02,Math.sin(a)*R);}
const cg=new THREE.BufferGeometry();
cg.setAttribute("position", new THREE.Float32BufferAttribute(circ,3));
scene.add(new THREE.Line(cg, lineMat));

const markers = SCENE.nodes.map(n=>{
  const r = n.kind==="ball" ? 0.18 : 0.5;
  const m = new THREE.Mesh(new THREE.SphereGeometry(r,16,12),
    new THREE.MeshStandardMaterial({color:new THREE.Color(n.color[0],n.color[1],n.color[2])}));
  scene.add(m); return {node:n, mesh:m};
});

let tMin=Infinity, tMax=-Infinity;
for(const n of SCENE.nodes){ if(n.times.length){
  tMin=Math.min(tMin,n.times[0]); tMax=Math.max(tMax,n.times[n.times.length-1]); } }
if(!isFinite(tMin)){ tMin=0; tMax=0; }
document.getElementById("info").textContent =
  SCENE.nodes.length+" nodes · "+L.toFixed(0)+"×"+W.toFixed(0)+" m · "
  +(tMax-tMin).toFixed(1)+" s";

function sample(n, t){
  const ts=n.times, ps=n.positions;
  if(ts.length===0) return null;
  if(t<=ts[0]) return ps[0];
  if(t>=ts[ts.length-1]) return ps[ts.length-1];
  let i=1; while(i<ts.length && ts[i]<t) i++;
  const a=ts[i-1], b=ts[i], f=(t-a)/((b-a)||1), pa=ps[i-1], pb=ps[i];
  return [pa[0]+(pb[0]-pa[0])*f, pa[1]+(pb[1]-pa[1])*f, pa[2]+(pb[2]-pa[2])*f];
}

let yaw=0.6, tilt=0.9, dist=Math.max(L,W)*0.9, drag=false, px=0, py=0;
addEventListener("mousedown",e=>{drag=true;px=e.clientX;py=e.clientY;});
addEventListener("mouseup",()=>drag=false);
addEventListener("mousemove",e=>{ if(!drag)return;
  yaw-=(e.clientX-px)*0.005;
  tilt=Math.max(0.05,Math.min(1.5,tilt-(e.clientY-py)*0.005));
  px=e.clientX; py=e.clientY; });
addEventListener("wheel",e=>{ dist=Math.max(5,Math.min(2000,dist*(1+Math.sign(e.deltaY)*0.1))); },
  {passive:true});
addEventListener("resize",()=>{ camera.aspect=innerWidth/innerHeight;
  camera.updateProjectionMatrix(); renderer.setSize(innerWidth,innerHeight); });
let paused=false;
addEventListener("keydown",e=>{ if(e.code==="Space"){ paused=!paused; e.preventDefault(); } });

const clock = new THREE.Clock();
let playT = tMin;
function frame(){
  requestAnimationFrame(frame);
  const dt = clock.getDelta();
  if(!paused){ playT += dt; if(playT>tMax) playT=tMin; }
  for(const {node,mesh} of markers){
    const p = sample(node, playT);
    if(p){ mesh.visible=true; mesh.position.set(p[0],p[1],p[2]); } else mesh.visible=false;
  }
  const cy=Math.sin(tilt)*dist, cr=Math.cos(tilt)*dist;
  camera.position.set(Math.cos(yaw)*cr, cy, Math.sin(yaw)*cr);
  camera.lookAt(0,0,0);
  renderer.render(scene,camera);
}
frame();
</script>
</body>
</html>
"""


__all__ = ["build_viewer_payload", "write_web_bundle"]
