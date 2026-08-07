// The whole-pitch 3D view, as a mountable component.
//
// It answers a different question from index.html's 3D panel: that one is a pelvis-centred editor
// for ONE subject's joints, this one is every player on the measured pitch at one frame, free to
// orbit. Judging pose and physics work needs the second — scrubbing frames beats paying ~30 min
// and GPU money for a full render just to see whether bodies move like bodies.
//
// Extracted from world.html so the app's right panel can host the same view rather than carry a
// second copy of it. The host owns all chrome (toolbar, scrubber, status text); this module owns
// the scene, the camera and the drawing, and reports back through callbacks.
//
//   const view = createWorldView({ mount, THREE, OrbitControls, onStatus, onPick });
//   await view.init();  await view.show(0);
//
// Every endpoint it calls is same-origin and absolute, so no base-URL plumbing.

// Same topology as the annotation UI (SMPL-X body, 22 joints).
const BONES = [
  [0, 1], [0, 2], [0, 3], [1, 4], [4, 7], [7, 10], [2, 5], [5, 8], [8, 11],
  [3, 6], [6, 9], [9, 12], [12, 15], [9, 13], [13, 16], [16, 18], [18, 20],
  [9, 14], [14, 17], [17, 19], [19, 21],
];
const NO_TEAM = 0xbdbdbd;
// The overlay arm gets ONE colour, not team colours: the question it answers is "where do these
// two runs disagree", and two palettes on top of each other answer nothing.
const ARM_B_COLOUR = 0xff3ea5;
const HIGHLIGHT = 0xa8d4ff;              // matches index.html's accentBright
const DEFAULT_VIEW = { pos: [0, -95, 55], target: [0, 0, 0] };

const rgbToHex = (c) => (Math.round(c[0] * 255) << 16) | (Math.round(c[1] * 255) << 8)
                        | Math.round(c[2] * 255);

export function createWorldView(cfg) {
  const { mount, THREE, OrbitControls } = cfg;
  const onStatus = cfg.onStatus || (() => {});
  const onStatus2 = cfg.onStatus2 || (() => {});
  const onError = cfg.onError || (() => {});
  const onFrame = cfg.onFrame || (() => {});
  const onPick = cfg.onPick || null;

  let TEAM_COLOUR = {};
  const opts = { ids: false, trails: false, prov: true, overlayB: false, highlight: null };

  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#14161a");
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 600);
  camera.up.set(0, 0, 1);                     // world is Z-up, like the rest of the pipeline
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(devicePixelRatio);
  mount.appendChild(renderer.domElement);

  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.enableDamping = true;
  orbit.enablePan = true;
  // Pan along the PITCH, not the screen plane: on a Z-up world screen-space panning drifts the
  // camera off the ground the moment you are looking down at it, which is most of the time here.
  orbit.screenSpacePanning = false;
  orbit.panSpeed = 1.2;
  // Middle-drag pans too — right-drag alone is easy to miss, and the eye review asked for this.
  orbit.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.PAN,
  };

  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const key = new THREE.DirectionalLight(0xffffff, 0.6);
  key.position.set(40, -60, 80);
  scene.add(key);

  const pitchGroup = new THREE.Group(); scene.add(pitchGroup);
  const poseGroup = new THREE.Group(); scene.add(poseGroup);
  const poseGroupB = new THREE.Group(); scene.add(poseGroupB);
  const trailGroup = new THREE.Group(); scene.add(trailGroup);
  const labelGroup = new THREE.Group(); scene.add(labelGroup);

  const jointGeo = new THREE.SphereGeometry(0.055, 8, 8);
  const jointGeoB = new THREE.SphereGeometry(0.04, 6, 6);
  const jointGeoHi = new THREE.SphereGeometry(0.085, 10, 10);

  // Group.clear() detaches children but frees nothing. This view redraws every group on every
  // frame, so without disposal a play-through leaks one geometry + two materials per subject per
  // frame — invisible for a few frames, minutes of scrubbing later it is not.
  function clearGroup(g) {
    g.traverse((o) => {
      if (o === g) return;
      if (o.geometry && o.geometry !== jointGeo && o.geometry !== jointGeoB
          && o.geometry !== jointGeoHi) o.geometry.dispose();
      const mats = Array.isArray(o.material) ? o.material : (o.material ? [o.material] : []);
      for (const m of mats) {
        if (m.map) m.map.dispose();
        m.dispose();
      }
    });
    g.clear();
  }

  function polyline(points, colour, z = 0) {
    const pts = points.map((p) => new THREE.Vector3(p[0], p[1], p.length > 2 ? p[2] : z));
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    return new THREE.Line(geo, new THREE.LineBasicMaterial({ color: colour }));
  }

  async function buildPitch() {
    const g = await fetch("/api/world/geometry").then((r) => r.json());
    // The turf: a plain plane so the skeletons read as standing on something. Slightly below 0 so
    // the markings, which ARE at z=0, do not z-fight with it.
    const turf = new THREE.Mesh(
      new THREE.PlaneGeometry(g.length + 12, g.width + 12),
      new THREE.MeshLambertMaterial({ color: 0x2f5d33 }),
    );
    turf.position.z = -0.01;
    pitchGroup.add(turf);
    for (const line of g.markings) {
      if (line.length < 2) {                   // a one-point polyline is a spot (centre/penalty)
        const dot = new THREE.Mesh(new THREE.CircleGeometry(0.15, 12),
                                   new THREE.MeshBasicMaterial({ color: 0xffffff }));
        dot.position.set(line[0][0], line[0][1], 0.005);
        pitchGroup.add(dot);
        continue;
      }
      pitchGroup.add(polyline(line, 0xffffff, 0.004));
    }
    for (const up of g.uprights) pitchGroup.add(polyline(up, 0x9fd8ff));
    return g;
  }

  function makeLabel(text, at, colour, dz = 0.55) {
    const c = document.createElement("canvas");
    c.width = 128; c.height = 64;
    const ctx = c.getContext("2d");
    ctx.font = "bold 44px system-ui, sans-serif";
    ctx.fillStyle = "#" + colour.toString(16).padStart(6, "0");
    ctx.textAlign = "center";
    ctx.fillText(String(text), 64, 46);
    const spr = new THREE.Sprite(new THREE.SpriteMaterial({
      map: new THREE.CanvasTexture(c), depthTest: false }));
    spr.position.set(at[0], at[1], at[2] + dz);
    spr.scale.set(1.6, 0.8, 1);
    return spr;
  }

  // One Group per subject, tagged with its track id, so a raycast hit can be traced back to a
  // player rather than to an anonymous triangle.
  function subjectGroup(s, colour, real, geo) {
    const g = new THREE.Group();
    g.userData.track_id = s.track_id;
    g.userData.pelvis = s.joints[0];
    const mat = real
      ? new THREE.MeshLambertMaterial({ color: colour })
      : new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.22 });
    const lineMat = new THREE.LineBasicMaterial({ color: colour,
      transparent: !real, opacity: real ? 1 : 0.25 });
    for (const j of s.joints) {
      const m = new THREE.Mesh(geo, mat);
      m.position.set(j[0], j[1], j[2]);
      g.add(m);
    }
    const segs = [];
    for (const [a, b] of BONES) {
      if (!s.joints[a] || !s.joints[b]) continue;
      segs.push(new THREE.Vector3(...s.joints[a]), new THREE.Vector3(...s.joints[b]));
    }
    g.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(segs), lineMat));
    return g;
  }

  function drawSkeletons(subjects) {
    clearGroup(poseGroup); clearGroup(labelGroup);
    let nImputed = 0;
    for (const s of subjects) {
      const picked = opts.highlight != null && s.track_id === opts.highlight;
      const colour = picked ? HIGHLIGHT : (TEAM_COLOUR[s.team] ?? NO_TEAM);
      // "measured" is the only provenance that was actually seen; the rest were filled in when the
      // crop was degenerate or the track had a gap. Drawn ghosted so an invented body cannot pass
      // for a reconstructed one — the scene has always recorded this, the view just hid it.
      const real = !opts.prov || s.provenance === "measured" || s.provenance == null;
      if (!real) nImputed += 1;
      poseGroup.add(subjectGroup(s, colour, real, picked ? jointGeoHi : jointGeo));
      // The selected player is always labelled, whatever the ids toggle says — otherwise the panel
      // showing "which one is he" needs a second toggle flipped before it can answer.
      if (opts.ids || picked) {
        labelGroup.add(makeLabel((real ? "" : "~") + s.track_id,
                                 s.joints[15] || s.joints[0], colour));
      }
    }
    return nImputed;
  }

  // Arm B is drawn as bones only, thinner and in one colour, so the primary run stays readable
  // underneath it. Where the two agree the magenta sits inside the coloured skeleton and reads as
  // trim; where they disagree it separates, and that separation is the whole point.
  function drawArmB(subjects) {
    clearGroup(poseGroupB);
    for (const s of subjects) {
      poseGroupB.add(subjectGroup(s, ARM_B_COLOUR, true, jointGeoB));
      // Arm B needs its own labels, offset above arm A's, or a magenta skeleton standing on its
      // own — exactly the interesting case, a track one run has and the other does not — is
      // unidentifiable.
      if (opts.ids) {
        labelGroup.add(makeLabel(s.track_id, s.joints[15] || s.joints[0], ARM_B_COLOUR, 1.05));
      }
    }
  }

  // ── frame state ───────────────────────────────────────────────────────────
  let nFrames = 0, frame = 0, inflight = false, disposed = false;
  const cache = new Map();          // frame → subjects, so scrubbing back is instant

  async function subjectsAt(n, arm = "a") {
    const k = arm === "a" ? n : `b${n}`;
    if (cache.has(k)) return cache.get(k);
    const r = await fetch(`/api/world/${n}/skeletons?arm=${arm}`);
    if (!r.ok) throw new Error(`frame ${n} (${arm}): HTTP ${r.status}`);
    const d = await r.json();
    if (arm === "a" && d.teams) {
      for (const [id, rgb] of Object.entries(d.teams)) TEAM_COLOUR[id] = rgbToHex(rgb);
    }
    cache.set(k, d.subjects);
    return d.subjects;
  }

  // Trails answer "did this player walk there or teleport?" without leaving the frame you are on.
  async function drawTrails() {
    clearGroup(trailGroup);
    const step = Math.max(1, Math.round(nFrames / 60));
    const byTrack = new Map();
    for (let n = 0; n <= frame; n += step) {
      let subs;
      try { subs = await subjectsAt(n); } catch { continue; }
      for (const s of subs) {
        if (!byTrack.has(s.track_id)) byTrack.set(s.track_id, { team: s.team, pts: [] });
        byTrack.get(s.track_id).pts.push(s.joints[0]);        // pelvis
      }
    }
    for (const [tid, t] of byTrack) {
      if (t.pts.length < 2) continue;
      const col = opts.highlight != null && tid === opts.highlight
        ? HIGHLIGHT : (TEAM_COLOUR[t.team] ?? NO_TEAM);
      trailGroup.add(polyline(t.pts, col));
    }
  }

  async function show(n) {
    if (inflight || disposed) return;
    inflight = true;
    try {
      frame = Math.max(0, Math.min(Math.max(0, nFrames - 1), n));
      const subs = await subjectsAt(frame);
      const nImp = drawSkeletons(subs);
      let nb = 0;
      if (opts.overlayB) {
        const b = await subjectsAt(frame, "b");
        nb = b.length;
        drawArmB(b);
        const only = b.filter((x) => !subs.some((y) => y.track_id === x.track_id))
                      .map((x) => x.track_id);
        onStatus2(only.length ? `only in B: ${only.join(", ")}` : "");
        if (!nb) onError("overlay: no second scene — set POSEANNOT_SCENE_JSON_B and restart");
      } else {
        clearGroup(poseGroupB);
        onStatus2("");
      }
      if (opts.trails) await drawTrails(); else clearGroup(trailGroup);
      onStatus(`frame ${frame} / ${Math.max(0, nFrames - 1)} · ${subs.length} subject(s)`
               + (nImp ? ` · ${nImp} imputed (ghosted, "~")` : "")
               + (opts.overlayB ? ` · overlay B ${nb}` : ""));
      onFrame(frame);
      onError("");
    } catch (e) {
      onError(String(e.message || e));
    } finally {
      inflight = false;
    }
  }

  // ── camera ────────────────────────────────────────────────────────────────
  function resize() {
    const w = mount.clientWidth, h = mount.clientHeight;
    if (!w || !h) return;                 // panel not laid out yet — the observer re-fires
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  function resetView() {
    camera.position.set(...DEFAULT_VIEW.pos);
    orbit.target.set(...DEFAULT_VIEW.target);
    orbit.update();
  }

  // Re-centre without changing the viewing angle: the operator has usually just spent a few
  // seconds finding an angle, and snapping it back to a canned one to move the pivot is hostile.
  function centreOn(p) {
    const shift = new THREE.Vector3(p[0], p[1], p[2]).sub(orbit.target);
    orbit.target.add(shift);
    camera.position.add(shift);
    orbit.update();
  }

  function frameAll() {
    const box = new THREE.Box3().setFromObject(poseGroup);
    if (box.isEmpty()) return resetView();
    const c = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length() || 20;
    const dir = camera.position.clone().sub(orbit.target).normalize();
    orbit.target.copy(c);
    camera.position.copy(c).add(dir.multiplyScalar(Math.max(25, size * 1.4)));
    orbit.update();
  }

  function focusTrack(tid) {
    for (const g of poseGroup.children) {
      if (g.userData.track_id === tid && g.userData.pelvis) {
        centreOn(g.userData.pelvis);
        return true;
      }
    }
    return false;
  }

  // Keyboard panning. Arrows are frame-stepping in both hosts, so WASD moves the world; the vector
  // is built from the camera's own axes and flattened to the pitch, so "forward" means what it
  // looks like from wherever you are standing.
  const held = new Set();
  function panStep() {
    if (!held.size) return;
    const speed = held.has("shift") ? 2.0 : 0.6;   // metres per frame; halved 2026-08-07 on use
    const fwd = new THREE.Vector3();
    camera.getWorldDirection(fwd);
    fwd.z = 0;
    if (fwd.lengthSq() < 1e-6) fwd.set(0, 1, 0);
    fwd.normalize();
    const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 0, 1)).normalize();
    const d = new THREE.Vector3();
    if (held.has("w")) d.add(fwd);
    if (held.has("s")) d.sub(fwd);
    if (held.has("d")) d.add(right);
    if (held.has("a")) d.sub(right);
    if (held.has("e")) d.z += 1;
    if (held.has("q")) d.z -= 1;
    if (d.lengthSq() === 0) return;
    d.normalize().multiplyScalar(speed);
    camera.position.add(d);
    orbit.target.add(d);
  }

  // Bound to the MOUNT, not to window: inside index.html, `a` already toggles show-all and `d`
  // toggles diagnostics. Consuming the key here and stopping propagation means the pan keys work
  // when the 3D panel has focus and the app's own shortcuts work everywhere else.
  function onKeyDown(e) {
    const k = e.key.toLowerCase();
    if (k === "shift") { held.add("shift"); return; }
    if (k.length === 1 && "wasdqe".includes(k)) {
      held.add(k); e.stopPropagation(); e.preventDefault(); return;
    }
    if (k === "f") { frameAll(); e.stopPropagation(); }
  }
  function onKeyUp(e) { held.delete(e.key.toLowerCase()); }
  const clearHeld = () => held.clear();     // never leave a key stuck down on tab-away

  mount.tabIndex = mount.tabIndex >= 0 ? mount.tabIndex : 0;
  mount.style.outline = "none";
  mount.addEventListener("keydown", onKeyDown);
  mount.addEventListener("keyup", onKeyUp);
  mount.addEventListener("blur", clearHeld);
  addEventListener("blur", clearHeld);

  // Click a skeleton to orbit around HIM. Rotating about the pitch centre is useless when the
  // thing under examination is one player in the box.
  const ray = new THREE.Raycaster();
  function onPointerDown(ev) {
    mount.focus({ preventScroll: true });
    if (ev.button !== 0) return;
    const r = renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((ev.clientX - r.left) / r.width) * 2 - 1,
      -((ev.clientY - r.top) / r.height) * 2 + 1,
    );
    ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObjects([poseGroup, poseGroupB], true)[0];
    if (!hit) return;
    let o = hit.object, tid;
    while (o && tid === undefined) { tid = o.userData?.track_id; o = o.parent; }
    centreOn(hit.point.toArray());
    if (tid !== undefined && onPick) onPick(tid);
  }
  renderer.domElement.addEventListener("pointerdown", onPointerDown);

  const ro = new ResizeObserver(resize);
  ro.observe(mount);

  let raf = 0;
  function tick() {
    if (disposed) return;
    raf = requestAnimationFrame(tick);
    panStep();
    orbit.update();
    renderer.render(scene, camera);
  }

  return {
    async init() {
      resize();
      resetView();
      await buildPitch();
      const meta = await fetch("/api/scene").then((r) => r.json());
      nFrames = meta.n_frames || 0;
      if (!raf) tick();
      return nFrames;
    },
    show,
    resize,
    resetView,
    frameAll,
    focusTrack,
    setOpts(o) { Object.assign(opts, o); },
    get opts() { return { ...opts }; },
    get frame() { return frame; },
    get nFrames() { return nFrames; },
    get busy() { return inflight; },
    // After an edit the cached skeletons are stale; the host says so rather than this guessing.
    invalidate() { cache.clear(); },
    dispose() {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      removeEventListener("blur", clearHeld);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      for (const g of [pitchGroup, poseGroup, poseGroupB, trailGroup, labelGroup]) clearGroup(g);
      jointGeo.dispose(); jointGeoB.dispose(); jointGeoHi.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode) renderer.domElement.remove();
    },
  };
}
