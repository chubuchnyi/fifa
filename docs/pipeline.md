# The pipeline — broadcast clip → novel-view video

One command on the pod runs the whole thing (STATUS §0):

```bash
ANIM_CAMERAS=sideline OUT=out/anim_finish bash scripts/pod_finish_batch.sh
```

Two halves. **Reconstruction** measures the clip into a canonical 3D scene and renders it from
a new camera (the measured core — ADR-0005/0011). **Finishing** turns that render into a
photoreal broadcast frame with a structure-locked generative pass (reconstruct-then-condition,
STATUS §6). Everything below is per-camera; the finishing half eats exactly one camera's frames.

```mermaid
flowchart LR
    clip["broadcast clip\n(floodlit night, 60 frames)"] --> A
    subgraph A["A · reconstruction (measured)"]
        direction LR
        recon["perception\n→ scene.json"] --> export["anim_export\n→ *.npz + manifest"] --> render["Cycles beauty render\n1280×720"]
    end
    A --> B
    subgraph B["B · finishing (structure-locked generative)"]
        direction LR
        grade["night grade"] --> inject["kit-colour\ninject"] --> v2v["Wan-VACE\nrgb control"] --> up["SeedVR2\n720p"] --> pin["hue pins\nA + B"]
    end
    B --> final["FINAL\n*_pinned2.mp4"]
```

## A · Reconstruction — clip → scene → beauty render

`scripts/pod_make_video.sh` (wrapped as step 1 of the batch; `REUSE_SCENE=1` skips perception
when `scene.json` already exists).

```mermaid
flowchart TD
    clip["clip frames (decode, FRAMES=60)"] --> det["RF-DETR — player/ball detect"]
    det --> trk["ByteTrack — identity tracks"]
    trk --> pose["SMPLest-X-H — per-player SMPL-X pose"]
    clip --> ball["WASB — ball 2D + lift"]
    clip --> calib["PnLCalib — camera solve\n(--real-calib, ~10 kp/frame)"]
    pose & ball & calib --> asm["assemble + coherence + stitch\n→ canonical scene.json (ADR-0005)"]

    asm --> exp["anim_export.py"]
    exp --> subj["anim_subject_*.npz\nper-frame SMPL-X verts, kit-boost colours,\nmeasured per-vertex body texture"]
    exp --> pitchnpz["pitch.npz — grass PBR + lines + goals"]
    exp --> stad["stadium.npz — bowl + crowd QUILT\n(tinted mosaic, non-repeating)"]
    exp --> boards["boards.npz — LED ad-board ring\n+ dark walkway band (geometric prior)"]
    exp --> light["lighting.npz — light-from-clip\n(floodlit-night: 4 soft suns, cool rgb)"]
    exp --> cams["cameras.npz — virtual operator\n(sideline / broadcast / top / goal)"]
    subj & pitchnpz & stad & boards & light & cams --> man["manifest.json (schema v1)\nanim_contract validates every artifact"]

    man --> cyc["blender_animate.py — Cycles GPU\nOptiX denoise, persistent data, 32 samples"]
    cyc --> beauty["frames/&lt;cam&gt;/frame_*.png + video/&lt;cam&gt;.mp4\n(the beauty render)"]
```

Contract: the exporter and renderer only talk through `manifest.json`
(`src/pitch3d/adapters/blender/anim_contract.py`) — an unknown or key-incomplete artifact kills
the run at export time, not mid-render. New artifacts must be registered in `REQUIRED_KEYS`.

## B · Finishing — beauty render → photoreal night broadcast

Steps 2–8 of `scripts/pod_finish_batch.sh`. The control-signal insight (A/B 2026-07-03): the
generative pass copies **structure** from the control frames but re-invents anything it can't
read — so every step before v2v exists to make identity (kit colour, boards, night tone)
unambiguous in the control, and every step after undoes the drift v2v/upscale introduce.

```mermaid
flowchart TD
    beauty["beauty frames (A)"] --> grade["2 · night grade — grade3\neq brightness −0.28 · contrast 1.12 · gamma 0.75\n+ cool colorbalance"]
    beauty --> mask["3 · team-mask AOV pass\nblender_animate --team-mask 1\n832×480, 1 sample (R = team A, G = team B)"]
    grade --> inj["4 · control_kit_inject.py\npush H (circular) + S (up only) toward kit colour\ninside eroded masks; V untouched\nα 0.8 · erode 3 · A = 65°/0.85 · B = 185°/0.95"]
    mask --> inj
    inj --> v2v["5 · Wan-VACE v2v — rgb control\n1280×720 · flow-shift 5.0 · conditioning 1.0\n(480p repaints distant players — 2–3 latent px)"]
    v2v --> seed["6 · SeedVR2 3B — 720p upscale"]
    seed --> pinb["7 · hue-pin team B (channel g)\none clip-wide hue delta, target auto-measured\nfrom THIS run's beauty render"]
    pinb --> pina["8 · hue-pin team A (channel r)\nband 5–80° · sat-min 0.35 keeps faces out"]
    pina --> final["FINAL — *_pinned2.mp4"]
```

Why each step survives (all eye-judged A/B on the target clip, log in STATUS §6):

| # | Step | Reason it exists | Kill switch |
|---|------|------------------|-------------|
| 2 | grade3 night grade | render is day-bright; the clip is floodlit night | edit `GRADE` in the script |
| 3 | team-mask AOV | one cheap pass feeds steps 4, 7, 8 | — |
| 4 | kit inject | grade3 erases kit colour in clusters → Wan hallucinates kits | `KIT_INJECT=0` |
| 5 | v2v rgb control | the photoreal pass; rgb beats depth/pose controls on this scene | `V2V_WIDTH/HEIGHT/FLOW`, `CS` |
| 6 | SeedVR2 | Wan output is soft; upscale restores broadcast crispness | — |
| 7 | pin B | v2v+upscale drift azure → blue-violet | `TARGET_HUE` override |
| 8 | pin A | same drift pulls yellow → orange | `PIN_A=0` |

## Where things live

| Piece | Path |
|-------|------|
| one-command batch | `scripts/pod_finish_batch.sh` |
| recon + export + render | `scripts/pod_make_video.sh` → `scripts/pod_real_e2e.sh`, `src/pitch3d/app/anim_export.py`, `scripts/blender_animate.py` |
| scene geometry (bowl, boards) | `src/pitch3d/core/scene/stadium.py` |
| kit inject / hue pin | `scripts/control_kit_inject.py`, `scripts/hue_pin.py` |
| v2v / upscale wrappers | `scripts/pod_v2v.sh`, `scripts/pod_seedvr2.sh` |
| export↔render contract | `src/pitch3d/adapters/blender/anim_contract.py` |
| durable run log + verdicts | `docs/STATUS.md` (§0 TL;DR, §6 A/B log) |

Every measured estimator ships a manual override (auto → CLI flag/env → validated default) —
the `TEAM_*_HSV`, `TARGET_HUE`, `--board-height 0` knobs above are instances of that rule.
