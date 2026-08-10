# Four camera parameters are missing from the render, and two of them we measured first

**2026-08-10**, from the user's observation: *"для камеры не указаны углы поворота вокруг осей и
FoV y — ещё 4 параметра не хватает помимо x, y, z, FoV x, focal."*

Correct, and the count is exactly four. What the observation does not yet say is the sharper part:
**two of the four are measured upstream and thrown away**, and the camera that renders the
deliverable is not the solved camera at all.

---

## 1. There are four camera representations and each drops something

| | position | rotation | focal | principal point | distortion |
|---|---|---|---|---|---|
| PnLCalib raw `cam_params` | 3 | 3×3 matrix | **`x_focal_length`, `y_focal_length` — separate** | **`principal_point`** | — |
| `fit_rigid_camera.py` model | 3 (`centre`) | 3/frame (`rvecs`) | **one scalar** | **assumed `W/2, H/2`** | none |
| `calib/*.npz` on disk | `centre` (3,) | `rvecs` (60, 3) | `focal` — scalar `()` | **absent** | absent |
| `core.scene.camera.CameraTrack` (scene.json) | 3/frame | quat/frame — **all 3 DOF** | `fx`, `fy` fields, **written equal** | `cx`, `cy` fields, **written as centre** | field exists, `None` |
| `core.scene.cameras.CameraTrack` (render) | 3, fixed mount | **`look_at` only** | `fov_x_deg` | **no field** | **no field** |

The scene contract is fine — `rotation_quat` carries all three rotation DOF and there are real
`fx/fy/cx/cy` fields. The loss happens on both sides of it.

## 2. The four that are missing from the render camera

`core/scene/cameras.py:45`:

```python
def rotation(self, row: int) -> np.ndarray:
    fwd = self.look_at[row] - self.position
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))   # <- always perpendicular to world up
    up = np.cross(right, fwd)
```

The right vector is *constructed* perpendicular to world up, so the camera's horizon is always
level: **roll is identically zero and there is no field that could hold it.** Yaw and pitch are
present, encoded as `look_at` rather than as angles.

`blender_animate.py:529` sets `cam_data.sensor_fit = "HORIZONTAL"`, so Blender derives the vertical
angle from the render aspect ratio: **`fov_y` is not a parameter**, only a consequence. And there
is no `cx`/`cy` at all, so the virtual camera's principal point is the image centre by
construction.

So against a full pinhole (6 extrinsic + 4 intrinsic), the render camera has **6**: three position,
two aim, one fov. Missing: **roll · fov_y (i.e. `fx ≠ fy`) · cx · cy.** Distortion is a fifth if
counted.

## 3. The render camera is not the solved camera

`anim_export.py:577` calls `plan_virtual_cameras(roots, ball_arr, union)` and writes only
`{name}_pos`, `{name}_look`, `{name}_fov_deg` to `cameras.npz`. The three mounts — `broadcast`,
`sideline`, `goal` — are **planned from the reconstructed player positions**, not derived from the
calibration. `broadcast` is a synthetic main-stand mount at `(0, −(hy + out), 12.0)`, not the
operator's camera.

That is the right design for a *novel* view. It also means the solved rotation, `fx/fy` and
`cx/cy` never reach Blender at all, and there is no render of the original viewpoint to put beside
the real frame.

## 4. Why this is not cosmetic — the open residual has a candidate

`fit_rigid_camera.py:110`:

```python
def kmat(focal: float) -> np.ndarray:
    return np.array([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1.0]])
```

The principal point is **assumed to be the image centre and never fitted**, and
`apply_rigid_camera.py:138` writes that assumption into every scene as if it were measured. But
PnLCalib returns a `principal_point`, and `homography_from_camera_params` (`calibration.py:645`)
consumes `x_focal_length`, `y_focal_length` and `principal_point` as three separate things. So the
information exists upstream of the fit that discards it.

**STATUS #140 leaves this open:** *"Remaining residual grows 6.2 → 15.7 px centre-to-edge, which is
where distortion becomes testable."* A displaced principal point produces **exactly that
signature** — small error at the centre, growing toward the edge — and it is 2 free parameters
against a distortion model's 2–5, with a measurement already available for one of them.

**So the principal point should be tested before distortion is reached for.** If the edge residual
falls when `cx, cy` are freed, the growth was never distortion and a distortion model fitted first
would have absorbed the error into the wrong term and looked like it worked.

## 5. For camlab specifically — a capability that cannot land

`docs/camlab-spec.md` §5.4 specifies the numeric panel as *«`X Y Z` (м), `yaw / pitch / roll` (°),
`focal` (px) **и** `FOV гориз.` (°)»*.

`roll` has **no field to be stored in** on the render side, and `fov_y` exists nowhere. A human who
dials in a roll in camlab would see the reprojection error move, save, and have the value silently
dropped between the editor and the render.

That is **#141 before the fact** — a capability that exists in the editor and cannot reach the run
— and it is cheap to prevent now: either add `roll` (and `cx/cy`) to
`core.scene.cameras.CameraTrack`, or have the panel refuse the field and say why.

## 6. What this does not claim

- **Whether any of the four matters to the eye.** Broadcast pixels are square, so `fx ≠ fy` is
  almost certainly worth nothing; a real broadcast camera keeps its horizon level, so roll on the
  *virtual* cameras is a nicety. The two with a measured argument behind them are **`cx`, `cy`**.
- **That the principal point is displaced.** That is a hypothesis with a matching signature, not a
  measurement. It needs the fit run with two more free parameters over the same 60 frames — the
  same instrument that produced 2.35 px, plus two columns in the Jacobian.
- **The 180° roll landmine is unrelated to this.** That one is a convention mismatch between the
  solved camera and raw video, handled by rotating the frame (`raw_frame_aligned`), not by a roll
  parameter.
