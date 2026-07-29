"""R7 (#99): the accuracy metric for a video judged **by eye at a novel viewpoint**.

The briefs spec Global MPJPE 0.35-0.45 m and warn against speccing below the broadcast
envelope. That is the right bar for a coaching-analytics product, where a world coordinate is
the deliverable. It is the wrong bar for ours, and the briefs' own decomposition says why:
roughly 55% of global error is camera-driven, and camera error is **common-mode** — it moves
every player by the same rigid transform. Re-render that scene from a new viewpoint and a
common-mode error is a slightly different novel camera. The viewer cannot see it, because there
is no reference frame in the shot to see it against.

What the viewer *can* see is the part that survives any re-placement of the camera: players
standing in the wrong spots relative to each other, and a scene that swims under a camera that
should be still. So this module scores an error field by how much of it a camera re-fit can
absorb, and reports the remainder:

``global_mpjpe_m``
    The briefs' number, unmodified, so the two bars stay comparable.
``after_static_camera_m``
    Best single rigid re-placement of the whole clip. Removes a fixed camera error — free.
``after_perframe_camera_m``
    Best rigid re-placement *per frame*. **The R7 headline**: what no camera choice can fix.
``scene_swim_m``
    The gap between those two: common-mode error that *changes over time*. A rigid scene
    sliding under a locked-off camera is visible, so this is emphatically not free — splitting
    it out is the point, since a single whole-clip fit would hide it inside "camera error".

These are three means of a shrinking error, not orthogonal components; the drop from one to the
next is what that camera freedom absorbs, and adding them is meaningless.

**Scale is deliberately not absorbed.** A similarity fit would swallow the ~3x scale defect in
#61, and that defect is not free: our novel view renders players against a true-size pitch, so
wrong-scale players are visibly wrong-size. The fit is rigid, and the similarity scale is
reported alongside as a *diagnostic* — to tell "our error is a scale error" apart from "our
error is scatter" — never folded into the headline.

The second half is the speed stratification. Mean local MPJPE hides exactly the frames that
matter: this project already found, independently, that a yaw low-pass removed 90% of the
jitter while flattening real 100-degree turns. A mean cannot see that trade; the top speed
decile can. Any future temporal smoother is gated on beating plain Gaussian smoothing there.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "MIN_SUBJECTS",
    "decompose_global_error",
    "fit_rigid",
    "local_mpjpe_by_speed",
    "per_player_residual_m",
]

#: The metric is *defined* by several players sharing one camera — with a single subject a
#: per-frame fit would absorb that subject's own placement error and report it as "camera".
#:
#: Two is the mathematical floor, not a usable one. The per-frame fit has 6 DOF, so with few
#: bodies it launders genuine per-player scatter into "camera error". Measured by
#: ``scripts/bench_novel_view_metric.py`` on pure per-player scatter, where an honest metric
#: must absorb 0% (mean ± sd over 40 draws, worst draw in brackets):
#:
#: ===========  ==========================
#: subjects     falsely absorbed
#: ===========  ==========================
#: 2            70.9% ± 18.9  [97.7%]
#: 3            49.7% ± 17.8  [83.8%]
#: 5            30.1% ± 11.5  [63.8%]
#: 8            16.3% ±  7.8  [41.7%]
#: 12            9.0% ±  4.8  [18.1%]
#: 21            6.8% ±  4.0  [18.8%]
#: ===========  ==========================
#:
#: The target clip carries 21 subjects. Even there the leak does not vanish, so
#: ``after_perframe_camera_m`` is a **lower bound** on what a viewer sees — never an unbiased
#: estimate, and meaningless below roughly 8 bodies.
MIN_SUBJECTS = 2

#: Below this many finite points a rigid fit is unconstrained; such frames absorb nothing.
_MIN_FIT_POINTS = 3


def fit_rigid(
    pred: np.ndarray, gt: np.ndarray, *, with_scale: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares map ``s·R·pred + t ≈ gt`` (Kabsch / Umeyama) over points ``(M, 3)``.

    Returns ``(R, t, s)``; ``s`` is 1.0 unless ``with_scale``. The reflection-free branch is
    taken explicitly — an SVD will happily return ``det(R) = -1``, which is a mirror, and a
    mirrored scene is the failure mode ``tests/unit/test_golden_projection_sign.py`` exists for.
    """
    p, g = np.asarray(pred, float), np.asarray(gt, float)
    if p.shape != g.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected matching (M, 3) point sets, got {p.shape} and {g.shape}")
    if p.shape[0] < _MIN_FIT_POINTS:
        return np.eye(3), np.zeros(3), 1.0

    pc, gc = p.mean(0), g.mean(0)
    pd, gd = p - pc, g - gc
    u, sv, vt = np.linalg.svd(pd.T @ gd)
    flip = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(vt.T @ u.T)))])
    rot = vt.T @ flip @ u.T

    scale = 1.0
    if with_scale:
        var = float((pd**2).sum())
        scale = float((sv * np.diag(flip)).sum() / var) if var > 0 else 1.0
    return rot, gc - scale * rot @ pc, scale


def _apply(rot: np.ndarray, transl: np.ndarray, scale: float, pts: np.ndarray) -> np.ndarray:
    return scale * pts @ rot.T + transl


def _checked(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None):
    """Validate the ``(T, N, J, 3)`` contract and return arrays plus a broadcast bool mask."""
    p, g = np.asarray(pred, float), np.asarray(gt, float)
    if p.shape != g.shape or p.ndim != 4 or p.shape[-1] != 3:
        raise ValueError(f"expected matching (T, N, J, 3) arrays, got {p.shape} and {g.shape}")
    if p.shape[1] < MIN_SUBJECTS:
        raise ValueError(
            f"the common-mode decomposition needs at least {MIN_SUBJECTS} subjects sharing a "
            f"camera; got {p.shape[1]}. With one subject a per-frame rigid fit absorbs that "
            "subject's own placement error and mislabels it as camera error."
        )
    if mask is None:
        return p, g, np.ones(p.shape[:-1], dtype=bool)
    m = np.asarray(mask, dtype=bool)
    if m.shape != p.shape[:-1]:
        raise ValueError(f"visibility mask {m.shape} does not match joints {p.shape[:-1]}")
    return p, g, m


def _align_per_frame(p: np.ndarray, g: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Per-frame rigid fit over every visible joint of every subject → aligned ``(T, N, J, 3)``."""
    out = np.empty_like(p)
    for t in range(p.shape[0]):
        vis = m[t]
        rot, transl, scale = fit_rigid(p[t][vis], g[t][vis])
        out[t] = _apply(rot, transl, scale, p[t])
    return out


def _align_whole_clip(p, g, m, *, with_scale: bool = False):
    """One rigid (or similarity) fit over the whole clip → ``(aligned, R, t, s)``."""
    rot, transl, scale = fit_rigid(p[m], g[m], with_scale=with_scale)
    return _apply(rot, transl, scale, p), rot, transl, scale


def _mean_err(p: np.ndarray, g: np.ndarray, m: np.ndarray) -> float:
    err = np.linalg.norm(p - g, axis=-1)[m]
    return float(err.mean()) if err.size else float("nan")


def per_player_residual_m(
    pred_world: np.ndarray, gt_world: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Each subject's mean error ``(N,)`` after the per-frame common-mode fit is removed.

    This is the R7 headline broken out by player: the spread across these values is what a
    viewer reads as "that one is standing in the wrong place".
    """
    p, g, m = _checked(pred_world, gt_world, mask)
    err = np.linalg.norm(_align_per_frame(p, g, m) - g, axis=-1)
    return np.array(
        [
            float(err[:, n][m[:, n]].mean()) if m[:, n].any() else float("nan")
            for n in range(p.shape[1])
        ]
    )


def decompose_global_error(
    pred_world: np.ndarray, gt_world: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float]:
    """Split global error into what a camera re-fit absorbs and what a viewer is left with.

    ``pred_world`` / ``gt_world`` are ``(T, N, J, 3)`` world metres; ``mask`` is an optional
    ``(T, N, J)`` visibility bool. See the module docstring for what each key means and why
    ``scene_swim_m`` is reported apart from the static camera error rather than with it.
    """
    p, g, m = _checked(pred_world, gt_world, mask)

    static, rot, _, _ = _align_whole_clip(p, g, m)
    similarity, _, _, scale = _align_whole_clip(p, g, m, with_scale=True)

    raw_m = _mean_err(p, g, m)
    static_m = _mean_err(static, g, m)
    perframe_m = _mean_err(_align_per_frame(p, g, m), g, m)
    players = per_player_residual_m(p, g, m)

    cos = (float(np.trace(rot)) - 1.0) / 2.0
    return {
        "global_mpjpe_m": raw_m,
        "after_static_camera_m": static_m,
        "after_perframe_camera_m": perframe_m,
        "scene_swim_m": static_m - perframe_m,
        "camera_absorbed_frac": 1.0 - perframe_m / raw_m if raw_m > 0 else 0.0,
        "player_spread_m": float(np.nanmax(players) - np.nanmin(players)),
        "static_fit_rotation_deg": float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))),
        # diagnostics: a scale error is NOT absorbed above, because it is not free at a novel
        # viewpoint — players are rendered against a true-size pitch. Reported to tell a scale
        # defect (#61) apart from scatter. Inverted from the fit's correction factor so it reads
        # as the defect itself: 1.0 is correct, 3.0 means "our scene is three times too big".
        "predicted_scale": 1.0 / scale if scale > 0 else float("inf"),
        "after_similarity_m": _mean_err(similarity, g, m),
    }


def _articulation_speed(g: np.ndarray, fps: float, root: int) -> np.ndarray:
    """Root-relative joint speed ``(T, N, J)`` in m/s, central differences, one-sided at edges.

    Root-relative because this stratifies *local* (articulation) MPJPE: a sprinting player whose
    arms are still is slow by this measure, which is the correct reading for a metric that has
    already subtracted the root.
    """
    rel = g - g[:, :, root : root + 1, :]
    if rel.shape[0] < 2:
        return np.zeros(rel.shape[:-1])
    return np.linalg.norm(np.gradient(rel, 1.0 / float(fps), axis=0), axis=-1)


def local_mpjpe_by_speed(
    pred_world: np.ndarray,
    gt_world: np.ndarray,
    fps: float,
    root: int = 0,
    mask: np.ndarray | None = None,
    quantile: float = 0.9,
) -> dict[str, float]:
    """Root-relative MPJPE overall vs. in the fastest-moving decile of joint-frames.

    Speed is taken from the **ground truth**, so the strata do not move when the prediction
    changes and two methods are always compared on the same joint-frames. ``quantile=0.9``
    is the top decile; the bottom decile is reported next to it because the contrast is the
    finding — a smoother that trades the two looks free on the mean.
    """
    p, g = np.asarray(pred_world, float), np.asarray(gt_world, float)
    if p.shape != g.shape or p.ndim != 4 or p.shape[-1] != 3:
        raise ValueError(f"expected matching (T, N, J, 3) arrays, got {p.shape} and {g.shape}")
    m = np.ones(p.shape[:-1], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    err = np.linalg.norm(
        (p - p[:, :, root : root + 1, :]) - (g - g[:, :, root : root + 1, :]), axis=-1
    )
    speed = _articulation_speed(g, fps, root)
    fast = speed >= np.quantile(speed[m], quantile) if m.any() else np.zeros_like(m)
    slow = speed <= np.quantile(speed[m], 1.0 - quantile) if m.any() else np.zeros_like(m)

    overall = _sel(err, m)
    top = _sel(err, m & fast)
    return {
        "local_mpjpe_m": overall,
        "local_mpjpe_top_decile_m": top,
        "local_mpjpe_bottom_decile_m": _sel(err, m & slow),
        "top_decile_penalty": top / overall if overall > 0 else float("nan"),
        "speed_threshold_m_s": float(np.quantile(speed[m], quantile)) if m.any() else float("nan"),
    }


def _sel(err: np.ndarray, sel: np.ndarray) -> float:
    return float(err[sel].mean()) if sel.any() else float("nan")
