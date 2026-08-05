"""Kit zones — the face/limb lever (2026-07-04).

Batch #1 rendered every player as a whole-body team-colour "morphsuit": the measured texture is
too dark/noisy at broadcast distance to carry the kit LAYOUT, and the flat fallback has none, so
shirt/shorts/skin all painted the same and the generative tail kept it. The layout now comes from
the body model itself (dominant LBS joint per vertex → garment zone) while the COLOURS stay
measured (per-team zone medians pooled across subjects). These pin the zone assignment (including
the knee split and the shorts-length thigh cut), the median pooling, and the fallback chain
(shorts→shirt, socks→shorts, skin→tan prior) with its manual env overrides.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.adapters.models.avatar import (
    KIT_BOOTS,
    KIT_SHIRT,
    KIT_SHORTS,
    KIT_SKIN,
    KIT_SOCKS,
    smplx_kit_zones,
)
from pitch3d.adapters.models.smplx_lbs import locate_smplx_model
from pitch3d.app.anim_export import _compose_kit_vcolor, _team_zone_medians


def _synthetic_rig():
    """A toy skeleton: joint markers pinned by a one-hot regressor, one-hot vertex weights.

    Joint heights (Y-up, like the SMPL-X template): hips at 0, knees at -1, ankles at -1.6 —
    so the shorts thigh cut sits at hip + 0.55*(knee-hip) = -0.55.
    """
    n_joints = 22
    joint_y = np.zeros(n_joints)
    joint_y[[4, 5]] = -1.0  # knees
    joint_y[[7, 8]] = -1.6  # ankles
    # (dominant joint, template y, expected zone)
    cases = [
        (16, 0.4, KIT_SHIRT),  # shoulder → short sleeve
        (0, -0.2, KIT_SHORTS),  # pelvis
        (1, -0.3, KIT_SHORTS),  # upper thigh, above the cut
        (1, -0.7, KIT_SKIN),  # lower thigh, below the cut → bare skin
        (4, -0.9, KIT_SKIN),  # knee-dominated, above the knee joint → thigh side
        (4, -1.1, KIT_SOCKS),  # knee-dominated, below the knee joint → calf side
        (7, -1.5, KIT_SOCKS),  # ankle
        (10, -1.7, KIT_BOOTS),  # foot
        (15, 0.6, KIT_SKIN),  # head
        (20, 0.1, KIT_SKIN),  # wrist → bare arm
    ]
    n_body = len(cases)
    verts = n_body + n_joints
    weights = np.zeros((verts, n_joints))
    v_template = np.zeros((verts, 3))
    for i, (j, y, _) in enumerate(cases):
        weights[i, j] = 1.0
        v_template[i, 1] = y
    # Joint-marker vertices: row j of the regressor picks marker vertex n_body+j exactly.
    j_regressor = np.zeros((n_joints, verts))
    for j in range(n_joints):
        weights[n_body + j, j if j not in (4, 5, 1, 2, 0) else 15] = 1.0  # park markers on skin
        j_regressor[j, n_body + j] = 1.0
        v_template[n_body + j, 1] = joint_y[j]
    return weights, j_regressor, v_template, [z for _, _, z in cases]


def test_zones_knee_split_and_thigh_cut_on_synthetic_rig():
    weights, j_regressor, v_template, expected = _synthetic_rig()
    zones = smplx_kit_zones(weights, j_regressor, v_template)
    assert zones[: len(expected)].tolist() == expected


def test_zones_on_real_smplx_model_are_anatomically_ordered():
    path = locate_smplx_model("models/smplx")
    if path is None:
        pytest.skip("no local SMPL-X model")
    from pitch3d.adapters.models.smplx_lbs import SmplxModel

    m = SmplxModel.load(path)
    zones = smplx_kit_zones(m.weights, m.j_regressor, m.v_template)
    assert zones.shape == (m.v_template.shape[0],)
    counts = {z: int((zones == z).sum()) for z in range(5)}
    assert all(c > 0 for c in counts.values())
    # The kit stacks: shirt above shorts above socks above boots; skin spans head→thigh gap.
    y = m.v_template[:, 1]
    mean_y = {z: float(y[zones == z].mean()) for z in range(5)}
    assert mean_y[KIT_SHIRT] > mean_y[KIT_SHORTS] > mean_y[KIT_SOCKS] > mean_y[KIT_BOOTS]
    joints = m.j_regressor @ m.v_template
    # Socks stay below the knee (2 cm slack: the sock-top ring hugs the joint), shorts above
    # it, boots below the ankle.
    knee_y, ankle_y = float(joints[4, 1]), float(joints[7, 1])
    assert y[zones == KIT_SOCKS].max() <= knee_y + 0.02
    assert y[zones == KIT_SHORTS].min() >= knee_y
    assert y[zones == KIT_BOOTS].max() <= ankle_y


def _posed(team, vcolor, measured):
    class _Subj:
        team_id = team

    return dict(subj=_Subj(), vcolor=vcolor, measured=measured)


def test_team_zone_medians_pool_across_subjects_and_require_coverage():
    v = 120
    zones = np.full(v, KIT_SHORTS, dtype=np.uint8)
    m1 = np.zeros(v, bool)
    m1[:30] = True  # 30 samples alone — below the 40 floor
    m2 = np.zeros(v, bool)
    m2[30:60] = True
    med = _team_zone_medians(
        [
            _posed("A", np.full((v, 3), 0.2, np.float32), m1),
            _posed("A", np.full((v, 3), 0.4, np.float32), m2),
            _posed("B", None, None),  # no source clip → skipped, team still listed
        ],
        zones,
    )
    assert np.allclose(med["A"][KIT_SHORTS], 0.3)  # pooled 60 ≥ 40 → median of both subjects
    assert med["A"][KIT_SOCKS] is None and med["A"][KIT_SKIN] is None
    assert med["B"][KIT_SHORTS] is None


def test_zone_estimate_survives_grass_and_line_misprojections():
    # 2026-07-04, measured twice on the target clip: raw pooled medians were grass-green for
    # EVERY zone of BOTH teams, and after dropping grass they were line/LED white — thin legs
    # put MOST samples on the background. Grass gate + saturated-hue-mode must leave the
    # minority navy shorts owning the estimate; all-grass stays None; a genuinely neutral kit
    # (white socks, few saturated strays) takes the plain-median path.
    v = 140
    zones = np.full(v, KIT_SHORTS, dtype=np.uint8)
    vcolor = np.full((v, 3), (0.43, 0.57, 0.24), np.float32)  # grass pollutant
    vcolor[:40] = (0.75, 0.75, 0.75)  # white pitch lines / LED glare
    vcolor[40:85] = (0.10, 0.12, 0.35)  # navy shorts — a minority of all samples
    med = _team_zone_medians([_posed("A", vcolor, np.ones(v, bool))], zones)
    assert np.allclose(med["A"][KIT_SHORTS], [0.10, 0.12, 0.35], atol=1e-6)
    all_grass = np.full((v, 3), (0.43, 0.57, 0.24), np.float32)
    med2 = _team_zone_medians([_posed("A", all_grass, np.ones(v, bool))], zones)
    assert med2["A"][KIT_SHORTS] is None
    white_socks = np.full((v, 3), (0.82, 0.82, 0.80), np.float32)
    white_socks[:10] = (0.6, 0.1, 0.1)  # a stray opponent leg — under the 30% saturated gate
    med3 = _team_zone_medians([_posed("A", white_socks, np.ones(v, bool))], zones)
    assert np.allclose(med3["A"][KIT_SHORTS], [0.82, 0.82, 0.80], atol=1e-6)


def test_compose_uses_medians_then_falls_back_down_the_kit(monkeypatch):
    for var in ("PITCH3D_SHORTS_RGB_A", "PITCH3D_SOCKS_RGB_A", "PITCH3D_SKIN_RGB"):
        monkeypatch.delenv(var, raising=False)
    zones = np.array([KIT_SKIN, KIT_SHIRT, KIT_SHORTS, KIT_SOCKS, KIT_BOOTS], np.uint8)
    shirt = np.array([1.0, 0.9, 0.0], np.float32)
    medians = {"A": {KIT_SHORTS: np.array([0.1, 0.1, 0.3]), KIT_SOCKS: None, KIT_SKIN: None}}
    v = _compose_kit_vcolor(zones, shirt, medians, "A", 1.0, 1.0)  # no boost → exact colours
    assert np.allclose(v[1], shirt)
    assert np.allclose(v[2], [0.1, 0.1, 0.3])
    assert np.allclose(v[3], v[2])  # socks unmeasured → shorts colour (one-colour kit)
    assert np.allclose(v[0], [0.70, 0.52, 0.38])  # skin unmeasured → tan prior
    assert np.allclose(v[4], 0.05)
    # No medians at all (e.g. untracked group) → whole kit collapses to the shirt colour.
    v2 = _compose_kit_vcolor(zones, shirt, {}, "", 1.0, 1.0)
    assert np.allclose(v2[2], shirt) and np.allclose(v2[3], shirt)


def test_compose_env_overrides_win(monkeypatch):
    monkeypatch.setenv("PITCH3D_SOCKS_RGB_A", "1,1,1")
    monkeypatch.setenv("PITCH3D_SKIN_RGB", "0.5 0.4 0.3")
    zones = np.array([KIT_SKIN, KIT_SOCKS], np.uint8)
    medians = {"A": {KIT_SHORTS: None, KIT_SOCKS: np.array([0.2, 0.2, 0.2]), KIT_SKIN: None}}
    v = _compose_kit_vcolor(zones, np.array([1.0, 0.0, 0.0]), medians, "A", 1.0, 1.0)
    assert np.allclose(v[1], 1.0)  # override beats the measured median
    assert np.allclose(v[0], [0.5, 0.4, 0.3])
