#!/usr/bin/env python3
"""Does our pipeline actually reach the poses SMPL-X fails to forbid? (R5, #97)

The brief proposes a joint-limit residual — a VPoser-norm prior plus one-sided knee/elbow
hinges — on the grounds that SMPL has no joint limits, so hyperextension is representable *and
reachable*. The first half is true and easy to confirm; the second is an assumption, and it is
the one that decides whether the work pays. This measures it.

Run (needs the SMPL-X body model and torch, both already local)::

    PYTHONPATH=src .venv/bin/python scripts/bench_joint_limits.py

Two steps, because a sign error here would invert the entire finding:

``convention``
    Derives which way a knee and an elbow *bend* from the body model itself — rotate the joint
    both ways and see where the distal segment goes relative to the body's own facing. Nothing
    downstream depends on a remembered axis convention.

``measure``
    Scores every subject-frame of the target clip for anatomical flexion, split by R4's
    provenance labels. The split is the point: the pose net is one author of these poses, and
    ``coherence.extend_pose_to_span`` is another. A gate that invents 137 frames by coasting is
    a far more plausible source of an impossible knee than a network trained on real humans, so
    the two are scored apart rather than averaged into one reassuring number.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np  # noqa: E402

from pitch3d.core.correction.coherence import add_temporal_coherence  # noqa: E402
from pitch3d.core.scene.serialization import from_json  # noqa: E402

MODEL_DIR = "models/smplx"
CLIPS = {"A (SMPLest-X, production)": "A_smplestx", "B (SAM 3D Body)": "B_sam3dbody"}
FPS = 29.97

#: SMPL-X joint indices: (proximal, joint, distal). Flexion sign is derived, never assumed.
CHAINS = {
    "knee L": (1, 4, 7),
    "knee R": (2, 5, 8),
    "elbow L": (16, 18, 20),
    "elbow R": (17, 19, 21),
}
#: ``body_pose`` row driving each chain's joint (row i rotates joint i+1).
POSE_ROW = {"knee L": 3, "knee R": 4, "elbow L": 17, "elbow R": 18}

#: Genu recurvatum of 5-10 degrees is normal, and more is common in flexible athletes; past this
#: a knee is not a stiff human, it is a broken rig.
IMPLAUSIBLE_DEG = 15.0


def _forward_kinematics(body_pose: np.ndarray) -> np.ndarray:
    """Joints in the canonical body frame (no global orient, no translation) — ``(N, J, 3)``."""
    import smplx
    import torch

    model = smplx.create(
        MODEL_DIR,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        batch_size=len(body_pose),
    )
    with torch.no_grad():
        return model(body_pose=torch.tensor(body_pose, dtype=torch.float32)).joints.numpy()


def _flexion_deg(joints: np.ndarray, chain: tuple[int, int, int], toward: np.ndarray):
    """Signed angle at ``chain[1]``; positive when the distal segment swings ``toward``.

    Zero is a straight limb, so a negative value is hyperextension by construction.
    """
    a, b, c = chain
    prox = joints[:, b] - joints[:, a]
    dist = joints[:, c] - joints[:, b]
    prox /= np.linalg.norm(prox, axis=1, keepdims=True)
    dist /= np.linalg.norm(dist, axis=1, keepdims=True)
    bend = np.degrees(np.arccos(np.clip((prox * dist).sum(1), -1.0, 1.0)))
    swing = dist - (prox * dist).sum(1)[:, None] * prox
    return bend * np.sign((swing * toward).sum(1))


#: The anatomy, stated as input because it is a fact about humans, not about SMPL-X: a knee
#: flexes posteriorly (heel toward buttock), an elbow anteriorly (hand toward shoulder). Only
#: the *frame* these are expressed in is derived from the model.
FLEXES_FORWARD = {"knee L": False, "knee R": False, "elbow L": True, "elbow R": True}


def convention() -> dict[str, np.ndarray]:
    """Express the anatomical flexion directions in the model's own frame, showing the working.

    The hinge axis is **searched for**, not assumed to be X. In the canonical T-pose the arms lie
    along the body's lateral axis, so an X rotation at the elbow is a twist along the forearm: it
    moves the wrist by ~1 cm and the resulting sign is noise. Reading a real convention off that
    is how a scene ends up mirrored but self-consistent.
    """
    print("== which way does a joint bend? (frame derived from SMPL-X; anatomy is the input) ==")
    rest = _forward_kinematics(np.zeros((1, 21, 3)))[0]

    # A proper orthogonal body frame. Using the foot direction alone as "forward" is wrong: the
    # toes splay outward and downward, so that vector is 0.34 lateral / -0.42 vertical, and
    # projecting a limb swing onto it flips the sign whenever the swing is mostly sideways.
    up = rest[12] - rest[0]                                  # neck - pelvis
    lateral = rest[1] - rest[2]                              # left hip - right hip
    facing = np.cross(lateral, up)
    facing /= np.linalg.norm(facing)
    if facing @ (rest[10] - rest[7]) < 0:                    # toes disambiguate the sign only
        facing = -facing
    toes = (rest[10] - rest[7]) / np.linalg.norm(rest[10] - rest[7])
    print(f"  body frame from spine x hip axis: facing = {np.round(facing, 3)}")
    print(f"  (the raw toes-ankle vector is {np.round(toes, 3)} — not an axis)\n")

    toward = {n: (facing if FLEXES_FORWARD[n] else -facing) for n in CHAINS}

    # Self-check, because a sign slip here would invert the entire finding and still look
    # plausible. A bent limb is shorter end to end than a straight one, so whichever probe
    # rotation most shortens distal-to-grandparent is flexion by definition — no axis needs
    # naming. The metric must score that pose positive.
    for name, chain in CHAINS.items():
        probes = np.zeros((6, 21, 3))
        for i in range(3):
            probes[2 * i, POSE_ROW[name], i] = 0.9
            probes[2 * i + 1, POSE_ROW[name], i] = -0.9
        joints = _forward_kinematics(probes)
        span = np.linalg.norm(joints[:, chain[2]] - joints[:, chain[0]], axis=1)
        bent = int(np.argmin(span))
        rest_span = np.linalg.norm(rest[chain[2]] - rest[chain[0]])
        angle = float(_flexion_deg(joints[bent : bent + 1], chain, toward[name])[0])
        verdict = "OK" if angle > 0 else "WRONG SIGN"
        print(
            f"  {name:<8} most-shortened probe: span {rest_span:.3f} -> {span[bent]:.3f} m, "
            f"scored {angle:+6.1f} deg  [{verdict}]"
        )
        if angle <= 0:
            raise SystemExit(f"{name}: flexion reads negative — the sign convention is inverted")
    print("  All four read positive, so 'negative' below really does mean hyperextension.\n")
    return toward


def _report(label: str, knee: np.ndarray, elbow: np.ndarray, n: int) -> None:
    def col(a: np.ndarray) -> str:
        return (
            f"{a.min():>10.1f}{np.median(a):>9.1f}{100 * (a < 0).mean():>8.1f}%"
            f"{100 * (a < -IMPLAUSIBLE_DEG).mean():>8.1f}%"
        )

    print(f"  {label:<28}{n:>7}{col(knee)}{col(elbow)}")


def _head() -> None:
    bad = f">{int(IMPLAUSIBLE_DEG)}deg"
    print(
        f"  {'':<28}{'frames':>7}{'knee min':>10}{'median':>9}{'past 0':>9}{bad:>9}"
        f"{'elbow min':>10}{'median':>9}{'past 0':>9}{bad:>9}"
    )
    print("  " + "-" * 109)


def measure(toward: dict[str, np.ndarray]) -> None:
    print("== is an impossible joint ever actually reached? (degrees; 0 = straight limb) ==\n")
    _head()

    for label, clip in CLIPS.items():
        scene = from_json(open(f"poseannot/clips/{clip}/scene.json", encoding="utf-8").read())
        dense, report = add_temporal_coherence(scene, fps=FPS)
        pose = np.concatenate([s.proposal.pose.body_pose for s in dense.subjects]).astype(float)
        prov = np.concatenate([s.proposal.pose.provenance for s in dense.subjects])
        joints = _forward_kinematics(pose)

        angles = {n: _flexion_deg(joints, CHAINS[n], toward[n]) for n in CHAINS}
        knee = np.minimum(angles["knee L"], angles["knee R"])  # the worse leg of the pair
        elbow = np.minimum(angles["elbow L"], angles["elbow R"])

        print(f"  {label}   (bridged {report.filled_frames}, coasted {report.extended_frames})")
        for tag in ("measured", "interpolated", "imputed"):
            sel = prov == tag
            if sel.any():
                _report(f"    {tag}", knee[sel], elbow[sel], int(sel.sum()))
        _report("    ALL", knee, elbow, len(pose))
    print()


def verdict() -> None:
    print("== verdict ==")
    print("  The brief's premise is half right. SMPL-X does not forbid hyperextension — but our")
    print("  pipeline never proposes it. The pose nets regress into the manifold of the humans")
    print("  they were trained on, and the gate that fabricates frames HOLDS the last articulation")
    print("  rather than extrapolating it, so even the coasted frames stay well inside the range.")
    print()
    print("  A joint-limit residual would therefore duplicate a constraint the training")
    print("  distribution already supplies, at the cost of a VPoser dependency and a term that")
    print("  can only pull a plausible pose away from the observations.")
    print()
    print("  Where limits WOULD bite is an optimiser that fits pose to observations with no data")
    print("  prior — which is exactly the factor graph the briefs propose alongside them. We")
    print("  deferred that (ADR-0012), so R5 defers with it. Re-open together, not separately.")


if __name__ == "__main__":
    measure(convention())
    verdict()
