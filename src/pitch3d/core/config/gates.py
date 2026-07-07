"""Pure per-gate config dataclasses — no imports beyond stdlib.

Kept separate from ``physics.py`` so the correction modules (``foot_floor``,
``joint_kinematics``, ``orientation``) can consume their configs without pulling
the YAML loader (which itself imports ``KinematicConfig`` / ``CoherenceConfig``
from ``core.correction``). Breaking that cycle lives here — this module has no
package-internal imports.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FootFloorConfig:
    """Auto-default foot-floor clamp (Tier 1a) — read from ``config/physics.yaml``."""

    enabled: bool = False
    floor_m: float = 0.0
    warn_hover_m: float = 0.30


@dataclass(frozen=True)
class JointKinematicConfig:
    """Per-joint angular-velocity gate (Tier 1b). Schema reserved, not built."""

    enabled: bool = False
    max_omega_dps: float = 600.0


@dataclass(frozen=True)
class OrientationConfig:
    """Root-orientation turn-rate gate (Tier 1c). Schema reserved, not built."""

    enabled: bool = False
    max_turn_rate_dps: float = 720.0


@dataclass(frozen=True)
class BallConfig:
    """Ball measurement thresholds — used by probes, not a gate yet."""

    max_speed: float = 36.0
    max_accel: float = 200.0


@dataclass(frozen=True)
class IdentityConfig:
    """Identity gate (GTA-style split + cross-track merge) to keep a player's
    kit / team stable across the whole video.

    Symptom being closed: within a single ByteTrack track_id, appearance can
    change if the tracker ID-swapped (one physical player crossed the frame
    and their track got hijacked by another). The scene then paints ONE person
    in TWO colours across the clip. GTA (Sun et al. ACCV 2024) tackles this
    with two passes:

    1. **Split** — DBSCAN over per-frame appearance features INSIDE each
       track; clusters ≥ 2 → the track fused two people → split.
    2. **Merge** — cross-track pairwise cosine-distance over the whole
       tracklet set; disjoint pairs closer than ``merge_cosine_threshold``
       are the same physical player under a different track_id → merge.
    """

    enabled: bool = False
    #: DBSCAN eps in feature-cosine-distance units. Larger = more permissive
    #: (fewer splits). 0.20 is a reasonable start for HSV features.
    dbscan_eps: float = 0.20
    #: Minimum frames a cluster must contain to count. Discards tiny outliers.
    dbscan_min_samples: int = 4
    #: A split is only accepted when the two clusters are separated by at
    #: least this many contiguous frames — protects against per-frame flicker
    #: masquerading as an identity change.
    min_split_gap_frames: int = 3
    #: Enable the cross-track merge pass after the split pass.
    merge_enabled: bool = True
    #: Cosine distance below which two tracklets' mean features are considered
    #: the same identity for the merge pass. Tighter than dbscan_eps so we
    #: don't fuse team members into a single track.
    merge_cosine_threshold: float = 0.10
    #: Merge is only considered for tracklet pairs whose frame ranges do NOT
    #: overlap (temporal disjointness). Setting >0 also enforces a maximum
    #: gap between end(a) and start(b); a large gap = probably a different
    #: physical player who happens to wear the same kit shade.
    merge_max_gap_frames: int = 60
    #: When True, detect splits/merges but do NOT mutate the tracks. Useful
    #: for A/B evaluation before enabling the emit path in production.
    dry_run: bool = False


@dataclass(frozen=True)
class FootPlantConfig:
    """T6.a — foot plant: recenter root Z so subjects don't hover.

    Modes:
      * ``off`` — measure only, no corrections emitted.
      * ``median_lock`` — shift each subject's whole Z track by
        ``target_pelvis_m - median(Z)`` when the offset exceeds
        ``bias_threshold_m``. Preserves stride/jump variance but kills the
        systematic bias that makes every player float.
      * ``hard_lock`` — clamp every frame's Z to ``target_pelvis_m``. Kills
        legitimate jumps too; use only for debug.
    """

    enabled: bool = False
    mode: str = "median_lock"           # "off" | "median_lock" | "hard_lock"
    target_pelvis_m: float = 0.92       # nominal pelvis-above-floor for a standing player
    bias_threshold_m: float = 0.05      # skip subjects whose median Z is already within this
    min_correction_m: float = 1e-4


@dataclass(frozen=True)
class CollisionConfig:
    """Capsule-collision post-process (Tier 3).

    Not a physics sim: at each frame the subjects are treated as vertical
    capsules of radius ``capsule_radius_m`` on the pitch plane; overlapping
    pairs get a soft push apart, one Jacobi iteration per pass. ``strength=1.0``
    fully resolves each overlap in one pass; ``0.5`` halves it (softer, less
    twitchy). ``n_passes`` iterates the Jacobi step so a stack of three near
    each other converges.
    """

    enabled: bool = False
    capsule_radius_m: float = 0.35     # ~ shoulder half-width for a standing player
    strength: float = 0.5              # fraction of overlap resolved per pass
    n_passes: int = 4                  # bounded iterations per frame
    max_push_per_frame_m: float = 0.30  # safety cap — never move a subject more than this
    min_correction_m: float = 1e-4      # skip emitting a correction below this net push


@dataclass(frozen=True)
class MomentumSmoothConfig:
    """Step 4b — low-pass root translation to kill CoM jerk."""

    enabled: bool = False
    smooth_window: int = 5
    preserve_contact: bool = True
    contact_z_threshold_m: float = 0.05
    min_contact_run_frames: int = 2
    min_correction_m: float = 1e-3


@dataclass(frozen=True)
class PoseMotionSyncConfig:
    """Procedural walk-cycle patch when the pose is desynced from root motion."""

    enabled: bool = False
    velocity_threshold_mps: float = 1.0
    joint_activity_threshold: float = 0.10
    knee_amplitude_rad: float = 0.35
    hip_amplitude_rad: float = 0.20
    full_speed_mps: float = 6.0
    strides_per_metre: float = 0.7


@dataclass(frozen=True)
class FacingAlignConfig:
    """Rotate body yaw to match motion direction (moonwalking fix)."""

    enabled: bool = False
    velocity_threshold_mps: float = 1.0
    yaw_tolerance_rad: float = 0.79
    yaw_ewma_window: int = 5


@dataclass(frozen=True)
class InertiaSmoothConfig:
    """Low-pass yaw to bound angular acceleration."""

    enabled: bool = False
    smooth_window: int = 3
    max_alpha_rad_s2: float = 15.0
    min_correction_rad: float = 1e-3


@dataclass(frozen=True)
class JerkClampConfig:
    """Iteratively low-pass root translation until peak jerk under ceiling."""

    enabled: bool = False
    max_jerk_mps3: float = 200.0
    smooth_window: int = 5
    max_passes: int = 10
    min_correction_m: float = 1e-3


@dataclass(frozen=True)
class JointSmoothConfig:
    """Low-pass over per-joint axis-angle body_pose to kill twitch/jitter."""

    enabled: bool = False
    smooth_window: int = 5
    min_correction_rad: float = 5e-3


@dataclass(frozen=True)
class GravityProjectConfig:
    """Force airborne Z onto a ballistic parabola."""

    enabled: bool = False
    airborne_z_threshold_m: float = 0.10
    min_airborne_run_frames: int = 3
    min_correction_m: float = 1e-3


@dataclass(frozen=True)
class ContactProbeConfig:
    """Step 3 probe — foot-contact detection + slide measurement.

    A foot is CONTACTED when its Z sits below ``contact_z_threshold_m`` for
    at least ``min_contact_run_frames`` consecutive frames. Within each
    contact run we measure the XY displacement of the foot: if it drifted
    by more than ``slide_threshold_m`` the subject was foot-sliding
    (unphysical — the classic "glide-walking" complaint).

    This is measurement-only; the correction that zeroes the sliding
    velocity is the next iteration (WHAM contact head + IK anti-slide).
    """

    enabled: bool = False
    contact_z_threshold_m: float = 0.05     # foot Z below floor + this = contact
    min_contact_run_frames: int = 2         # ignore single-frame touches
    slide_threshold_m: float = 0.05         # ignore < 5cm slides (jitter)


@dataclass(frozen=True)
class ProbeConfig:
    """Thresholds only ``scripts/motion_stats.py`` (and future probes) consume."""

    turn_min_speed: float = 2.0
    joint_min_omega_dps: float = 600.0
    orient_min_dps: float = 720.0
    foot_hover_m: float = 0.30
