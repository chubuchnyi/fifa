"""ViewSynthesizer — generative novel-view (video-diffusion), TWO integration seams.

This is the single most important new contract in v0.3 (ADR-0007). One port exposes
both ways the toolkit uses generative novel-view:

* **Seam A — render adapter** (:meth:`render_orbit`). Re-shoots the clip from a *limited*
  alternative camera trajectory (an orbit/dolly around the broadcast view) and returns a
  photoreal **video** (`SynthViewRef`, ``seam=A_RENDER``, ``editable=False``). An adapter
  in ``adapters/render`` wraps this as a :class:`RenderPass`. Fast path to photoreal for
  moderate moves; NOT geometry, NOT editable (R-15).

* **Seam B — data amplifier** (:meth:`amplify`, :meth:`inpaint_occlusions`). Synthesizes
  extra viewpoints from the single camera to feed reconstruction (mono → pseudo-multi-view,
  FR-30) and inpaints unseen sides of subjects (FR-31). Output feeds
  :class:`EnvReconstructor` / :class:`AvatarBuilder`.

**Hard boundary (record in ADR-0007, enforce in UX):** generative novel-view requires
sufficient frustum overlap, so it is a *bounded* re-aim, not free-viewpoint. It does **not**
replace explicit 3D where editing (poses/trajectories) or an arbitrary free camera is needed.
Both seams emit ``frustum_overlap`` so callers can gate/limit application (R-14, R-16).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from ..scene.assets import SynthViewRef
from ..scene.camera import CameraTrack
from .base import ModelProvider
from .io import ClipRef, CropRef


class ViewSynthesizer(ModelProvider):
    """Generative novel-view backend (ReCamMaster / TrajectoryCrafter / GEN3C / …).

    Backends may differ per seam (seam A favors orbit fidelity; seam B favors 3D
    consistency); both are reached through this one port (NFR-6).
    """

    # --- Seam A: render adapter (limited orbit -> video) -----------------------
    @abstractmethod
    def render_orbit(
        self,
        clip: ClipRef,
        target_camera: CameraTrack,
        scene_hints: dict | None = None,
    ) -> SynthViewRef:
        """Re-shoot ``clip`` along ``target_camera`` (a *limited* orbit/dolly).

        Returns a `SynthViewRef` with ``seam=A_RENDER`` and ``editable=False`` whose
        ``uri`` points at the generated video, ``camera`` is the prescribed trajectory,
        and ``frustum_overlap`` reflects how far the orbit strays from the source view.
        ``scene_hints`` may carry optional 3D guidance (target trajectory, point cloud).
        """
        raise NotImplementedError

    # --- Seam B: data amplifier (mono -> pseudo-multi-view) --------------------
    @abstractmethod
    def amplify(
        self,
        clip: ClipRef,
        n_views: int,
        deviation: float,
    ) -> list[SynthViewRef]:
        """Synthesize ``n_views`` extra viewpoints within ``deviation`` of the source.

        Returns `SynthViewRef`s with ``seam=B_AMPLIFY``; these feed reconstruction as
        pseudo-multi-view input (FR-30). ``deviation`` bounds the camera offset to keep
        frustum overlap adequate (R-14).
        """
        raise NotImplementedError

    @abstractmethod
    def inpaint_occlusions(
        self,
        subject_views: Sequence[CropRef],
    ) -> SynthViewRef:
        """Synthesize plausible unseen sides of a subject for the avatar pipeline (FR-31).

        Returns a `SynthViewRef` with ``seam=B_INPAINT``. Plausible, not exact (R-16) —
        callers must not rely on it for analysing critical positions.
        """
        raise NotImplementedError
