# camlab — moved

The camera-calibration prototype has its own repository now, and its own documentation with it.
It installs, tests, runs and deploys without this one.

| | |
|---|---|
| repo | `/home/chubuchnyi/camlab` |
| spec | `camlab/docs/spec.md` |
| the measurement it rests on | `camlab/docs/findings/m1-fixed-centre.md` |
| what it took from here, and what it re-verified | `camlab/docs/inherited-claims.md` |

**One thing from over there that is about this repo.** camlab treats nothing inherited from pitch3d
as true until it re-measures it against pixels, and its register lists six pitch3d claims that did
not survive — including the fan clip's headline 12 382 px, which was computed in the wrong image
space, and `--camera-carry` being "off by default", which it is not. Those are recorded in
[`findings/landmines.md`](findings/landmines.md); the register is the wider list.

**What still lives here and is camlab's:** `scripts/probe_handheld_centre.py`, the M-1 probe. It
drives `fit_rigid_camera` and the paint masks from this repo, so it cannot move until camlab ports
the fit — which is camlab's M2, and M2 needs it anyway.
