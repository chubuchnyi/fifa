# playerlab — moved

The identity-and-placement lab has its own repository now, and its documentation with it. It
installs, tests and runs without this one (ADR-0013).

| | |
|---|---|
| repo | `/home/chubuchnyi/playerlab` |
| the question, for someone who has not seen it | `playerlab/docs/PROBLEM.md` |
| the founding decision + inherited-claims register | `playerlab/docs/spec.md` |
| what is true now, and the ordered work list | `playerlab/docs/STATUS.md` |
| traps | `playerlab/docs/findings/landmines.md` |

**One thing from over there that is about this repo.** playerlab's first measurement closed the
boundary question ADR-0013 left open, and it lands on *this* codebase:
`adapters/models/pose.py:353` `_ground_root` sets the world root Z **to** `pelvis_above_foot` —
the pelvis's height above its **own** foot — which pins the foot to z = 0. A player leaving the
ground is therefore **structurally unrepresentable**, and his airborne bbox bottom projected
through the ground plane also drifts him away from the camera. Both halves of the eye's 2026-08-13
report on the header at f55, from one line. `root_z_source: measured` says the pose backend
reported the number; it does not say the world height was measured.
See `playerlab/docs/findings/the-jump-is-structurally-impossible-2026-08-13.md`, and the landmine
in [`findings/landmines.md`](findings/landmines.md).

**Still here and unmoved:** `docs/playerlab-problem.md` (working copy — the authority is
`playerlab/docs/PROBLEM.md`), and every file listed under "Where the code is" in it. Nothing has
been copied out yet; when it is, it goes with an origin header and does not get hand-synced back.
