#!/usr/bin/env python3
"""Gate: a change may not add ruff violations to the files it touches.

The repo carries a lint backlog it is deliberately not fixing in one go (docs/STATUS.md
§5), so a plain `ruff check` on a touched file fails on debt somebody else wrote — which
would mean every commit has to clean up unrelated code first. Instead each changed file
is compared against its own previous version. The count may fall or stay put; it may not
rise. The backlog can only shrink.

This is the check CLAUDE.md has always described in prose ("lint only the lines you
changed"), made executable so the hook, CI and a human all get the same verdict.

    python scripts/lint_changed.py               # staged vs HEAD  (pre-commit)
    python scripts/lint_changed.py --base main   # HEAD vs a ref   (CI)
"""

from __future__ import annotations

import argparse
import subprocess
import sys

RUFF = ["ruff", "check", "--no-fix", "--output-format=concise"]


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True)


def _ruff(path: str, content: str) -> list[str]:
    """Violations ruff reports for `content`, attributed to `path`."""
    proc = subprocess.run(
        [*RUFF, "--stdin-filename", path, "-"],
        input=content,
        capture_output=True,
        text=True,
    )
    # 0 = clean, 1 = violations found. Anything else is ruff itself failing.
    if proc.returncode not in (0, 1):
        sys.exit(f"ruff failed on {path}:\n{proc.stderr}")
    return [ln for ln in proc.stdout.splitlines() if ln.startswith(f"{path}:")]


def _blob(ref: str, path: str) -> str | None:
    """File content at a ref, or None if it did not exist there."""
    proc = _git("show", f"{ref}:{path}")
    return proc.stdout if proc.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", help="compare HEAD against this ref (CI). Default: staged vs HEAD.")
    args = ap.parse_args()

    if args.base:
        old_ref, new_ref = args.base, "HEAD"
        listing = _git("diff", "--name-only", "--diff-filter=ACMR", args.base, "HEAD", "--", "*.py")
    else:
        old_ref, new_ref = "HEAD", ""  # "" is the index
        listing = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "--", "*.py")

    if listing.returncode != 0:
        sys.exit(f"git diff failed:\n{listing.stderr}")

    paths = [p for p in listing.stdout.split("\n") if p.strip()]
    if not paths:
        print("lint-changed: no Python files changed")
        return 0

    regressions: list[tuple[str, int, int, list[str]]] = []
    improved = 0

    for path in paths:
        new = _blob(new_ref, path)
        if new is None:
            continue  # vanished between the diff and now
        before = _blob(old_ref, path)
        old_hits = [] if before is None else _ruff(path, before)
        new_hits = _ruff(path, new)
        if len(new_hits) > len(old_hits):
            regressions.append((path, len(old_hits), len(new_hits), new_hits))
        elif len(new_hits) < len(old_hits):
            improved += 1

    if not regressions:
        print(f"lint-changed: {len(paths)} file(s) checked, no new violations", end="")
        print(f", {improved} improved" if improved else "")
        return 0

    print(f"lint-changed: {len(regressions)} file(s) gained ruff violations\n")
    for path, old, new, hits in regressions:
        print(f"  {path}: {old} -> {new}")
        for hit in hits:
            print(f"      {hit}")
        print()
    print("Fix the violations you introduced. Pre-existing ones in these files are not your")
    print("problem — only the increase is.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
