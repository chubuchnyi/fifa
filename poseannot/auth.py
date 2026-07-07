"""JWT + bcrypt auth for poseannot.

User registry lives in a YAML file (``users_yaml`` config field). Passwords
stored as bcrypt hashes. Login yields a JWT with ``exp`` claim; all
``/api/*`` routes require the JWT in either the ``Authorization: Bearer …``
header or a ``poseannot_token`` cookie (browser flow).

CLI:
    .venv/bin/python -m poseannot.auth hash <plaintext>
        prints a bcrypt hash to paste into users.yaml
    .venv/bin/python -m poseannot.auth check <plaintext> <hash>
        confirms a plaintext matches a hash
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import bcrypt
import yaml
from fastapi import Cookie, Depends, HTTPException, Request, status
from jose import JWTError, jwt

from .config import PoseAnnotConfig, load as load_config

JWT_ALG = "HS256"


def _users_from_yaml(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {u["username"]: u["password_hash"] for u in raw.get("users", [])}


def verify_password(plain: str, hashed: str) -> bool:
    # bcrypt caps at 72 bytes — passwords longer than that are truncated (documented).
    plain_bytes = plain.encode("utf-8")[:72]
    return bcrypt.checkpw(plain_bytes, hashed.encode("utf-8"))


def hash_password(plain: str) -> str:
    plain_bytes = plain.encode("utf-8")[:72]
    return bcrypt.hashpw(plain_bytes, bcrypt.gensalt()).decode("utf-8")


def issue_token(username: str, cfg: PoseAnnotConfig) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=cfg.jwt_expire_hours)
    return jwt.encode(
        {"sub": username, "exp": exp}, cfg.jwt_secret, algorithm=JWT_ALG,
    )


def _decode(token: str, cfg: PoseAnnotConfig) -> str:
    try:
        payload = jwt.decode(token, cfg.jwt_secret, algorithms=[JWT_ALG])
        return payload["sub"]
    except (JWTError, KeyError) as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid token: {e}") from e


def current_user(
    request: Request,
    poseannot_token: str | None = Cookie(default=None),
) -> str:
    """FastAPI dependency — resolve the caller's username or 401."""
    cfg = load_config()
    # Bearer header wins over cookie (API clients + browser both work)
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return _decode(auth[7:], cfg)
    if poseannot_token:
        return _decode(poseannot_token, cfg)
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "no token")


def authenticate(username: str, password: str) -> bool:
    cfg = load_config()
    users = _users_from_yaml(cfg.users_yaml)
    if username not in users:
        return False
    return verify_password(password, users[username])


def _cli(argv: Iterable[str]) -> int:
    args = list(argv)
    if not args:
        print(__doc__.strip())
        return 1
    cmd = args[0]
    if cmd == "hash" and len(args) == 2:
        print(hash_password(args[1]))
        return 0
    if cmd == "check" and len(args) == 3:
        ok = verify_password(args[1], args[2])
        print("OK" if ok else "FAIL")
        return 0 if ok else 1
    print(__doc__.strip())
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
