from __future__ import annotations

import hmac
import time
from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ai_gateway.config import Settings

_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True, slots=True)
class AdminIdentity:
    username: str
    auth_type: str


def verify_admin_credentials(username: str, password: str, settings: Settings) -> bool:
    user_ok = hmac.compare_digest(username.encode("utf-8"), settings.admin_username.encode("utf-8"))
    pass_ok = hmac.compare_digest(password.encode("utf-8"), settings.admin_password.encode("utf-8"))
    return user_ok and pass_ok


def create_access_token(username: str, secret: str, ttl_seconds: int) -> str:
    now = int(time.time())
    payload = {
        "sub": username,
        "role": "admin",
        "iat": now,
        "exp": now + ttl_seconds,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def verify_access_token(token: str, secret: str) -> str:
    payload = jwt.decode(token, secret, algorithms=["HS256"])
    if payload.get("role") != "admin" or not payload.get("sub"):
        raise jwt.InvalidTokenError("Invalid admin token payload")
    return str(payload["sub"])


async def require_admin(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AdminIdentity:
    settings: Settings = request.app.state.settings
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials.strip()
    if settings.admin_api_key and hmac.compare_digest(
        token.encode("utf-8"),
        settings.admin_api_key.encode("utf-8"),
    ):
        return AdminIdentity(username=settings.admin_username, auth_type="api_key")

    try:
        username = verify_access_token(token, settings.jwt_secret)
        return AdminIdentity(username=username, auth_type="jwt")
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc