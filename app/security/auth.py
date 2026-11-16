"""Cognito JWT validation and stable user identity extraction."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from functools import lru_cache

import jwt
from fastapi import HTTPException
from jwt import PyJWKClient

from app.config import settings


@dataclass(frozen=True)
class Identity:
    subject: str
    provider: str
    groups: tuple[str, ...] = ()


@lru_cache(maxsize=8)
def _jwks_client(issuer: str) -> PyJWKClient:
    return PyJWKClient(
        f"{issuer}/.well-known/jwks.json",
        cache_keys=True,
        lifespan=3600,
        timeout=5,
    )


def authenticate_request(authorization: str | None) -> Identity:
    """Authenticate a request; Cognito ID tokens are required for public mode."""
    mode = settings.AUTH_MODE
    if mode == "none":
        # Backward-compatible shared token for private, single-user staging.
        if settings.API_AUTH_TOKEN:
            return _authenticate_shared_token(authorization)
        return Identity(subject="local", provider="local")

    if mode == "shared_token":
        return _authenticate_shared_token(authorization)

    if mode != "cognito":
        raise HTTPException(status_code=503, detail="Authentication is misconfigured.")
    if not settings.COGNITO_ISSUER or not settings.COGNITO_CLIENT_ID:
        raise HTTPException(status_code=503, detail="Cognito authentication is not configured.")

    token = _bearer_token(authorization)
    try:
        signing_key = _jwks_client(settings.COGNITO_ISSUER).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=settings.COGNITO_CLIENT_ID,
            issuer=settings.COGNITO_ISSUER,
            options={"require": ["exp", "iat", "sub", "token_use"]},
        )
        if claims.get("token_use") != "id":
            raise jwt.InvalidTokenError("Cognito ID token required.")
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject.strip():
            raise jwt.InvalidTokenError("Token subject is missing.")
        raw_groups = claims.get("cognito:groups", [])
        if not isinstance(raw_groups, list) or any(not isinstance(group, str) for group in raw_groups):
            raise jwt.InvalidTokenError("Cognito group claim is invalid.")
        groups = tuple(raw_groups)
        required_group = settings.COGNITO_REQUIRED_GROUP
        if required_group and required_group not in groups:
            raise HTTPException(status_code=403, detail="Access to this application is not authorized.")
        return Identity(subject=subject, provider="cognito", groups=groups)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired identity token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def _bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="A bearer identity token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization[7:].strip()
    if not token or len(token) > 16_384:
        raise HTTPException(status_code=401, detail="Invalid bearer identity token.")
    return token


def _authenticate_shared_token(authorization: str | None) -> Identity:
    token = _bearer_token(authorization)
    if not settings.API_AUTH_TOKEN or not hmac.compare_digest(token, settings.API_AUTH_TOKEN):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Identity(subject="shared-token", provider="shared_token")


def identity_scope(subject: str) -> str:
    """Return a non-reversible stable identity namespace for stored threads."""
    return hashlib.sha256(subject.encode("utf-8")).hexdigest()[:32]


def scoped_thread_id(identity: Identity, thread_id: str) -> str:
    """Prevent callers from reading another identity's checkpoint thread."""
    return f"{identity_scope(identity.subject)}:{thread_id}"
