"""Bearer-token checks, CORS allowlist, and upload limits.

The hybrid server and the knowledge-base API both call into this module.
Authentication fails closed when RAG_API_KEY is unset. CORS never uses a
wildcard origin and never enables credentialed cross-origin requests.
"""

import hashlib
import hmac
import os
import re
from pathlib import Path

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

API_KEY_ENV = "RAG_API_KEY"
CORS_ORIGINS_ENV = "CORS_ALLOWED_ORIGINS"
MAX_UPLOAD_BYTES_ENV = "MAX_UPLOAD_BYTES"
DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".docx",
        ".xlsx",
        ".pptx",
        ".txt",
        ".md",
        ".rtf",
        ".html",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }
)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_bearer = HTTPBearer(auto_error=False)


def tokens_match(provided: str, expected: str) -> bool:
    """Compare tokens without leaking the expected value or its length."""
    provided_digest = hashlib.sha256(provided.encode("utf-8")).digest()
    expected_digest = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(provided_digest, expected_digest)


async def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Require Authorization: Bearer <RAG_API_KEY>. Fail closed if unset."""
    expected = os.getenv(API_KEY_ENV, "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured",
        )
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if not tokens_match(credentials.credentials, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


def cors_allowlist() -> list[str]:
    """Parse CORS_ALLOWED_ORIGINS. Drop blanks and the wildcard origin."""
    raw = os.getenv(CORS_ORIGINS_ENV, "")
    origins: list[str] = []
    for part in raw.split(","):
        origin = part.strip()
        if not origin or origin == "*":
            continue
        origins.append(origin)
    return origins


def cors_middleware_kwargs() -> dict:
    """CORS settings for browser clients using the bearer token.

    allow_credentials stays false so a wildcard can never be combined with
    credentialed responses. Origins come only from the allowlist.
    """
    return {
        "allow_origins": cors_allowlist(),
        "allow_credentials": False,
        "allow_methods": ["GET", "POST", "DELETE", "OPTIONS"],
        "allow_headers": ["Authorization", "Content-Type"],
    }


def max_upload_bytes() -> int:
    raw = os.getenv(MAX_UPLOAD_BYTES_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_UPLOAD_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_UPLOAD_BYTES
    if value <= 0:
        return DEFAULT_MAX_UPLOAD_BYTES
    return value


def safe_upload_filename(original_filename: str | None) -> str:
    """Return a single-path-segment filename with an allowed extension."""
    if not original_filename or "\x00" in original_filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    basename = Path(original_filename.replace("\\", "/")).name
    if basename in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")

    extension = Path(basename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not allowed")

    stem = _SAFE_NAME.sub("_", Path(basename).stem).strip("._")
    if not stem:
        stem = "upload"
    return f"{stem[:80]}{extension}"


async def read_limited_upload(upload) -> bytes:
    """Read an upload, rejecting empty files and files over the size cap."""
    limit = max_upload_bytes()
    content = await upload.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(status_code=413, detail="File exceeds upload size limit")
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    return content
