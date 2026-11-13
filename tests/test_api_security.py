"""Auth, CORS, and upload-limit checks. No network and no real secrets."""

import os
import unittest
from unittest.mock import patch

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from api_security import (
    cors_middleware_kwargs,
    max_upload_bytes,
    read_limited_upload,
    require_api_key,
    safe_upload_filename,
    tokens_match,
)


class TokensMatchTests(unittest.TestCase):
    def test_matches_equal_tokens(self):
        self.assertTrue(tokens_match("local-test-key", "local-test-key"))

    def test_rejects_different_lengths(self):
        self.assertFalse(tokens_match("short", "local-test-key"))


class CorsTests(unittest.TestCase):
    def test_wildcard_is_dropped_and_credentials_stay_off(self):
        with patch.dict(
            os.environ,
            {"CORS_ALLOWED_ORIGINS": "*, https://portfolio.example"},
            clear=False,
        ):
            settings = cors_middleware_kwargs()
        self.assertEqual(settings["allow_origins"], ["https://portfolio.example"])
        self.assertFalse(settings["allow_credentials"])
        self.assertNotIn("*", settings["allow_origins"])

    def test_empty_allowlist(self):
        with patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": "  , * "}, clear=False):
            self.assertEqual(cors_middleware_kwargs()["allow_origins"], [])


class UploadPolicyTests(unittest.TestCase):
    def test_strips_directories_and_rejects_unknown_types(self):
        self.assertEqual(safe_upload_filename("../../notes.md"), "notes.md")
        self.assertEqual(safe_upload_filename("Q4 Report.PDF"), "Q4_Report.pdf")
        with self.assertRaises(Exception) as raised:
            safe_upload_filename("malware.exe")
        self.assertEqual(raised.exception.status_code, 400)

    def test_size_limit_defaults_and_rejects_invalid_values(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAX_UPLOAD_BYTES", None)
            self.assertEqual(max_upload_bytes(), 20 * 1024 * 1024)
        with patch.dict(os.environ, {"MAX_UPLOAD_BYTES": "nope"}, clear=False):
            self.assertEqual(max_upload_bytes(), 20 * 1024 * 1024)


class _FakeUpload:
    def __init__(self, data: bytes):
        self._data = data

    async def read(self, size: int = -1) -> bytes:
        if size < 0:
            return self._data
        return self._data[:size]


class ReadLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_oversize_payload(self):
        with patch.dict(os.environ, {"MAX_UPLOAD_BYTES": "4"}, clear=False):
            with self.assertRaises(Exception) as raised:
                await read_limited_upload(_FakeUpload(b"12345"))
        self.assertEqual(raised.exception.status_code, 413)


def _client() -> TestClient:
    app = FastAPI()

    @app.post("/upload", dependencies=[Depends(require_api_key)])
    def upload():
        return {"ok": True}

    @app.post("/query", dependencies=[Depends(require_api_key)])
    def query():
        return {"ok": True}

    return TestClient(app)


class AuthTests(unittest.TestCase):
    def test_fails_closed_when_key_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_API_KEY", None)
            client = _client()
            self.assertEqual(client.post("/upload").status_code, 503)
            self.assertEqual(client.post("/query").status_code, 503)

    def test_rejects_missing_and_wrong_token(self):
        with patch.dict(os.environ, {"RAG_API_KEY": "local-test-key"}, clear=False):
            client = _client()
            self.assertEqual(client.post("/upload").status_code, 401)
            wrong = client.post(
                "/query", headers={"Authorization": "Bearer wrong-key"}
            )
            self.assertEqual(wrong.status_code, 401)

    def test_accepts_matching_bearer_token(self):
        with patch.dict(os.environ, {"RAG_API_KEY": "local-test-key"}, clear=False):
            client = _client()
            response = client.post(
                "/upload", headers={"Authorization": "Bearer local-test-key"}
            )
            self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
