"""Vercel Serverless Function entry point for TransitPulse-AI API."""

import os
import sys
from urllib.parse import parse_qsl, urlencode

# Ensure repository root and src directory are on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from api.main import app as base_app  # noqa: E402


class VercelPathMiddleware:
    """ASGI middleware to restore original request path from Vercel rewrite parameter."""

    def __init__(self, inner_app):
        self.inner_app = inner_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            qs = scope.get("query_string", b"").decode("utf-8", errors="ignore")
            if "_vpath" in qs:
                params = parse_qsl(qs, keep_blank_values=True)
                vpath = None
                remaining = []
                for k, v in params:
                    if k == "_vpath":
                        vpath = v
                    else:
                        remaining.append((k, v))

                if vpath is not None:
                    clean_path = "/" + vpath.lstrip("/")
                    scope["path"] = clean_path
                    scope["raw_path"] = clean_path.encode("utf-8")
                    scope["query_string"] = urlencode(remaining).encode("utf-8")
            elif scope.get("path") in ("/api/index.py", "/api/index", "/api"):
                scope["path"] = "/"
                scope["raw_path"] = b"/"

        await self.inner_app(scope, receive, send)


app = VercelPathMiddleware(base_app)

__all__ = ["app"]
