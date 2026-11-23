"""Vercel Serverless Function entry point for TransitPulse-AI API."""

import os
import sys

# Ensure repository root and src directory are on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from api.main import app as base_app  # noqa: E402


class VercelPathMiddleware:
    """ASGI middleware to restore original request path from Vercel edge rewrite headers."""

    def __init__(self, inner_app):
        self.inner_app = inner_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            header_dict = {k.lower(): v for k, v in scope.get("headers", [])}
            # Print available headers to logger / stdout for debugging
            # Check common Vercel original URL headers
            matched_path = (
                header_dict.get(b"x-now-route-matches")
                or header_dict.get(b"x-forwarded-uri")
                or header_dict.get(b"x-original-url")
                or header_dict.get(b"x-invoke-path")
            )
            if matched_path:
                orig_path = matched_path.decode("utf-8", errors="ignore").split("?")[0]
                if orig_path:
                    scope["path"] = orig_path
                    scope["raw_path"] = orig_path.encode("utf-8")
            elif scope.get("path") in ("/api/index.py", "/api/index", "/api"):
                # If no specific header was found and the path is the entrypoint itself, route to root
                scope["path"] = "/"
                scope["raw_path"] = b"/"

        await self.inner_app(scope, receive, send)


app = VercelPathMiddleware(base_app)

__all__ = ["app"]
