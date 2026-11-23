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

from api.main import app  # noqa: E402

__all__ = ["app"]
