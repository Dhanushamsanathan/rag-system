"""
Pytest configuration for clean process teardown on macOS with ONNX Runtime.
"""

import os
import pytest


def pytest_sessionfinish(session, exitstatus):
    """Ensures clean exit status without macOS ONNX runtime mutex teardown warning."""
    os._exit(session.exitstatus)
