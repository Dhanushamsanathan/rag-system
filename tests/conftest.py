"""
Pytest configuration and shared fixtures.
Handles clean process teardown on macOS with ONNX Runtime mutex issue.
"""

import sys
import os
import pytest


def pytest_unconfigure(config):
    """
    Graceful teardown after all reporting and cleanups have completed.
    Only applies the forced exit on macOS where ONNX Runtime triggers
    a recursive_mutex lock failure during interpreter shutdown.
    """
    if sys.platform == "darwin":
        sys.stdout.flush()
        sys.stderr.flush()
        exit_code = getattr(config, "_exitstatus", 0)
        os._exit(exit_code)


def pytest_sessionfinish(session, exitstatus):
    """Store exit status for use in pytest_unconfigure."""
    session.config._exitstatus = exitstatus
