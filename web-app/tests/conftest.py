import sys
from pathlib import Path

import pytest

# This file is only used in tests; in CI pylint cannot resolve dynamic imports
# for app.py, so we ignore import-error here.
# pylint: disable=import-error

# Make sure the parent folder (where app.py lives) is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import the whole app module so we can patch its globals (events, controls)
import app as app_module  # noqa: E402  pylint: disable=wrong-import-position


class FakeEventsCollection:
    """Very small fake collection for gesture_events."""

    def __init__(self):
        self.docs = []

    def find_one(self, sort=None):  # pylint: disable=unused-argument
        """Return the latest doc (we just use the last pushed)."""
        if not self.docs:
            return None
        return self.docs[-1]


class FakeControlsCollection:
    """Fake collection for controls (capture_control document)."""

    def __init__(self):
        # Default document used by the web app
        self.doc = {"_id": "capture_control", "enabled": False}

    def update_one(self, _filter, update, upsert=False):  # pylint: disable=unused-argument
        """Update the enabled flag based on the request body."""
        if "$set" in update:
            self.doc.update(update["$set"])

    def find_one(self, _filter):  # pylint: disable=unused-argument
        """Return the single control document."""
        return self.doc


@pytest.fixture
def fake_events():
    """Provide a fresh fake events collection for each test."""
    return FakeEventsCollection()


@pytest.fixture
def fake_controls():
    """Provide a fresh fake controls collection for each test."""
    return FakeControlsCollection()


@pytest.fixture
def app(monkeypatch, fake_events, fake_controls):  # pylint: disable=redefined-outer-name
    """
    pytest-flask fixture.

    Patch the real Mongo collections in app.py with our fake ones so that
    tests do not require a running MongoDB instance.
    """
    # Patch the module-level globals used by the Flask routes
    monkeypatch.setattr(app_module, "events", fake_events)
    monkeypatch.setattr(app_module, "controls", fake_controls)

    # Return the Flask app object for pytest-flask
    return app_module.app
