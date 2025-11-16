import sys
from pathlib import Path

import pytest

# Make sure the parent folder (where app.py lives) is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app as web_app  # noqa: E402  pylint: disable=wrong-import-position


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
        # default document used by the web app
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
def app(monkeypatch, fake_events, fake_controls):
    """
    pytest-flask fixture.

    Patch the real Mongo collections in app.py with our fake ones so that
    tests do not require a running MongoDB instance.
    """
    monkeypatch.setattr(web_app, "events", fake_events)
    monkeypatch.setattr(web_app, "controls", fake_controls)
    return web_app.app  # this is the Flask application defined in app.py
