"""Basic integration tests for the Flask web app."""

from datetime import datetime

# pylint: disable=import-error
from app import format_ts


def test_index_page_renders(client):
    """Root path should return HTML page."""
    response = client.get("/")
    assert response.status_code == 200
    # We don't check exact HTML, only that it's not empty
    assert b"<html" in response.data or b"<!DOCTYPE html" in response.data


def test_latest_no_events_returns_none(client):
    """When there are no gesture events, API should return gesture = None."""
    response = client.get("/api/latest")
    assert response.status_code == 200
    data = response.get_json()
    assert data["gesture"] is None


def test_latest_with_event_returns_latest(client, fake_events):
    """When events exist, /api/latest should return the last inserted one."""
    # Insert a fake event document
    fake_events.docs.append(
        {
            "gesture": "palm",
            "confidence": 0.95,
            "handedness": "Right",
            "timestamp": "2025-11-16T18:00:00Z",
        }
    )

    response = client.get("/api/latest")
    assert response.status_code == 200
    data = response.get_json()

    assert data["gesture"] == "palm"
    assert data["confidence"] == 0.95
    assert data["handedness"] == "Right"
    assert data["timestamp"] == "2025-11-16T18:00:00Z"


# pylint: disable=unused-argument
def test_control_toggle_updates_status(client, fake_controls):
    """
    POST /api/control should update the capture_control document, and
    /api/control/status should reflect the change.
    """
    assert fake_controls is not None

    # Default is False
    status_before = client.get("/api/control/status").get_json()
    assert status_before["enabled"] is False

    # Turn capture ON
    response = client.post("/api/control", json={"enabled": True})
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["enabled"] is True

    # Now /api/control/status should return enabled = True
    status_after = client.get("/api/control/status").get_json()
    assert status_after["enabled"] is True


def test_format_ts_with_datetime():
    """Datetime object should be formatted correctly."""
    dt = datetime(2025, 1, 1, 10, 30, 0)
    formatted = format_ts(dt)
    assert formatted == "2025-01-01 10:30:00"


def test_format_ts_with_iso_string():
    """ISO timestamp string should be parsed and formatted correctly."""
    raw = "2025-01-01T15:45:30"
    formatted = format_ts(raw)
    assert formatted == "2025-01-01 15:45:30"


def test_format_ts_bad_string_returns_raw():
    """Invalid timestamp string should be returned unchanged."""
    raw = "not-a-timestamp"
    assert format_ts(raw) == raw


def test_latest_full_no_events_returns_exists_false(client, fake_events):
    """When no events exist, /api/latest_full should report exists=False."""
    fake_events.docs.clear()

    response = client.get("/api/latest_full")
    assert response.status_code == 200
    data = response.get_json()

    assert data["exists"] is False


def test_latest_full_with_event(client, fake_events):
    """When events exist, /api/latest_full should return formatted timestamp etc."""
    fake_events.docs.append(
        {
            "gesture": "fist",
            "confidence": 0.88,
            "handedness": "Left",
            "timestamp": "2025-11-16T20:00:00Z",
        }
    )

    response = client.get("/api/latest_full")
    assert response.status_code == 200
    data = response.get_json()

    assert data["exists"] is True
    assert data["gesture"] == "fist"
    assert data["confidence"] == 0.88
    assert data["handedness"] == "Left"
    assert data["timestamp_display"].startswith("2025-11-16")


# def test_format_ts_with_none_returns_none():
#     """Non-datetime / non-string values should be returned as-is."""
#     from app import format_ts

#     assert format_ts(None) is None
#     assert format_ts(123) == 123
