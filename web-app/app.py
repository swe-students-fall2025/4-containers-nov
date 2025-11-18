from datetime import datetime

import os
from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from pymongo.errors import PyMongoError


def format_ts(raw):
    """Format timestamp for display in templates"""
    if isinstance(raw, datetime):
        return raw.strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(raw, str):
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return raw

    return raw


app = Flask(__name__)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "handsense")

print("💡 MONGO_URI =", MONGO_URI)

print("🔌 Connecting to MongoDB...")
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    client.server_info()  # Force connection (otherwise lazy connection hides errors)
    print("✅ MongoDB connected")
except PyMongoError as e:
    print("❌ MongoDB connection FAILED:", e)

db = client[MONGO_DB_NAME]
print("📁 DB Loaded:", db)


events = db["gesture_events"]
controls = db["controls"]
print("📁 Collections:", events, controls)


@app.route("/")
def index():
    # recent 50 events
    if hasattr(events, "find"):
        recent_events = list(events.find().sort("timestamp", -1).limit(50))
    else:
        # For testing with fake collections
        recent_events = []

    for e in recent_events:
        e["timestamp_display"] = format_ts(e.get("timestamp"))

    if hasattr(events, "count_documents"):
        total_count = events.count_documents({})
    else:
        total_count = len(recent_events)

    pipeline = [
        {"$group": {"_id": "$gesture", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]

    if hasattr(events, "aggregate"):
        gesture_stats = list(events.aggregate(pipeline))
    else:
        gesture_stats = []

    latest = recent_events[0] if recent_events else None

    return render_template(
        "index.html",
        latest=latest,
        recent_events=recent_events,
        total_count=total_count,
        gesture_stats=gesture_stats,
    )


@app.route("/api/latest")
def get_latest():
    doc = events.find_one(sort=[("timestamp", -1)])  # Latest record

    if not doc:
        return {"gesture": None}

    return {
        "gesture": doc.get("gesture"),
        "confidence": doc.get("confidence"),
        "handedness": doc.get("handedness"),
        "timestamp": doc.get("timestamp"),
    }


@app.route("/api/control", methods=["POST"])
def control_capture():
    """
    Control whether ML client should capture
    Write to the controls collection
    """
    data = request.get_json()
    enabled = data.get("enabled", False)

    controls.update_one(
        {"_id": "capture_control"}, {"$set": {"enabled": enabled}}, upsert=True
    )

    return jsonify({"status": "ok", "enabled": enabled})


@app.route("/api/control/status")
def get_control_status():
    """
    ML client can read this when starting (or directly read Mongo)
    """
    doc = controls.find_one({"_id": "capture_control"})
    if not doc:
        return jsonify({"enabled": False})
    return jsonify({"enabled": doc.get("enabled", False)})


if __name__ == "__main__":
    # For development testing; use gunicorn in production
    app.run(host="0.0.0.0", port=5000)
