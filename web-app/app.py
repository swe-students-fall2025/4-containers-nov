from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient

app = Flask(__name__)

MONGO_URI = "mongodb://localhost:27017"
print("💡 MONGO_URI =", MONGO_URI)

print("🔌 Connecting to MongoDB...")
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    client.server_info()  # Force connection (otherwise lazy connection hides errors)
    print("✅ MongoDB connected")
except Exception as e:
    print("❌ MongoDB connection FAILED:", e)

db = client["handsense"]
print("📁 DB Loaded:", db)

events = db["gesture_events"]
controls = db["controls"]
print("📁 Collections:", events, controls)


@app.route("/")
def index():
    return render_template("index.html")


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
