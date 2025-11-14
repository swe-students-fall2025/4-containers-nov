from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb://localhost:27017")

db = client["handsense"]
collection = db["gesture_events"]

event = {
    "timestamp": datetime.utcnow().isoformat(),
    "gesture": "test_event",
    "confidence": 0.99,
}

collection.insert_one(event)
print("Inserted:", event)
