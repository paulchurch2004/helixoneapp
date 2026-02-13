"""
Basic analytics for HelixOne
Tracks app usage without collecting personal data
"""

import hashlib
import json
import os
import platform
import uuid
from datetime import datetime
from pathlib import Path

ANALYTICS_ENABLED = not os.getenv("HELIXONE_DEV") and not os.getenv("HELIXONE_NO_ANALYTICS")
ANALYTICS_FILE = Path.home() / ".helixone_analytics.json"


def get_device_id() -> str:
    """
    Get or create a unique device ID (anonymous)
    """
    analytics_data = _load_analytics_data()

    if "device_id" in analytics_data:
        return analytics_data["device_id"]

    # Generate a new device ID
    device_id = str(uuid.uuid4())
    analytics_data["device_id"] = device_id
    _save_analytics_data(analytics_data)

    return device_id


def _load_analytics_data() -> dict:
    """Load analytics data from file"""
    if ANALYTICS_FILE.exists():
        try:
            with open(ANALYTICS_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_analytics_data(data: dict):
    """Save analytics data to file"""
    try:
        with open(ANALYTICS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[Analytics] Failed to save data: {e}")


def track_event(
    event_name: str,
    properties: dict | None = None,
    user_id: str | None = None,
):
    """
    Track an event

    Args:
        event_name: Name of the event (e.g., "app_started", "module_completed")
        properties: Additional properties (no PII)
        user_id: User ID (will be hashed for privacy)
    """
    if not ANALYTICS_ENABLED:
        return

    try:
        analytics_data = _load_analytics_data()

        # Initialize events list if needed
        if "events" not in analytics_data:
            analytics_data["events"] = []

        # Create event
        event = {
            "name": event_name,
            "timestamp": datetime.now().isoformat(),
            "device_id": get_device_id(),
            "platform": platform.system(),
            "platform_version": platform.release(),
        }

        # Add hashed user_id if provided
        if user_id:
            event["user_hash"] = hashlib.sha256(user_id.encode()).hexdigest()[:16]

        # Add properties if provided
        if properties:
            # Filter out any potential PII
            safe_properties = {
                k: v
                for k, v in properties.items()
                if k not in ["email", "password", "token", "name", "phone"]
            }
            event["properties"] = safe_properties

        # Add to events list
        analytics_data["events"].append(event)

        # Keep only last 100 events to avoid file bloat
        if len(analytics_data["events"]) > 100:
            analytics_data["events"] = analytics_data["events"][-100:]

        # Update counters
        if "counters" not in analytics_data:
            analytics_data["counters"] = {}

        counter_key = f"event_{event_name}"
        analytics_data["counters"][counter_key] = analytics_data["counters"].get(counter_key, 0) + 1

        # Save
        _save_analytics_data(analytics_data)

    except Exception as e:
        print(f"[Analytics] Failed to track event: {e}")


def track_app_start(version: str):
    """Track app start event"""
    track_event("app_started", {"version": version})


def track_app_crash(error_type: str, error_message: str):
    """Track app crash"""
    track_event(
        "app_crashed",
        {
            "error_type": error_type,
            "error_message": error_message[:200],  # Truncate long messages
        },
    )


def track_feature_usage(feature_name: str, action: str = "used"):
    """Track feature usage"""
    track_event(
        "feature_usage",
        {
            "feature": feature_name,
            "action": action,
        },
    )


def track_module_completed(module_id: str, duration_seconds: int | None = None):
    """Track training module completion"""
    properties = {"module_id": module_id}
    if duration_seconds:
        properties["duration"] = duration_seconds

    track_event("module_completed", properties)


def get_analytics_summary() -> dict:
    """Get analytics summary (for debugging/display)"""
    analytics_data = _load_analytics_data()
    return {
        "device_id": analytics_data.get("device_id", "N/A"),
        "total_events": len(analytics_data.get("events", [])),
        "counters": analytics_data.get("counters", {}),
        "first_event": (
            analytics_data.get("events", [{}])[0].get("timestamp", "N/A")
            if analytics_data.get("events")
            else "N/A"
        ),
        "last_event": (
            analytics_data.get("events", [{}])[-1].get("timestamp", "N/A")
            if analytics_data.get("events")
            else "N/A"
        ),
    }


def clear_analytics_data():
    """Clear all analytics data (for privacy/debugging)"""
    try:
        if ANALYTICS_FILE.exists():
            ANALYTICS_FILE.unlink()
        print("[Analytics] Data cleared")
    except Exception as e:
        print(f"[Analytics] Failed to clear data: {e}")


# Example usage:
if __name__ == "__main__":
    print("Analytics Summary:")
    print(json.dumps(get_analytics_summary(), indent=2))
