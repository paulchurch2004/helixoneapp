# src/api_tracker.py

import os
import json

TRACKER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "api_stats.json")

def _load_stats():
    if os.path.exists(TRACKER_PATH):
        with open(TRACKER_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_stats(data):
    os.makedirs(os.path.dirname(TRACKER_PATH), exist_ok=True)
    with open(TRACKER_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def update_stats(api_name, success):
    stats = _load_stats()
    if api_name not in stats:
        stats[api_name] = {"success": 0, "fail": 0}
    if success:
        stats[api_name]["success"] += 1
    else:
        stats[api_name]["fail"] += 1
    _save_stats(stats)

def get_sorted_sources():
    stats = _load_stats()
    if not stats:
        return ["Yahoo", "Finnhub", "Polygon", "Twelvedata", "Wolfram", "Alphavantage", "Eod"]

    def score(entry):
        total = entry["success"] + entry["fail"]
        return entry["success"] / total if total > 0 else 0

    sorted_sources = sorted(stats.items(), key=lambda x: score(x[1]), reverse=True)
    return [name for name, _ in sorted_sources]
