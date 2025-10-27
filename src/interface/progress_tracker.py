import json
import os

PROGRESS_PATH = os.path.join("data", "progression.json")

def load_progress():
    if not os.path.exists(PROGRESS_PATH):
        return {}
    try:
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_progress(progress):
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)

def mark_module_complete(module_id):
    progress = load_progress()
    progress[module_id] = True
    save_progress(progress)

def is_module_completed(module_id):
    progress = load_progress()
    return progress.get(module_id, False)

def get_progress_stats(module_ids):
    progress = load_progress()
    total = len(module_ids)
    completed = sum(1 for mid in module_ids if progress.get(mid))
    return completed, total, (completed / total * 100) if total > 0 else 0
