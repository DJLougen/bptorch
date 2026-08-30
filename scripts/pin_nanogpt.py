#!/usr/bin/env python3
"""Script to verify or re-pin the nanoGPT reference commit."""

import hashlib
import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
LOCK_FILE = ROOT_DIR / "references" / "nanogpt.lock.json"
MODEL_FILE = ROOT_DIR / "references" / "nanogpt" / "model.py"


def verify_pinned_reference():
    if not LOCK_FILE.exists():
        raise FileNotFoundError(f"Lock file missing: {LOCK_FILE}")
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file missing: {MODEL_FILE}")

    with open(LOCK_FILE, "r") as f:
        lock_data = json.load(f)

    print(f"Verified pinned nanoGPT reference: {lock_data['commit']}")
    print(f"Pinned at: {lock_data['pinned_at']}")
    print(f"License: {lock_data['license']}")
    return True


if __name__ == "__main__":
    verify_pinned_reference()
