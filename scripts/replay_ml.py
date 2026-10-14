"""Replay cached GPU/quality features into the real fog ML endpoint."""
import argparse
import json
from pathlib import Path
import time

import httpx
from fog.anomaly import AnomalyWindow, NAMES


def windows(path, clip=None):
    for row in json.loads(Path(path).read_text(encoding="utf-8")):
        if clip and row["clip_id"] != clip:
            continue
        yield AnomalyWindow(zone_id="ml-" + row["clip_id"], clip_id=row["clip_id"],
                            start_s=row["start_s"], end_s=row["end_s"],
                            feature_names=NAMES, features=row["joint"],
                            quality_eligible=not row["quality_warning"],
                            quality_reason="Offline numerical quality gate; not independently validated" if not row["quality_warning"] else "Unreliable camera/lighting/window estimates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="outputs/training/quality_motion_2026_10_14_run04/features_quality.json")
    parser.add_argument("--fog-url", default="http://127.0.0.1:8000")
    parser.add_argument("--clip")
    parser.add_argument("--interval", type=float, default=1)
    args = parser.parse_args()
    if args.interval < 0:
        parser.error("interval must be nonnegative")
    rows = list(windows(args.features, args.clip))
    if not rows:
        parser.error("No matching windows")
    with httpx.Client(timeout=10) as client:
        for row in rows:
            response = client.post(args.fog_url.rstrip("/") + "/api/ml/windows", json=row.model_dump())
            response.raise_for_status()
            result = response.json()
            print(f"{row.clip_id} {row.start_s:.2f}s: {result['observation']} score={result['score']}", flush=True)
            if args.interval:
                time.sleep(args.interval)


if __name__ == "__main__":
    main()
