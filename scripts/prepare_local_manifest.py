"""Configure the four historical experiment clips on a teammate's computer."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2

FILES = {"L002": "2.mp4", "L003": "3.mp4", "L009": "Video Project 10.mp4", "L010": "Video Project 11.mp4"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", default="data/review/local_video_manifest.json")
    args = parser.parse_args()
    folder, output = Path(args.video_dir).resolve(), Path(args.output)
    if output.exists():
        parser.error("Manifest exists: preserve it and choose another --output")
    rows = []
    for clip_id, filename in FILES.items():
        path = folder / filename
        if not path.is_file():
            parser.error(f"Missing original experiment clip: {path}")
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                parser.error(f"Cannot open {path}")
            fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or frames <= 0:
                parser.error(f"Invalid constant-FPS metadata: {path}")
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        rows.append(dict(id=clip_id, filename=filename, path=str(path), fps=fps, frames=frames,
                         width=width, height=height, duration_s=frames / fps, sha256=digest.hexdigest(),
                         event_group="local-shared-corridor-scene-A" if clip_id in {"L002", "L003"} else f"unverified-{clip_id}",
                         rights_status="Source-specific permission still needs verification", rights_approved=False,
                         training_approved=False, risk_label=None))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(source_directory=str(folder), count=len(rows), videos=rows,
                                     warning="Historical experiment mapping, not permission or label approval"), indent=2), encoding="utf-8")
    print(f"Configured four clips in {output}; no rights or risk labels approved")


if __name__ == "__main__":
    main()
