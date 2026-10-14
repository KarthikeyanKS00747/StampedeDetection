"""Create a local annotation package without approving any labels or rights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from training.motion_baseline import MOTION_NAMES
from training.gpu_person_baseline import PERSON_NAMES


def build(root: Path, quality_dir: Path):
    features = json.loads((quality_dir / "features_quality.json").read_text(encoding="utf-8"))
    proposals = json.loads((quality_dir / "review_proposals.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "data/review/local_video_manifest.json").read_text(encoding="utf-8"))
    catalogue = json.loads((root / "data/review/video_candidates_50.json").read_text(encoding="utf-8"))
    clips = {r["id"]: {"path": r["path"], "filename": r["filename"], "rights_status": r["rights_status"]}
             for r in manifest["videos"] if r["id"] in {r["clip_id"] for r in features}}
    windows = []
    for feature, proposal in zip(features, proposals):
        key = f"{feature['clip_id']}:{feature['start_s']:.6f}:{feature['end_s']:.6f}:{feature['segment']}"
        if feature["clip_id"] != proposal["clip_id"] or feature["start_s"] != proposal["start_s"]:
            raise ValueError("Feature and proposal timestamps differ")
        windows.append({"id": hashlib.sha256(key.encode()).hexdigest()[:16],
                        "clip_id": feature["clip_id"], "event_group": feature["event_group"],
                        "start_s": feature["start_s"], "end_s": feature["end_s"], "segment": feature["segment"],
                        "quality_eligible": not feature["quality_warning"], "features": feature["joint"],
                        "proposal": proposal["proposed_observation"]})
    if len(features) != len(proposals):
        raise ValueError("Missing proposals/features")
    return {"schema": "local_behaviour_annotation_v1", "feature_names": MOTION_NAMES + PERSON_NAMES,
            "clips": clips, "windows": windows, "source_counts": {
                "candidate_links": len(catalogue["candidates"]),
                "candidate_downloads": sum(r["download_status"] == "Downloaded" for r in catalogue["candidates"]),
                "user_supplied_local_clips": len(manifest["videos"]),
                "experimental_clips": len(clips), "review_windows": len(windows)},
            "warning": "Behaviour review is not four-tier risk ground truth. Exports require reviewed observations, quality eligibility and source-specific rights evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-dir", default="outputs/training/quality_motion_2026_10_14_run04")
    parser.add_argument("--output", default="data/review/annotation_package.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Package already exists; choose a fresh output rather than replacing annotations' source package")
    package = build(root, Path(args.quality_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(package, indent=2), encoding="utf-8")
    print(f"Prepared {len(package['windows'])} windows; no annotations approved")


if __name__ == "__main__":
    main()
