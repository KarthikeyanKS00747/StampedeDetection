"""Experimental video audit of density, head, pose, occlusion and tracking signals.

This reports separate measurements. It does not sum overlapping detections or
change the classifier/alarm. Unprovided model weights produce blank CSV values.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from classifier.density_model import DensityNet, IMAGE_SIZE, OUTPUT_STRIDE, letterbox
from edge.person_tracking import PersonTracker


FIELDS = ["frame", "timestamp_s", "person_boxes", "tracked_people",
          "head_boxes", "pose_instances", "occlusion_person_boxes",
          "density_estimated_people"]


def optional_yolo(path: Path | None, task: str) -> YOLO | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"{task} weights not found: {path}")
    model = YOLO(str(path))
    if model.task != task:
        raise ValueError(f"{path} has task {model.task!r}; expected {task!r}")
    return model


def load_density(path: Path | None) -> DensityNet | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Density weights not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("image_size") != IMAGE_SIZE or checkpoint.get("stride") != OUTPUT_STRIDE:
        raise ValueError("Density checkpoint preprocessing does not match")
    model = DensityNet()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def density_count(model: DensityNet, frame) -> float:
    canvas, _, _, _ = letterbox(frame)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None] / 255.0
    return float(model(tensor).sum().item())


def pose_count(result, min_keypoint_confidence: float) -> int:
    """Count pose instances with at least two supported visible keypoints."""
    keypoints = result.keypoints
    if keypoints is None:
        return 0
    if keypoints.conf is not None:
        visible = keypoints.conf >= min_keypoint_confidence
    else:
        visible = (keypoints.xy != 0).any(dim=2)
    return int((visible.sum(dim=1) >= 2).sum().item())


def run(args: argparse.Namespace) -> int:
    if not args.person_weights.is_file():
        raise FileNotFoundError(f"Person weights not found: {args.person_weights}")
    person = YOLO(str(args.person_weights))
    head = optional_yolo(args.head_weights, "detect")
    pose = optional_yolo(args.pose_weights, "pose")
    occlusion = optional_yolo(args.occlusion_weights, "detect")
    density = load_density(args.density_weights)
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {args.video}")
    tracker = PersonTracker()
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise ValueError("Video FPS must be positive")
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    try:
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            while processed < args.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                person_result = person(frame, classes=[0], verbose=False)[0]
                tracked_people, _ = tracker.update(person_result, 1 / fps)
                head_result = head(frame, classes=[args.head_class], verbose=False)[0] if head else None
                pose_result = pose(frame, imgsz=args.pose_imgsz, verbose=False)[0] if pose else None
                occlusion_result = (occlusion(frame, classes=[args.occlusion_class],
                                              verbose=False)[0] if occlusion else None)
                writer.writerow({
                    "frame": args.start_frame + processed,
                    "timestamp_s": round((args.start_frame + processed) / fps, 3),
                    "person_boxes": len(person_result.boxes),
                    "tracked_people": tracked_people,
                    "head_boxes": len(head_result.boxes) if head_result is not None else "",
                    "pose_instances": pose_count(pose_result, args.keypoint_conf)
                    if pose_result is not None else "",
                    "occlusion_person_boxes": len(occlusion_result.boxes)
                    if occlusion_result is not None else "",
                    "density_estimated_people": round(density_count(density, frame), 2)
                    if density is not None else "",
                })
                processed += 1
    finally:
        capture.release()
    print(f"Audited {processed} consecutive frames -> {args.output}")
    for name, model in (("head", head), ("pose", pose), ("occlusion", occlusion),
                        ("density", density)):
        if model is None:
            print(f"{name}: unavailable (no trained weights supplied)")
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--person-weights", type=Path, default=Path("classifier/yolov8n.pt"))
    parser.add_argument("--head-weights", type=Path)
    parser.add_argument("--pose-weights", type=Path)
    parser.add_argument("--occlusion-weights", type=Path)
    parser.add_argument("--density-weights", type=Path)
    parser.add_argument("--head-class", type=int, default=0)
    parser.add_argument("--occlusion-class", type=int, default=0)
    parser.add_argument("--keypoint-conf", type=float, default=0.5)
    parser.add_argument("--pose-imgsz", type=int, default=640)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=30)
    args = parser.parse_args()
    if (args.max_frames < 1 or args.start_frame < 0 or args.pose_imgsz < 64
            or not 0 <= args.keypoint_conf <= 1):
        parser.error("Invalid frame range, pose size, or keypoint confidence")
    run(args)


if __name__ == "__main__":
    main()
