"""
train_yolo.py  [OPTIONAL / EXPERIMENTAL]
-----------------------------------------
Fine-tunes YOLOv8n on the 361 Love Parade "fight_onset" bounding-box frames.

IMPORTANT SCOPE NOTES:
- These boxes localise a panic-onset region in ONE viewpoint of ONE incident.
- The class name is fight_onset (preserved verbatim from the VOC annotation).
  Do NOT rename to "stampede" — this label has not been validated as a
  general stampede marker.
- This is NOT the primary model. It is a single-task localisation experiment.
- Results must not be presented as a general crowd panic detector.

Usage
-----
  python train_yolo.py \
      --data_dir "Abnormal High-density Crowds/3_Love Parade/Test" \
      --label_dir "Abnormal High-density Crowds/3_Love Parade/Test/Label/YOLO_darknet" \
      --output_dir runs/fight_onset_yolo \
      --epochs 50

The script will:
  1. Copy frames + labels into a YOLO dataset structure (80/20 split within
     Love Parade Test only — random but seeded, since this is one incident).
  2. Write dataset.yaml with class name "fight_onset".
  3. Fine-tune yolov8n.pt.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

YAML_TEMPLATE = """\
# YOLO dataset: Love Parade fight_onset fine-tune
# ONE incident, ONE viewpoint — not a general stampede detector.
path: {abs_dataset_dir}
train: images/train
val:   images/val

nc: 1
names: ['fight_onset']
"""


def build_dataset(data_dir: Path, label_dir: Path, dataset_dir: Path, seed: int = 42):
    frames = sorted(data_dir.glob("scene-*.png"))
    labels = {p.stem: p for p in label_dir.glob("*.txt")}

    # Only use frames that have a matching label file
    paired = [(f, labels[f.stem]) for f in frames if f.stem in labels]
    if not paired:
        sys.exit(
            f"ERROR: No frame/label pairs found.\n"
            f"  frames dir: {data_dir}\n"
            f"  labels dir: {label_dir}"
        )

    random.seed(seed)
    random.shuffle(paired)
    split = int(0.8 * len(paired))
    train_pairs = paired[:split]
    val_pairs = paired[split:]

    print(f"Paired: {len(paired)} | train: {len(train_pairs)} | val: {len(val_pairs)}")

    for split_name, pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_out = dataset_dir / "images" / split_name
        lbl_out = dataset_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for frame_path, label_path in pairs:
            shutil.copy2(frame_path, img_out / frame_path.name)
            shutil.copy2(label_path, lbl_out / label_path.name)

    yaml_path = dataset_dir / "dataset.yaml"
    yaml_path.write_text(YAML_TEMPLATE.format(abs_dataset_dir=str(dataset_dir.resolve())))
    print(f"Dataset YAML -> {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="[EXPERIMENTAL] Fine-tune YOLOv8n on Love Parade fight_onset frames."
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to 3_Love Parade/Test (contains scene-NNNNN.png)."
    )
    parser.add_argument(
        "--label_dir", required=True,
        help="Path to 3_Love Parade/Test/Label/YOLO_darknet (contains .txt labels)."
    )
    parser.add_argument(
        "--output_dir", default="runs/fight_onset_yolo",
        help="Directory for training outputs (default: runs/fight_onset_yolo)."
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs (default: 50)."
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640)."
    )
    parser.add_argument(
        "--batch", type=int, default=8,
        help="Batch size (default: 8 — conservative for 361 frames)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ERROR: ultralytics not installed. Run: pip install ultralytics")

    data_dir = Path(args.data_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    dataset_dir = output_dir / "dataset"

    print("=" * 60)
    print("EXPERIMENTAL: fight_onset YOLO fine-tune")
    print("  Scope: ONE incident, ONE viewpoint (Love Parade 2010)")
    print("  Class: fight_onset  (do not rename to 'stampede')")
    print("  361 labelled frames total")
    print("=" * 60)

    yaml_path = build_dataset(data_dir, label_dir, dataset_dir, seed=args.seed)

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(output_dir),
        name="train",
        seed=args.seed,
        exist_ok=True,
        # Light augmentation only given small dataset
        degrees=5.0,
        flipud=0.1,
        fliplr=0.5,
        mosaic=0.5,
        augment=True,
    )

    print("\nTraining complete.")
    print(f"Weights -> {output_dir / 'train' / 'weights' / 'best.pt'}")
    print(
        "\nREMINDER: This model localises panic onset in Love Parade video only. "
        "Do not use as a general stampede detector or integrate into the "
        "primary alert pipeline without validation."
    )


if __name__ == "__main__":
    main()
