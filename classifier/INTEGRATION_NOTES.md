# Experimental classifier integration

This package contains `infer.py`, `extract_features.py`, `yolov8n.pt`,
`model/classifier.pkl`, `model/feature_order.txt`, `model/metadata.json`,
`requirements.txt`, `TRIGGER_POLICY.md`, and `SHA256SUMS.json`.

The classifier was trained with YOLOv8n person detections at the source video
frame rate. Use the packaged detector weights and extractor, 30-frame windows,
15-frame stride, and the actual source FPS. The existing live baseline uses
YOLO11s at 5 FPS; its features are not interchangeable with these features.
The extractor expects capture-ordered `scene-*.png` frames. It computes person
count and box overlap from YOLOv8n and motion from grayscale frame difference.

Output is an experimental observation in a separate channel. It does not map
to a risk tier, activate the buzzer, or override the existing baseline.
The proposed trigger and clearance policy is in `TRIGGER_POLICY.md`.

Training-footage replay observed 0/345 false-positive windows. Independent
false-positive window rate and false-alert episodes/hour remain unknown.
Offline processing duration is not live detection delay. Evaluate independent
normal footage and measure live timing before activation.

Example from this directory:

```powershell
python infer.py --image_dir path/to/frames --fps 25 --model_dir model --ground_truth 0 --output_csv results/normal.csv
```

Record footage provenance, duration, FPS, ground truth, positive windows,
total windows, and distinct alert episodes per hour under the draft policy.
