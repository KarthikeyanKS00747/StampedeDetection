# Experimental head-density model

This is a **separate, untrained** crowd-counting model in [density_model.py](density_model.py).
It predicts a nonnegative density map; summing the map gives an estimated count.
It is not a stampede detector, is not part of the current classifier or alarm,
and has no trained checkpoint yet. It will not improve the present video results
until it has independent human labels and passes evaluation.

The current baseline's `density = YOLO count / area_m2` is not independent of
YOLO. This model instead learns from manually placed head-center points. Its
targets are unit-mass Gaussians, so the target map sums to the annotated count.
This is a common crowd-counting formulation; see [CSRNet](https://arxiv.org/abs/1802.10062).
The network here is deliberately smaller and **is not CSRNet or pretrained**.

## Prepare annotations

Launch the [local head-point labeler](../review/head_labeler.py) from the
project root:

```powershell
.\.venv\Scripts\python.exe -m review.head_labeler serve --port 8012
```

Open `http://127.0.0.1:8012/`. The 15 images from
[`outputs/person_audit/manifest.csv`](../outputs/person_audit/manifest.csv)
are loaded locally. Zoom and click the centre of each visibly identifiable
head, including distant heads. Save each frame as Reviewed with reviewer name
and observation note. Use Uncountable when blur/occlusion prevents a reliable
count; it is not a zero-person label. Do not infer hidden heads or copy YOLO's
boxes as labels. Your saved review is kept at `data/review/head_points.json`.

The [count-error evaluator](../scripts/evaluate_person_counts.py) works after
some frames are reviewed, **without** training a density model:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_person_counts --comparison outputs/person_resolution/aerial/15923114-uhd_3840_2160_30fps_frames_01259_comparison.csv outputs/person_resolution/concert/12695727-hd_1920_1080_24fps_frames_00138_00139_comparison.csv outputs/person_resolution/march/4622524-uhd_3840_2160_24fps_frames_00078_comparison.csv
```

It reports absolute count error against **reviewed visible heads**, not person
box precision/recall or the number of people hidden behind others. With no
reviewed labels, it refuses to print an accuracy score. The 15 sampled frames
are only a first audit, not enough to establish generalization.

For **training**, export reviewed frames with one entire video held out.
The local labeler records the user's stated source permission in metadata;
this has not been independently verified. No licensing questions appear in
the form. The original audit images remain on disk, while the Save button
stores head coordinates in `data/review/head_points.json`. The separate
Download button saves a marked-up preview image to the browser's download
location:

```powershell
.\.venv\Scripts\python.exe -m review.head_labeler export --val-video 15923114-uhd_3840_2160_30fps.mp4
.\.venv\Scripts\python.exe classifier/density_model.py train --train data/review/density_labels/head_train.jsonl --val data/review/density_labels/head_val.jsonl --output outputs/density/best.pt --epochs 20
```

The export refuses empty training/validation splits and never puts frames of
one video into both. You can also create separate UTF-8 JSONL files manually.
One line per image, for example:

```json
{"image":"../outputs/person_audit/example_frame_original.jpg","scene":"camera-A-video-1","points":[[120,80],[195,98]]}
```

`image` is relative to the JSONL file (or absolute). Points are head centers
in the **original image's** pixel coordinates, not YOLO's boxes. Use `[]` for a
verified empty frame. Review frames that are too blurred or occluded to label;
do not invent positions. Give all frames from one video/camera event the same
`scene` ID. Training and validation must use different scenes, ideally
different recording sessions/cameras. The audit CSV contains YOLO predictions,
**not** these labels.

## Train and inspect

From the project root, using the project's configured Python environment:

```powershell
python classifier/density_model.py train --train path/to/train.jsonl --val path/to/val.jsonl --output outputs/density/best.pt --epochs 20
python classifier/density_model.py predict --weights outputs/density/best.pt --image path/to/frame.jpg --output outputs/density/overlay.jpg
```

The trainer reports validation count mean absolute error (MAE) and saves the
best checkpoint with its scene split. Check predictions and overlays against
human counts on *independent* normal and abnormal footage. A low validation MAE
on the same event is not evidence of generalization. The default 512-pixel
letterbox can still lose very small aerial heads; annotation quality, camera
perspective, resolution, and inference time must be assessed before deployment.
The overlay highlights estimated density, not individual person detections.

The user's [head_count screenshot](../../head_count/Screenshot%202026-10-15%20135650.png)
contains numbered head marks and is useful as a visual review reference. It is
a cropped/resized screenshot, not head-point coordinates for any original video
frame. Do not treat its marks as training labels without identifying the exact
original frame and mapping/reviewing every point at original resolution.

Do not add these outputs to existing classifier features or change alert
thresholds without re-extracting features, retraining, and independent testing.
Do not sum the density-map count and YOLO count directly: their regions overlap
and people would be counted twice. Spatial fusion is a later experiment.

## Four-signal video audit

[density_multisignal.py](density_multisignal.py) can report independent head
detector, pose, occlusion-aware detector, and person-tracking outputs alongside
the density estimate on **consecutive** video frames. It deliberately does not
sum these overlapping counts or feed them into the anomaly classifier.

```powershell
python -m classifier.density_multisignal --video path/to/video.mp4 --output outputs/density/audit.csv --max-frames 30 --head-weights path/to/head.pt --pose-weights path/to/pose.pt --occlusion-weights path/to/crowd_person.pt --density-weights outputs/density/best.pt
```

All four optional weight paths are locally trained/obtained checkpoints; no
weights are silently downloaded. The head and occlusion weights must be YOLO
**detection** models and the pose weights a YOLO **pose** model. Set their class
IDs if person/head is not class 0. The script checks task type but cannot prove
the models were trained for the named purpose. Missing weights produce blank
columns and an explicit availability message, never a fabricated zero. The
existing YOLOv8n person detector and ByteTrack path can run without them.
`pose_instances` requires at least two keypoints above the chosen threshold;
it is diagnostic, not a reliable independent people count. Tracking does not
recover people never seen by a detector. Only manual annotations on held-out
videos can establish whether any combination improves accuracy.

## 2026-10-15 offline background-person audit

The official Ultralytics [YOLOv8n pose model](https://docs.ultralytics.com/models/yolov8/)
was downloaded to
`checkpoints/yolov8n-pose.pt` for **experimental** use. Its SHA-256 is
`c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212`.
It is ignored by Git, so copy it separately if this experiment is shared.
It is not a head detector or an occlusion-aware detector.

The [resolution comparison](../scripts/compare_person_resolution.py) uses the
same YOLOv8n person weights at 640, 1280, and 1024-pixel overlapping tiles
(tile inference size 960). Tile-centre ownership and NMS reduce seam
duplicates. On one aerial frame (15923114, frame 1259), candidate box counts
were **0 / 8 / 117**. On one concert frame (12695727, frame 138), they were
**10 / 15 / 13**; the next concert frame had **10 / 17 / 27**. These are
candidate boxes, **not ground-truth people counts or accuracy scores**. Visual
review found both recovered people and false/merged boxes. Tiling helped the
aerial view but did not give stable concert counts. A protest/march frame
(4622524, frame 78) had **16 / 14 / 24**, with many people still hidden by
signs, occlusion, and darkness. Earlier comparison CSVs
without tile-centre ownership are superseded by the `frames_*_comparison.csv`
results.

The [tracking comparison](../scripts/track_tiled_people.py) fed all three
methods to independent ByteTrack instances over five consecutive frames. On
the aerial sample, full-640 had 0 tracks, full-1280 had 0–4, and tiled had
81–95. On the concert sample, full-640 had 6–7, full-1280 had 6–9, and
tiled had 4–9 unstable tracks. After warm-up, full-1280 inference took about
0.10–0.16 s/frame, tiled inference about 1.49–1.64 s/aerial frame and
0.61–0.65 s/concert frame on this CPU. Tiling is well short of a 5-FPS
(0.20 s/frame) live budget. Full-1280 inference alone is closer, but total
end-to-end live latency has not been measured. The tracker cannot verify that
each track is a person; false detections, duplicates and ID switches remain.
The tested CSVs are
[`aerial`](../outputs/person_resolution/aerial/tracking_three_methods_1259.csv)
and [`concert`](../outputs/person_resolution/concert/tracking_three_methods_138.csv).

The pose model at inference size 1280 yielded 0 supported pose instances on
the aerial frame and 18 on the concert frame. It did not solve tiny aerial
people. Head and occlusion-aware models, and a trained density checkpoint,
remain unavailable. Thus no reliable four-way fused count or improved
stampede classification is claimed.

Example commands from the project root:

```powershell
python scripts/compare_person_resolution.py --video 15923114-uhd_3840_2160_30fps.mp4 --frames 1259 --output-dir outputs/person_resolution/aerial
python -m scripts.track_tiled_people --video 15923114-uhd_3840_2160_30fps.mp4 --start-frame 1259 --frames 5 --output outputs/person_resolution/aerial/tiled_tracking_1259.csv
python -m classifier.density_multisignal --video 12695727-hd_1920_1080_24fps.mp4 --start-frame 138 --max-frames 1 --pose-weights classifier/checkpoints/yolov8n-pose.pt --pose-imgsz 1280 --output outputs/density/concert_pose_frame138.csv
```

Next: manually annotate person/head boxes or points on held-out camera scenes,
measure recall/precision and count error, then consider targeted fine-tuning
or a faster ROI-based detector. Keep these experimental outputs out of the
trained sequence classifier until feature extraction, training, and evaluation
are aligned.

## Contrast-and-split audit

[audit_contrast_splits.py](../scripts/audit_contrast_splits.py) reruns the same
YOLO person detector after LAB-luminance CLAHE enhancement. It flags an original
box as a possible split only if two smaller candidate boxes have centres inside
it. This is a prompt for human review, not an automatic person count. Zooming
or contrast cannot reveal fully occluded people and may create false positives.

On JPEG audit samples from 2026-10-15, full-frame 1280-pixel inference gave
**8 original / 8 contrast** boxes on aerial frame 1259 and **14 / 13** on
concert frame 138. The tiled aerial pass gave **114 / 103** (these JPEG input
counts need not exactly match previous direct-video frame counts). The split
heuristic found **zero** parent boxes on all three comparisons. Thus this small
test does not support contrast enhancement as a solution to the missed-person
problem. See the saved overlays and CSVs under
[`outputs/contrast_audit/`](../outputs/contrast_audit/). Counts are unverified
candidate boxes, not accuracy or verified people counts.
