# Experimental combined crowd detector

`combined_detector.py` evaluates two existing signals on the **same recording**
with one YOLOv8n detection pass per source frame:

1. The saved Random Forest classifies each 30-frame, 15-frame-stride window
   using the matching YOLOv8n extractor and source FPS.
2. On sampled frames, the same YOLO detections feed the live edge ByteTrack
   path. The existing edge feature extractor and fog risk classifier assess
   tracked counts, tracking velocities, and optical flow, normally at 5 FPS.
3. The draft policy opens an anomaly episode after three consecutive abnormal
   windows and closes it after five consecutive normal windows. A **combined
   candidate** appears while that episode is active and the latest baseline
   observation is High Risk or Critical and at most 1.5 seconds old.

Each transition into a combined candidate is a separate candidate episode.
The output is a CSV for human review. It does not call the fog alert engine,
change a risk tier, notify an operator, or operate GPIO. This is a rule-based
fusion of two signals, **not a newly trained or validated stampede model**.

From the repository root, run on a recorded video or ordered frame directory:

```powershell
python -m classifier.combined_detector --video path/to/recording.mp4 --fps 25 --area-m2 25 --model-dir classifier/model_retrained_2026_10_15 --output-csv outputs/combined.csv
```

For a frame directory, replace `--video` with `--image-dir`; it must contain
sequential `scene-*.png` files. Set `--fps` to the constant source frame rate
and `--area-m2` to the calibrated monitored area. The
extractor requires `classifier/yolov8n.pt` and the frozen files in
`classifier/model/`. `--sample-fps` defaults to 5. `--risk-model` may point to
a separately validated seven-feature risk classifier; otherwise the current
explainable heuristic is used. Run this command in the full repository with
the root `requirements.txt` installed. The standalone `classifier/deploy/`
package contains the sequence classifier only.

For a bounded integration check, add `--max-frames 60`. The CSV logs the seven
sequence features, tracked baseline features and reasons, both decisions, and
each candidate episode transition. The `.csv.meta.json` sidecar records model
and detector hashes, source settings, and runtime. `window_processing_wall_s`
starts when the first frame of that window is read and ends at the decision;
`end_frame_processing_wall_s` starts when its last frame is read. These are
**offline computation durations**, not measured live capture-to-alert delay.

The 2026-10-15 retraining used the same 1,041 feature windows, 200 trees, and
seed 42 as the prior fit. The new classifier binary has the same SHA-256 hash
as `classifier/model/classifier.pkl`, and both versions make the same 88/90
positive-window predictions on the saved Italy holdout feature rows. This
retraining did not change the learned model or establish new accuracy,
precision, or false-positive performance. The previous default model and
standalone deployment package remain unchanged; pass `--model-dir` to select
the retrained artifact explicitly.

The dry run reuses the live edge ByteTrack logic and the fog risk classifier,
but it processes every source frame sequentially instead of capturing a live
stream. The baseline and sequence components share detections; the tracker is
updated at the selected sample rate. Recorded media timestamps and processing
time are logged separately. The run does not establish real-time throughput
or end-to-end live latency.

For a later matched live comparison, configure the edge process with the same
`classifier/yolov8n.pt` weights, sample FPS, and calibrated area. Feeding
YOLO11s/5-FPS features to the saved sequence model would violate its feature
contract; the sequence branch must still process source-rate frames.

No independent normal footage or multi-incident stampede ground truth has
been evaluated for this fusion. Before any alert integration, collect
independent normal and abnormal incidents, freeze the rule on a development
set, measure false candidate episodes/hour and event detection on separate
test incidents, and measure end-to-end live delay. Do not use training clips
to estimate those rates.
