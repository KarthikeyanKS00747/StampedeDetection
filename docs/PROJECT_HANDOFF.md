# ML integration handoff — 14 October 2026

## What now works

The existing teammate fog dashboard, HTTP feature API, WebSocket updates, history and GPIO risk path are preserved. A separate **trained experimental anomaly** path now works end to end:

Video → pretrained GPU person detection/tracking → quality-gated nine-feature windows → fitted Isolation Forest → fog HTTP API → WebSocket dashboard observations.

No fabricated density or four-tier labels bridge the incompatible seven-feature risk and nine-feature image contracts. The research stream cannot operate GPIO, assert Normal/safe, or report stampede probabilities. Unreliable windows display **Cannot assess**. The original heuristic/supervised risk route remains separate.

## Demonstrate now

Open two terminals in `C:\Users\LENOVO\GitHubRepos\StampedeDetection`.

Terminal 1:

```powershell
python -m fog.app --host 127.0.0.1 --port 8000 --anomaly-model outputs/training/quality_motion_2026_10_14_run04/quality_joint.joblib
```

Open `http://127.0.0.1:8000/` before replay.

Terminal 2:

```powershell
python -m scripts.replay_ml --fog-url http://127.0.0.1:8000 --interval 1
```

The dashboard's **Experimental ML observations** section shows model scores, thresholds, window times and quality reasons. A scene/domain change can explain an anomaly; it is not confirmation of a dangerous event. Stop services with Ctrl+C.

The existing teammate hardware path can be demonstrated separately with `python -m edge.main --fog-url http://127.0.0.1:8000 --simulate`. Those are explicitly synthetic heuristic inputs, **not** results from the ML stream. Do not call this a trained ML buzzer demonstration.

## Analyse a new video on the RTX 4070

Use footage you have permission to process. This is batch processing, not real-time inference:

```powershell
python -m scripts.process_ml_video --source "C:\path\to\video.mp4" --output-dir outputs/training/my_new_video --device 0
python -m scripts.replay_ml --features outputs/training/my_new_video/features_quality.json --fog-url http://127.0.0.1:8000
```

Choose a fresh output directory; existing output is never overwritten. The local detector weights are `models/yolo11s.pt`; `--device cpu` is supported but CPU/Pi throughput is not measured. Constant-FPS real-time footage is required; edits, zooms, occlusion and lighting remain limitations.

## Finish supervised training when reviewed data exists

The [review tool](ANNOTATION_WORKFLOW.md) exports nine-feature **behaviour** labels, not risk labels. A supervised trainer now consumes that export:

```powershell
python -m training.train_behaviour --input "C:\path\to\reviewed_behaviour_features.json" --output-dir outputs/training/reviewed_behaviour_run01
```

It refuses empty/unapproved/uncertain/duplicate/nonfinite data and requires at least two independent event groups per observed class. It produces weights, an event-separated classification report and confusion matrix. This trainer is implemented and its gates tested, but **no real supervised behaviour weights have been fitted** because there are currently zero approved reviewed rows. Test fixtures do not count as project training data.

For the original four-tier fog classifier, retain `python -m training.train` with reviewed seven-feature risk data as described in README. Neither anomaly nor behaviour bundles may be supplied as `--model`; the loader now rejects them explicitly.

## Verification evidence

- 69 tests passed, including existing teammate integration tests, research endpoint, WebSocket delivery, no hardware/alert invocation, score equivalence, invalid-contract rejection and behaviour event-split gates.
- All 70 cached windows were processed through both the tested application and an actual running HTTP service: 37 within reference distribution, 13 review notices, 20 cannot assess. These are **not accuracy results**.
- The RTX 4070 processed `Video Project 10.mp4` through the new command, producing 12 windows. All were correctly withheld by the numerical quality gate, matching the earlier clip audit.
- Integration report: `outputs/completion/integration_report_20261014.json`.
- Browser check confirmed model observations on the existing dashboard.

Load only trusted local joblib artifacts; joblib/pickle can execute code. Fog is a trusted-lab service without authentication; do not expose it publicly. A lab-network host binding should be explicitly chosen for Pi testing.

## What is not complete

Software integration and a research ML demonstration are complete. **The original validated stampede-warning classifier remains incomplete**, even if both teammates finish their parts:

1. Source-specific training rights and real reviewed temporal labels are still missing for the current reference clips. The anomaly bundle retains `rights_verified=false` and `deployable=false`; research integration does not clear those limitations.
2. Independent-event positive/negative evaluation, false-alarm rate, missed-event rate and warning lead time are not established. The current reference/calibration clips share a scene.
3. A separately collected representative dataset, calibrated area/perspective and a camera/video source are needed for field deployment.
4. Physical Pi wiring, network behaviour, sustained Pi throughput and hardware tests require the team and devices. They have not been verified here.
5. Offline quality compensation has not been integrated into the live seven-feature camera path. Batch-video ML is not live-camera ML.

Do not report 100% project completion or a detection-accuracy percentage from these results. No real stampede ground truth has been manufactured to make the software appear finished.
