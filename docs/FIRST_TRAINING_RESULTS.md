# First experimental training run

Completed 14 October 2026. This is a real fitted motion-anomaly experiment, **not** a trained four-tier stampede-risk classifier. The existing fog risk model has not been replaced.

## What was trained

User-provided `2.mp4` (L002) and `3.mp4` (L003) supplied reference motion. `Video Project 10.mp4` (L009) and `Video Project 11.mp4` (L010) were scored as exploratory probes, not ground-truth positive events. The two reference clips appear to share a scene, so no independent-event accuracy is available.

- Four videos produced 70 roughly one-second feature windows.
- 26 reference windows fitted each Isolation Forest; 14 later reference windows calibrated a provisional threshold, with a two-second temporal gap. Remaining reference windows were excluded from fit/calibration to respect the boundary and gap, but were included in diagnostic scoring.
- Both bundles use five frame-diagonal-normalised motion features, not person counts, measured density or crowd pressure.
- One variant uses raw optical flow. The other subtracts median flow as an ablation proxy; it is not geometric camera stabilisation and can erase coherent crowd motion.
- No YOLO fine-tuning or risk-label invention occurred. The pipeline used the CPU. PyTorch now reports CUDA available for subsequent GPU work.
- The original videos were not changed. No new footage was downloaded. Getty-watermarked files were excluded.

The first attempt stopped before fitting because its lighting check confused moving pixels with global brightness change. The measurement was corrected and tested; thresholds were not lowered to force training. The incomplete `run01` directory is retained as a diagnostic record, not a finished model.

## Results

These percentages describe windows exceeding an exploratory motion threshold. They are **not accuracy, recall, danger probabilities, false-alarm rates or evidence of incident detection**.

| Clip | Windows | Raw-flow exceedances | Median-subtracted-flow exceedances |
| --- | ---: | ---: | ---: |
| L002: `2.mp4` | 23 | 0/23 | 0/23 |
| L003: `3.mp4` | 23 | 1/23 | 1/23 |
| L009: `Video Project 10.mp4` | 12 | 0/12 | 1/12 |
| L010: `Video Project 11.mp4` | 12 | 3/12 | 12/12 |

Each variant exceeded its calibration threshold on 1/14 reference calibration windows. This is a same-scene temporal diagnostic, not an independent test or a claim that the reference windows are safe.

The entrance clip's low motion score demonstrates why a motion-only detector cannot be assumed to detect congestion. The large disagreement for L010 indicates sensitivity to the feature representation and scene changes. The cut heuristic retained windows in segments 0 and 3 for L010; intervening segments were too short to form retained windows. These are unverified heuristic boundaries, so manual shot review is still required before temporal risk analysis.

**Decision: retain the models as research baselines only. Do not deploy either bundle to the fog alarm.** Model fitting succeeded; validated crowd-risk performance has not been established.

## Saved artefacts

All completed run files are in `outputs/training/motion_baseline_2026_10_14_run02/`:

- `motion_raw.joblib` and `motion_residual.joblib`: experimental bundles with `deployable=false`, no risk classes and unverified rights status. These are not compatible with `fog/classifier.py`.
- `features.json`: timestamped motion-window descriptors.
- `scores_raw.json` and `scores_residual.json`: per-window scores and provisional threshold exceedances.
- `report.json`: parameters, counts, source hashes, limitations and summaries.

Both saved bundles were reloaded and scored all 70 feature windows successfully. Load only trusted joblib files; deserialisation can execute code. The current run uses OpenCV 4.12.0, scikit-learn 1.9.1 and the project's installed environment.

22 focused tests passed, including regression tests for timing, split rules, motion descriptors and the lighting-check correction. This does not resolve the previously stalled full fog integration suite or validate physical alarms.

## Strategy if results are poor

1. **Fix data and feature quality first.** Preserve original FPS/edit history, split cuts manually, define crowd ROIs, test lighting sensitivity and use actual stabilisation or reject moving/zooming shots. Normalise pixel-motion scale consistently, but recognise perspective differences remain. Collect low-motion congestion examples, not only running clips.
2. **Add complementary local features.** Evaluate person/head counting on sample frames before trusting it. Compute occupancy changes and local grid flow near entrances. Small heavily occluded people can defeat a pretrained person detector; choose a different counting method only after reviewing errors. Unknown physical scene area must not be fabricated as people/m².
3. **Create reviewed time-window labels.** Record observable walking, running, congestion, loss-of-control evidence and uncertainty. Same-event crops share an event ID. A video's title or incident location is not a label. Four risk classes should only be attempted when all are supported by data; otherwise start with a narrower observable-behaviour task.
4. **Compare simple supervised baselines.** Once labels and permissions are available, compare a dummy baseline, class-weighted Random Forest and an appropriate alternative using event-separated validation. Report per-class precision/recall, confusion matrix, false alerts per minute and warning delay where onset is actually annotated. Keep a final independent event test untouched.
5. **Only then consider deep training.** More epochs or a larger GPU cannot repair missing labels, scene leakage or feature corruption. Detector fine-tuning requires bounding-box annotations; temporal neural models need much more independent event data than these four short clips. Select the simplest model meeting measured performance and latency requirements.

Do not keep tuning on L009/L010 until they look positive: that would turn probes into training targets without ground truth. Mark these as development probes and gather new independent events for evaluation.

## What Karthikeyan needs to supply later

The experiment is finished and saved. The exact Kaggle link has subsequently been received and inspected; see [GPU results and dataset audit](GPU_TRAINING_RESULTS.md). It lists 17 raw MP4s and no annotation files. Original-file mapping/edit history for renamed clips and upstream footage rights remain unresolved. Do not distribute the fitted models or publish the clips until appropriate rights are established. Reviewed temporal labels remain necessary for the intended risk classifier.

## Reproduce

From the project root, using a fresh output directory:

```powershell
python -m training.motion_baseline --output-dir outputs/training/motion_baseline_new_run
python -m pytest -q tests/test_features.py tests/test_fog.py tests/test_motion.py tests/test_training.py tests/test_motion_baseline.py
```

The output directory must not already exist, to protect saved runs. The input manifest points to the original local files. This experimental command is distinct from `python -m training.train`, whose supervised reviewed-data gates remain intact.

Method reference: [scikit-learn Isolation Forest documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html). Its anomaly scores describe reference-distribution isolation, not stampede risk.
