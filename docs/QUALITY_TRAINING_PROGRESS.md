# Quality-aware preprocessing and model progress

Completed 14 October 2026. This continues the experimental training work without requesting new user input. It does not establish a validated stampede classifier.

## Completed work

- Reran tiled YOLO11s extraction on the RTX 4070 and cached bounding boxes for all 403 sampled frames. Those boxes can now exclude detected people from candidate background points. No detector fine-tuning took place.
- Added black-border detection that retains dark pixels inside the visible content rectangle. Black clothes are not silently removed as if they were padding.
- Normalised optical-flow speeds against the visible-content diagonal, so adding black letterboxing does not change the denominator.
- Added feature-point tracking, forward/backward checks and a RANSAC partial-affine camera transform. Fits require enough points, broad spatial support, sufficient inliers and plausible transform parameters. Unsupported fits are rejected rather than invented.
- Added candidate shot-change detection and a 0.6-second transition guard. Optical flow is not computed across detected cuts or used inside the guard.
- Added per-frame and per-window quality metadata. Insufficient camera support or large luminance shifts mark the window indeterminate for this experiment.
- Fitted two quality-gated reference anomaly models: corrected motion alone, and corrected motion combined with person/count/coverage proxies.
- Created 70 timestamped behaviour-review proposals, plus 12 clean original-frame exports and four geometric comparison panels. Proposals retain null behaviour/risk labels and `training_approved=false`.

## Quality outcomes

| Clip | Windows | Numerically quality-eligible | Indeterminate |
| --- | ---: | ---: | ---: |
| L002: `2.mp4` | 23 | 18 | 5 |
| L003: `3.mp4` | 23 | 20 | 3 |
| L009: `Video Project 10.mp4` | 12 | 0 | 12 |
| L010: `Video Project 11.mp4` | 12 | 12 | 0 |
| Total | 70 | 50 | 20 |

Eligibility is a numerical quality gate, not evidence of correct counts, correct camera compensation, safety or a reviewed risk label. L009 had insufficient widely supported candidate-background anchors; it is not classified as safe or dangerous by the corrected experiment. It may still be useful for other reviewed features or manual case-study analysis.

L010 shot-change candidates appeared at **5.8, 6.0 and 6.4 seconds**. These sampled detections cover a transition region, not three verified independent editorial cuts. Review the original transition before assigning exact boundaries. Neither reference clip produced a cut candidate.

Each corrected model fitted **20 reference windows** and calibrated a provisional threshold on **14 later reference windows**, using the existing temporal gap. The reference clips share a scene, so this is not an independent-event test. No risk accuracy, recall, lead time or calibrated probability can be reported.

Both models exceeded the provisional threshold on all 12 eligible L010 windows. That should not be interpreted as successful stampede detection: it is a different street scene, with different scale, appearance and motion. This remains a domain-shift warning until reviewed labels and independent reference events exist. No model was selected for alarms.

## Verified checks

**33 focused tests pass**, covering existing feature/fog unit behaviour, training gates, motion units, GPU feature geometry, camera correction and quality decisions. Python compilation and saved-model reload/scoring were also checked.

New tests demonstrate:

- A known synthetic camera translation is recovered and geometrically compensated.
- Missing anchors yield no correction.
- Person bounding boxes are excluded from candidate anchor masks.
- Black padding does not alter normalised motion speed, and dark clothing inside content is retained.
- An all-unreliable clip has a null threshold result, not an apparent safe/anomalous result.
- Score statistics use only eligible windows, not rejected windows with placeholder values.

The full fog integration suite remains unverified following its earlier stalled run. Hardware tests and real-camera performance have not been performed. This work does not modify teammates' fog implementation or connect experimental scores to GPIO.

## Authoritative saved run

Use `outputs/training/quality_motion_2026_10_14_run04/`:

- `quality_corrected.joblib` and `quality_joint.joblib`: research-only bundles requiring quality gating. They cannot be used as the existing fog risk classifier.
- `features_quality.json` and `frame_quality.json`: window features and frame-level cut/camera/lighting diagnostics.
- `scores_corrected.json` and `scores_joint.json`: unreliable-window scores and decisions are null.
- `report.json`: eligible/indeterminate counts and same-scene fit/calibration summaries.
- `review_proposals.json`: 70 proposed review windows. These are not ground truth, and no risk class was inferred from the dataset title.
- `Lxxx_clean_1.jpg` through `_3.jpg`: clean original frames for future annotation. No blank YOLO label files were created, since those could be misread as verified negative examples.
- `Lxxx_camera_comparison.jpg`: previous, current and corrected grayscale views from a numerically accepted pair. These are diagnostic examples, not proof of stabilisation across the whole video.

The GPU source cache is `outputs/training/gpu_person_2026_10_14_run04_boxes/`. Previous quality run directories are retained as development records. Run01's raw summary included excluded windows; run02 preceded content-diagonal normalisation; run03 preceded explicit unit metadata. Do not use those directories as final summaries.

Original input videos remain unchanged. No additional source videos were downloaded. All footage permission flags remain unresolved; the previously checked Kaggle page lists MIT but supplies no annotation files or independently verified upstream rights.

## Remaining technical limits

- Missed people can contaminate candidate background points. A numerically coherent transform can still subtract genuine coherent crowd motion. Synthetic correctness is not field validation.
- A partial-affine transform cannot generally solve perspective/parallax, rolling shutter or large zooms.
- Person counts, local image-grid counts and bounding-box coverage remain unvalidated proxies. Bounding-box coverage still uses the frame canvas; it is not physical crowd density or a perspective-calibrated occupancy map.
- The joint model's track-speed proxy is padding-normalised but not geometrically camera-compensated.
- Text, logos and embedded captions inside the active content are not comprehensively masked. Local flashing lights can survive a global luminance check.
- Variable-frame-rate and speed-edited sources remain unsupported without timing verification.
- The experimental quality module is not yet integrated into the live edge pipeline. These are offline diagnostics and models.
- Verified temporal behaviour labels, trustworthy count/head annotations and independent events remain necessary for supervised training/evaluation. Automatic proposals must not be promoted silently to ground truth.

## Reproduce the quality stage

The cached GPU detections avoid another detector pass. From the project root, choose a fresh output directory:

```powershell
python -m training.quality_motion_baseline --gpu-dir outputs/training/gpu_person_2026_10_14_run04_boxes --output-dir outputs/training/quality_manual_run
```

This stage uses CPU optical flow/camera estimation and CPU tree fitting, with previously GPU-extracted detections. It does not train YOLO on the GPU or manufacture risk labels.

Method references: [OpenCV geometric estimation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) and [optical-flow tracking](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html).

## Later verification and review workflow

The environment-specific test stall was resolved by running outside the restricted sandbox with an owned temporary directory. All 58 project tests passed on 14 October 2026, including fog integration and the new review API tests. The local annotation page was browser-checked for video playback without creating real labels. See [annotation workflow](ANNOTATION_WORKFLOW.md). Supervised behaviour training remains pending reviewed data; no risk model has been validated or deployed.
