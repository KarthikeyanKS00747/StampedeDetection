# Karthikeyan's ML/data takeover checklist

Updated 14 October 2026. Read this first; it supersedes older handbook statements that no anomaly model or API tests exist. The original stampede-warning classifier is **not complete**.

## 1. Ownership after handover

Karthikeyan's unfinished responsibility is **data provenance, temporal labels, model training and independent evaluation**, not building another dashboard. Assign one teammate as the ML/data lead. Samridh's existing fog/dashboard work is in the repository; Samrudh owns Pi setup and hardware checks. Agree who takes ML/data ownership rather than assuming that finishing their original assignments finishes the classifier.

Karthikeyan only needs to:

1. Send the repository/commit reference and this document to the successor.
2. Share the local experimental artifact bundle described below through an agreed private channel if they need the immediate demo. No automatic upload or teammate messaging has been performed.
3. Explain the four clip filenames and their original sources/crops. Rights remain unresolved; do not claim that cropping or the Kaggle uploader's MIT badge clears third-party footage.
4. Hand over hardware to Samrudh: both Pi 4 boards, their power supplies/cables, LEDs, resistors, buzzer and wires. Confirm missing storage/camera/driver parts using `docs/DEVICE_HANDOVER.md`.

You do **not** need to review all 50 sources again or rerun the completed GPU experiments merely to transfer ownership.

## 2. What is in Git and where

| Location | Purpose |
| --- | --- |
| `edge/main.py`, `edge/features.py`, `edge/transport.py` | Existing live seven-feature extraction, transport and simulation |
| `edge/motion.py`, `edge/video_quality.py` | Correct time units and offline camera-quality helpers |
| `fog/app.py`, `fog/static/`, `fog/alerts.py`, `fog/gpio_alarm.py` | Samridh's fog/dashboard/alert path plus separate ML observation integration |
| `fog/classifier.py`, `shared/schemas.py` | Original seven-feature/four-tier risk contract and guarded model loading |
| `fog/anomaly.py` | Nine-feature research anomaly adapter; no hardware alarms or risk labels |
| `training/train.py` | Supervised four-tier risk trainer; requires reviewed seven-feature risk data |
| `training/train_behaviour.py` | Supervised observable-behaviour trainer; requires reviewed nine-feature exports |
| `training/*baseline.py` | Reproducible exploratory extraction/fitting; not field-validated models |
| `review/` | Local-only short-window annotation app |
| `scripts/process_ml_video.py`, `scripts/replay_ml.py` | New-video batch GPU extraction and model-to-dashboard replay |
| `scripts/prepare_local_manifest.py`, `scripts/build_annotation_package.py` | Set local video paths and build review windows |
| `scripts/verify_project.py`, `tests/` | Integration report generation and tests |
| `data/review/video_candidates_50.json` | All 50 source URLs and the team's existing review feedback |
| `data/review/kaggle_source_audit.json` | Kaggle metadata/file inventory audit; no ground-truth labels found |
| `docs/PROJECT_HANDOFF.md` | Demo commands and precise completion boundaries |
| `docs/ANNOTATION_WORKFLOW.md` | What to label and how to save/export observations |
| `docs/GPU_TRAINING_RESULTS.md`, `docs/QUALITY_TRAINING_PROGRESS.md` | Experimental results and limitations |
| `deploy/`, `docs/HANDBOOK_*.md` | Pi deployment and original teammate assignments |

## 3. What is deliberately NOT pushed

`.gitignore` now excludes `outputs/`, video media, joblib weights, detector weights, downloaded raw datasets, machine-specific local manifests/review packages and reviewer saves. Ignoring a file does not delete it locally. The six previously tracked generated review files are removed from the Git index, retained on the original laptop and recoverable from earlier commits. Their earlier approval claims are superseded by current curation documents.

The clone alone will not include a working experimental model or your private clips. Some artifact-dependent tests skip on a clean clone; the original machine passed all 69 tests with its local artifacts present. Do not interpret skipped tests as equivalent verification.

Local files to transfer for the existing ML replay, maintaining these relative paths:

These files are packaged locally in `outputs/handoff/experimental_ml_bundle.zip`. Extract it into the successor's repository root. It contains no raw videos, reviewer saves or machine-local manifests; it has not been uploaded. The archive also includes `ARTIFACT_README.md` with the recorded Python/library versions used to create it.

- `outputs/training/quality_motion_2026_10_14_run04/quality_joint.joblib`
- `outputs/training/quality_motion_2026_10_14_run04/features_quality.json`
- `outputs/training/quality_motion_2026_10_14_run04/report.json`
- `outputs/completion/integration_report_20261014.json`
- `models/yolo11s.pt` only if new-video detection is required; cached-feature replay does not need these detector weights.

Only load trusted artifacts received from the project's owner: joblib/pickle can execute code. The research model has `deployable=false` and `rights_verified=false`. Sharing or running it does not resolve permissions. No new media/model files have been published to GitHub.

For annotation, the original private folder is `C:\Users\LENOVO\Downloads\stampede-videos`. The current experiment uses `2.mp4`, `3.mp4`, `Video Project 10.mp4`, `Video Project 11.mp4`. Obtain lawful access before transferring/using footage. The two corridor clips must remain in the same event group. A renamed/edited clip is not an independent event.

## 4. First run on the successor's laptop

If GitHub has not received the handoff commit because the required release gate is blocked, use the local `outputs/handoff/StampedeDetection.bundle` instead. Copy it and the experimental artifact ZIP through your agreed private handover channel. On the successor's machine:

```powershell
git clone "D:\handoff\StampedeDetection.bundle" StampedeDetection
cd StampedeDetection
```

Replace the bundle path with its actual location. This is a local/offline transfer, not a GitHub push. The repository bundle contains committed source/history, not ignored videos or models; the artifact ZIP supplies the experimental demo files. The ZIP is only suitable for trusted internal research use under the limitations above.

Clone the repository into your own folder and work there. Python/dependency compatibility must be checked; the original run used Python 3.13, PyTorch 2.14.1+cu126 and Ultralytics 8.4.175. `requirements.txt` uses version ranges, not a complete lockfile. Installing plain requirements does not guarantee CUDA is available: use the official PyTorch installation instructions for your machine and verify `torch.cuda.is_available()` before GPU processing. No GPU is needed for cached-feature replay or tree fitting.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Extract the trusted artifact bundle into the clone root, preserving folders. Then use two terminals:

```powershell
# Terminal 1
.\.venv\Scripts\python.exe -m fog.app --host 127.0.0.1 --port 8000 --anomaly-model outputs/training/quality_motion_2026_10_14_run04/quality_joint.joblib
# Terminal 2
.\.venv\Scripts\python.exe -m scripts.replay_ml --fog-url http://127.0.0.1:8000 --interval 1
```

Open `http://127.0.0.1:8000/`. Expect experimental reference/anomaly observations and Cannot assess windows—not four risk probabilities. Stop with Ctrl+C. The service has no authentication; keep it on loopback or a deliberately configured trusted lab LAN.

## 5. Restore annotation on another laptop

Put the four original experiment clips in a local directory. Do not substitute different files under the same names when reproducing results.

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_local_manifest --video-dir "D:\project-videos"
.\.venv\Scripts\python.exe -m scripts.build_annotation_package --quality-dir outputs/training/quality_motion_2026_10_14_run04
.\.venv\Scripts\python.exe -m review.app --port 8011
```

Open `http://127.0.0.1:8011/`. The manifest builder refuses to overwrite an existing manifest. The package builder refuses to overwrite an existing package; preserve previous work and choose fresh outputs when necessary. Copy actual reviewer saves privately if any exist; do not check them into Git automatically.

For full regeneration instead of receiving the bundle, prepare the local manifest, then run:

```powershell
python -m training.motion_baseline --output-dir outputs/training/motion_rebuild
python -m training.gpu_person_baseline --motion-dir outputs/training/motion_rebuild --output-dir outputs/training/gpu_rebuild --model models/yolo11s.pt --device 0
python -m training.quality_motion_baseline --gpu-dir outputs/training/gpu_rebuild --output-dir outputs/training/quality_rebuild
```

Use the regenerated `quality_rebuild` path in fog, replay and package commands. All outputs must be fresh directories. Rebuilding this historical experiment still does not establish rights, label quality or field accuracy.

## 6. The remaining ML/data work, in order

1. **Choose the target and secure data.** Decide whether the assessed deliverable is crowd-behaviour/anomaly detection or a genuinely evaluated four-tier risk classifier. Preserve that distinction. Acquire suitable source-specific research/training permission and document it before assigning further source reviews. The 50-link catalogue is not 50 acquired, labelled videos.
2. **Review short windows.** Use the annotation app for observable walking/running/stationary/mixed motion, reviewer notes and uncertainty. Do not automatically assign danger from density, running, event names or anomaly scores. Resolve licence evidence separately.
3. **Add independent events.** L002/L003 share a scene. Adjacent windows, copies, crops and multiple shots of one event cannot populate both training and evaluation sides. Obtain representative normal and relevant unusual behaviour examples with a genuinely independent held-out set.
4. **Train behaviour if that is the agreed target.** Export reviewed data and run `training.train_behaviour` as in `docs/PROJECT_HANDOFF.md`. It requires at least two independent event groups per observed class and records a development holdout report. No real supervised behaviour model exists yet.
5. **Train risk only with risk evidence.** For the original fog contract, generate the seven features with measured monitored area and consistent pixels/second, then provide reviewed `risk`, `event_id`, `reviewer`, `rights_approved`, `permission_evidence`, `motion_units`. Run `training.train`. Behaviour labels and the nine-feature anomaly export cannot be substituted. Unobserved risk classes are not validated.
6. **Evaluate, don't just fit.** Measure false alarms, missed events, class confusion, footage-quality rejection and inference latency on independent material. Warning lead time needs reviewed event/onset timestamps. Save dataset version, source provenance, split IDs, model configuration and actual environment versions. Training completion alone is not a successful result.
7. **Handle poor results.** Inspect errors by lighting, occlusion, camera movement and scene. Improve data/ROI or detector/count annotations before trying a larger model. Do not repeatedly tune on the final test set. Keep unreliable observations Unknown; don't relax gates merely to get more predictions.
8. **Integrate approved output.** Samridh receives the model, exact feature order/units, dependency versions and evaluation report. The current `--anomaly-model` route is dashboard-only. The `--model` route is the original supervised risk path. Hardware enabling is a separately justified lab decision, not an automatic conversion of anomaly scores.
9. **Finish live/Pi tests.** Offline camera-quality compensation is not integrated into the live seven-feature path. Samrudh/Samridh must test actual camera acquisition, Pi processing rate, dropped frames, network interruptions, GPIO and startup/shutdown behaviour. The RTX batch timing is not Pi real-time performance.

## 7. What to give each teammate physically

- ML/data successor: laptop/project checkout, this handbook, trusted experimental artifacts, source/provenance information and lawful access to data. GPU access is useful for detection, not required for the Random Forest/Isolation Forest fits.
- Samridh: laptop plus full repository, then access to fog Pi B on the trusted LAN. No separate GPU needed for cached ML inference.
- Samrudh: both Pi 4 boards, adequate USB-C supplies, boot storage, LEDs with verified resistors, wires and compatible buzzer/driver. A camera/video source and secure prototyping arrangement are still needed. Check the buzzer before wiring; an unknown/5 V bare buzzer must not be connected directly to GPIO.

Never collect demonstration data by causing unsafe crowd conditions. Use consenting ordinary movement in an open space if recording your own material.
