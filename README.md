# Single Camera Edge Fog Stampede Early Warning

This project implements a privacy-preserving, single-camera early-warning system for abnormal crowd dynamics.  It is a decision-support prototype, **not a stampede predictor** and must not be used as the only safety control at a live event.

## What is included

- Edge pipeline: video/camera ingestion, optional YOLO person detection and ByteTrack tracking, Farneback optical flow, six crowd features, and durable HTTP delivery.
- Fog service: FastAPI REST and WebSocket API, rolling history, four-level risk classification, escalation rules, alert log, and a live browser dashboard.
- Hardware alarm: Raspberry Pi GPIO control for one buzzer and two LEDs (green and red), including a safe simulation fallback for development PCs.
- Offline training utility and a deterministic heuristic fallback, so the system starts without a labelled incident dataset.
- Unit tests for features, classification, and alert escalation.

## Architecture

An [experimental combined detector](classifier/COMBINED_DETECTOR.md) can
review agreement between the saved sequence classifier and the existing
edge/fog risk signal on the same recording. It produces offline
candidate episodes and is not connected to the operational alert or GPIO.

```
camera/video -> edge feature extractor -> HTTP JSON -> fog classifier/alerts -> dashboard + GPIO alarm
```

No raw frames are sent to the fog service. Only timestamped metrics are transmitted.

## Fast local demo

Use Python 3.10+ in a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn fog.app:app --host 0.0.0.0 --port 8000
```

In another terminal:

```powershell
python -m edge.main --simulate --fog-url http://127.0.0.1:8000
```

Open `http://127.0.0.1:8000`. The simulator deliberately moves through every risk tier so the UI and GPIO alarm can be verified without a camera.

## Real edge setup

On the edge Pi, install dependencies and run one of:

```bash
python -m edge.main --source 0 --fog-url http://FOG_PI_IP:8000 --model yolov8n.pt
python -m edge.main --source /home/pi/crowd.mp4 --fog-url http://FOG_PI_IP:8000 --model yolov8n.pt
```

The first YOLO run downloads `yolov8n.pt`; do that while the Pi has internet, then retain the model locally. Use `--no-yolo` only for plumbing tests: it does not generate meaningful people counts.

## Raspberry Pi wiring

Use physical-board BCM GPIO numbers, a common ground, and **one 220–330 ohm resistor in series with each LED**:

| Part | GPIO | Connection |
| --- | ---: | --- |
| Green LED (normal) | 17 | GPIO 17 -> resistor -> LED anode; cathode -> GND |
| Red LED (high/critical) | 27 | GPIO 27 -> resistor -> LED anode; cathode -> GND |
| Active buzzer | 22 | GPIO 22 -> buzzer signal; buzzer GND -> Pi GND |

This assumes a 3.3 V active buzzer drawing within the Pi GPIO current limit. If your buzzer is 5 V or its current is unknown, use a transistor driver and separate supply; do **not** connect it directly to a GPIO pin.

Run the fog server on the Pi connected to the LEDs/buzzer. The edge Pi only needs a camera/video source and network access to the fog Pi.

## Train a classifier when labels exist

Prepare a CSV with the feature columns in `shared/schemas.py` and reviewed `risk`, `event_id`, `reviewer`, `rights_approved`, `permission_evidence` and `motion_units` fields. Motion units must be `pixels_per_second`, permissions must be verified, and each class needs independent event groups for a held-out split. See `docs/CURATION_UPDATE_2026_10_14.md`. Then run:

```bash
python -m training.train --input data/labelled_windows.csv --output models/risk_classifier.joblib
```

Start fog with `--model models/risk_classifier.joblib` to use it. Until then, the transparent conservative heuristic is intentionally used for end-to-end development.

## Tests

The trained research anomaly model now reaches the existing fog dashboard through a separate, quality-gated API. See [project handoff, demo commands and remaining blockers](docs/PROJECT_HANDOFF.md). It does not supply the four-tier risk classifier or drive GPIO alarms.

An experimental motion-anomaly baseline has been fitted separately from the fog risk classifier. See [first training results](docs/FIRST_TRAINING_RESULTS.md). Its bundles are research-only and must not be used to drive alarms.

CUDA person-detection audits and augmented experimental model fits are also complete. See [GPU results, Kaggle audit and terminal commands](docs/GPU_TRAINING_RESULTS.md). Pretrained inference is not detector fine-tuning, and the supplied dataset contains no annotation files.

Quality-gated offline camera compensation, cut guards and two further reference anomaly fits are complete. See [quality-training progress](docs/QUALITY_TRAINING_PROGRESS.md). Indeterminate windows receive no score/threshold decision, and no experimental bundle is deployed to the fog alarm.

```bash
pytest
```

## Deployment boundary

Transferring Karthikeyan's work? Start with [the detailed teammate takeover checklist](docs/TEAMMATE_TAKEOVER.md). Generated models/media and machine-local review state are deliberately excluded from Git; the checklist explains private artifact transfer and regeneration.

The local video annotation tool is ready: see [launch instructions and review workflow](docs/ANNOTATION_WORKFLOW.md). It separates observable behaviour from danger labels and serves media locally without uploads.

Before use outside a lab: calibrate the monitored area, validate on representative labelled footage, tune thresholds with safety staff, provide redundant human monitoring, and document an escalation procedure. Never test by creating dangerous crowd conditions.
