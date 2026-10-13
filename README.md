# Single Camera Edge Fog Stampede Early Warning

This project implements a privacy-preserving, single-camera early-warning system for abnormal crowd dynamics.  It is a decision-support prototype, **not a stampede predictor** and must not be used as the only safety control at a live event.

## What is included

- Edge pipeline: video/camera ingestion, optional YOLO person detection and ByteTrack tracking, Farneback optical flow, six crowd features, and durable HTTP delivery.
- Fog service: FastAPI REST and WebSocket API, rolling history, four-level risk classification, escalation rules, alert log, and a live browser dashboard.
- Hardware alarm: Raspberry Pi GPIO control for one buzzer and two LEDs (green and red), including a safe simulation fallback for development PCs.
- Offline training utility and a deterministic heuristic fallback, so the system starts without a labelled incident dataset.
- Unit tests for features, classification, and alert escalation.

## Architecture

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

Prepare a CSV with `risk` plus the feature columns listed in `shared/schemas.py`, then run:

```bash
python -m training.train --input data/labelled_windows.csv --output models/risk_classifier.joblib
```

Start fog with `--model models/risk_classifier.joblib` to use it. Until then, the transparent conservative heuristic is intentionally used for end-to-end development.

## Tests

```bash
pytest
```

## Deployment boundary

Before use outside a lab: calibrate the monitored area, validate on representative labelled footage, tune thresholds with safety staff, provide redundant human monitoring, and document an escalation procedure. Never test by creating dangerous crowd conditions.
