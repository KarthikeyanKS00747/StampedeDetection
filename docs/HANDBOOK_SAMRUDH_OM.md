# Samrudh Om: Raspberry Pi deployment, alarms and system testing

Prepared 11 October 2026. Your job is to turn the software prototype into a repeatable two-device laboratory demonstration and document its performance. Do not test by creating dangerous crowd conditions or present this prototype as a certified safety system.

## Your assignment and equipment

Own OS/network setup, deployment, safe wiring, hardware tests and end-to-end measurements. Coordinate fog behaviour with Samridh; Karthikeyan supplies corrected edge software, approved videos and the eventual trained model.

Recommended initial handover: both Raspberry Pi 4 4GB boards, both suitable power supplies/cables, two boot-storage devices, buzzer, two LEDs, two verified LED resistors and jumper wires. You also need your laptop, shared network access and a breadboard or other secure prototyping arrangement. Check the separate [device handover checklist](DEVICE_HANDOVER.md) before wiring or purchasing anything.

Label boards clearly:

- Pi A: edge feature processing. No alarm components attached.
- Pi B: fog service and LED/buzzer alarm. Hardware stays attached to this board.

Samridh can develop on a laptop while you configure both boards. Once stable, give him access to Pi B for software integration rather than moving alarm wiring repeatedly.

## Safety gate: before powering or wiring

- Verify two suitable power adapters, not just charging cables. Pi 4 requires a good-quality USB-C supply rated 5V/3A; an official 5.1V/3A adapter is an appropriate option. See [official specifications](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/).
- Shut down and disconnect power before changing wiring. Keep boards off conductive surfaces.
- Identify LED polarity and verify resistor values. The existing design assumes one 220–330 ohm series resistor per LED. If the supplied values are unknown, verify them before use.
- Identify the buzzer's type, rated voltage, current and whether it has a driver/signal module. An unknown or 5V bare buzzer must NOT be driven directly from a GPIO output.
- If a driver is required, obtain a suitable transistor/MOSFET module or a verified circuit and associated parts. Have someone competent check it. Inductive devices may need flyback protection; the required circuit depends on the actual device.
- If specifications cannot be established, leave the buzzer disconnected and demonstrate LEDs plus software alerts. A passive buzzer also needs different control from the current on/off `gpiozero.Buzzer` implementation.

The current code automatically initialises GPIO on a Pi. Therefore wire only verified compatible devices, and understand expected startup/shutdown states before launching it.

## 1. Prepare the boards

1. Confirm whether storage already contains valuable data. Back it up before imaging; OS imaging erases the selected device.
2. Install Raspberry Pi OS using official Raspberry Pi Imager. Prefer a supported 64-bit installation for the vision dependency stack; verify package compatibility rather than assuming it.
3. Set explicit usernames, hostnames and SSH access. Do not assume a default `pi` user exists.
4. Join the same trusted LAN. Record hostnames/IP addresses and reserve Pi B's address where possible.
5. Copy the complete project folder onto each board. Use the actual home directory for that board's user.
6. Record OS version, Python version, hostname and power supply rating. Never put passwords in the project/report.

Use [official setup guidance](https://www.raspberrypi.com/documentation/computers/getting-started.html) for OS installation. A screen and micro-HDMI cable are optional if headless setup succeeds.

## 2. Start with fog-only dependencies

From the project directory on Pi B:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'fastapi>=0.115,<1' 'uvicorn[standard]>=0.30,<1' 'httpx>=0.27,<1' 'numpy>=1.26,<3' 'gpiozero>=2.0,<3'
python -m fog.app --host 0.0.0.0 --port 8000
```

This is a minimal initial setup, not a proven Pi installation. A trained model also needs compatible scikit-learn/joblib and its recorded environment. GPIO backends may need OS-specific configuration. If GPIO initialisation fails, the current application can fall back to simulation: successful HTTP responses do NOT prove physical alarms work. Inspect the logs and hardware separately.

The repository installer installs the full vision stack on every board and includes an OS package that may not exist on your OS release. Inspect it before running. Do not resolve errors with random privileged commands or expose the server to the internet.

On your laptop browser open `http://FOG_PI_IP:8000` and `/docs`, replacing the placeholder with Pi B's address. Keep laptops/Pis on the trusted LAN. The app has no public-deployment authentication or TLS configuration.

## 3. Verify communication before vision

Initially run the simulator on Karthikeyan's or your laptop and send to Pi B:

```bash
python -m edge.main --simulate --fog-url http://FOG_PI_IP:8000
```

Use the Python environment containing the project dependencies. This command has no camera requirement. On Pi A it needs OpenCV plus its other imports; install and verify the edge environment first.

Check that Pi B receives changing metrics and Samridh's dashboard displays them. Stop the source and confirm stale/disconnected behaviour. Restart it and confirm reconnection. Do not assume the existing simulator exercises all four tiers; use controlled API inputs from Samridh to test each state.

## 4. LED wiring and expected behaviour

These mappings match `fog/gpio_alarm.py`. BCM numbers are GPIO identities; physical numbers are header positions. Verify header orientation from the [official hardware documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html) before connecting anything.

| Component | BCM GPIO | Physical header pin | Wiring |
| --- | --- | --- | --- |
| Green LED | 17 | 11 | GPIO → verified series resistor → LED anode; cathode → GND |
| Red LED | 27 | 13 | GPIO → verified series resistor → LED anode; cathode → GND |
| Buzzer signal, only after compatibility/driver approval | 22 | 15 | Appropriate verified signal/driver connection, not an unknown load |
| Ground reference | GND | 6, for example | Common reference for verified alarm circuitry |

Do not confuse physical pin 17 with BCM GPIO17. Do not connect an LED without a series resistor. GPIO signal levels are not 5V power outputs.

| Requested state | Green | Red | Buzzer, if verified and alert active |
| --- | --- | --- | --- |
| Normal | On | Off | Off |
| Moderate, fewer than three consecutive Moderate samples | On | Off | Off |
| Third consecutive Moderate sample | On | Off | On |
| High Risk or Critical | Off | On | On |
| Normal following an active alert | On | Off | Off |

This table describes current code, not a validated safety policy. If a verified active buzzer module is unavailable, record buzzer testing as not performed.

## 5. Run actual video only after plumbing works

Karthikeyan must first resolve live/offline motion units and supply a reviewed, rights-cleared test clip. Do not bulk-download the 50 candidate videos.

On Pi A, create a virtual environment, install the compatible edge dependencies and keep a local YOLO model file available. The full `requirements.txt` is the current starting point but has not been verified on your boards. Record installation failures and versions.

```bash
python -m edge.main --source /absolute/path/to/approved_test.mp4 --fog-url http://FOG_PI_IP:8000 --model /absolute/path/to/yolov8n.pt --area-m2 MEASURED_AREA
```

Replace `MEASURED_AREA` with a defensible calibration value. If the clip's physical area is unknown, do not present count/area as measured people per square metre. `--no-yolo` is for plumbing only; it cannot provide valid crowd counts.

Pi 4 CPU throughput is unverified and may not meet the intended latency. If necessary, demonstrate laptop edge processing → Pi fog/alarm and label it honestly as that configuration. A camera is not required for recorded-video development. OpenCV source `0` requires a compatible camera that actually opens; a CSI camera may require a different capture path, so verify support before buying one.

## 6. Measurements and failure tests

Record these for each tested configuration, using an approved clip or synthetic signals:

- Detector/edge processed FPS, sampling rate, image size and model variant.
- Edge computation latency, fog inference/API latency and actual dashboard/physical-alert delay. Report median and slow-tail latency, not only the best sample.
- Actual serialized payload bytes and measured messages per second. Do not copy the document's estimated bandwidth as a measured result.
- CPU/memory use, temperature and power/throttling observations.
- Fog outage/recovery, edge restart, Wi-Fi interruption, malformed requests and replayed queued messages.
- Alarm behaviour at startup, alert, clearance, source loss and service shutdown.
- Offline video timestamps versus wall-clock processing: do not claim faster-than-real-time processing equals real-time playback performance.

Synchronise clocks before one-way cross-device timing, or use measurements that avoid clock-offset errors. Add instrumentation with Karthikeyan/Samridh; the existing code does not provide a full end-to-end measurement suite. The transport spool/replay code needs testing for duplicates, ordering, stale replay and failed retries.

Suggested results columns: test ID, configuration, input, expected outcome, observed outcome, pass/fail, timing, evidence, software version and unresolved limitation. Never substitute simulated inputs for accuracy or real crowd-risk validation.

## 7. Enable boot services last

Service templates in `deploy/` assume `/home/pi/StampedeDetection` and `User=pi`. Replace both with actual user/path values. Set the real fog address, input source and calibrated area. Pi A's template currently assumes a camera; use an approved video source or a deliberately labelled simulator for the demonstration.

Inspect the final service files before copying them into systemd. Then test restart-on-failure, reboot startup, logs and graceful shutdown. A saved video ends and exits normally; the current `Restart=on-failure` does not turn it into a looping demo. Do not claim the template solves that automatically.

## Handover and completion

Give the team labelled boards, wiring photos, actual pin map, component specifications, OS/package versions, startup/service instructions, test results, measured performance and unresolved issues. Keep credentials separate and private.

Your assignment is complete when another teammate can reproduce the recorded-video/simulation demonstration, verify the alarms actually tested, and understand every remaining limitation. This does not mean the system is approved for a real event.
