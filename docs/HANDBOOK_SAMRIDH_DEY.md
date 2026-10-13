# Samridh Dey: Fog service, dashboard and alerts

Prepared 11 October 2026. Proposed responsibility split for Karthikeyan KS, Samridh Dey and Samrudh Om. This is a development handbook, not a claim that the prototype is validated for crowd safety.

## Your assignment

Own the software after the edge produces numerical features: API ingestion, classifier integration, browser dashboard and alert behaviour. Build on the existing implementation. Karthikeyan owns video features, dataset labels and training. Samrudh owns the Pis, wiring and hardware/performance tests.

Your final deliverable is a fog service that accepts agreed feature vectors, clearly identifies simulation/heuristic versus trained-model output, updates the dashboard and alert log, and handles interruptions without showing stale information as live.

## Devices and files you need

- Your laptop, Windows or Linux, with Python 3.10+ and a browser. No GPU or Raspberry Pi is needed for initial work.
- A copy of the complete project folder, not only `fog/`. It needs `shared/`, `edge/` for simulation, and tests.
- Internet for installing packages; a trusted shared LAN for integration later.
- Later, access to the fog Pi configured by Samrudh. Do not independently change its wiring or OS.
- From Karthikeyan later: model file, feature order/units, dependency versions and evaluation notes. No trained model exists yet.

## Existing code and limitations

| File | Purpose |
| --- | --- |
| `fog/app.py` | FastAPI routes, WebSocket updates, in-memory state |
| `fog/classifier.py` | Model loading and temporary heuristic classifier |
| `fog/alerts.py` | Immediate High/Critical alerts and sustained Moderate alert |
| `fog/static/` | Browser interface |
| `shared/schemas.py` | Authoritative request/response schema and feature order |
| `tests/test_fog.py` | Two basic classifier/alert tests |

The baseline unit suite has four tests. It is not an API, browser or hardware validation suite. The alert list is fetched once on page load; it is not currently kept live. History is available through an endpoint but the promised time-series charts are not implemented. Data is in memory and is lost on restart. Heuristic scores currently include placeholder probability values: never present these as measured confidence or calibrated probabilities.

## First laptop run

Work from your own project copy. On Windows PowerShell, replace the folder below with your actual project location:

```powershell
cd C:\path\to\StampedeDetection
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m fog.app --host 127.0.0.1 --port 8000
```

Keep that terminal open. In a second terminal in the same project folder:

```powershell
.\.venv\Scripts\python.exe -m edge.main --simulate --fog-url http://127.0.0.1:8000
```

Open `http://127.0.0.1:8000` and `http://127.0.0.1:8000/docs`. Stop processes with Ctrl+C. GPIO is log-only on a normal laptop. Installing the complete requirements may be large because it includes vision libraries; a dedicated fog-only dependency list is a useful follow-up task.

Do not assume the simulator visits every risk tier. Test each tier explicitly with controlled feature requests through the interactive API documentation. Synthetic inputs verify plumbing, not model accuracy.

## Work sequence

### 1. Establish the baseline

- Run the service, simulator and four existing tests.
- Save screenshots of incoming metrics and an alert, noting that the source is simulation.
- Check `GET /api/status`, `/api/history/zone-1` and `/api/alerts`.
- Confirm malformed/negative fields receive validation errors rather than corrupting state.

### 2. Complete the operator interface

- Make alert creation and clearance appear without reloading the page. Use a documented event message or polling strategy.
- Add density and velocity-variance history charts with timestamps and units.
- Display source type and classifier mode prominently. Heuristic output must say unvalidated heuristic, not trained prediction.
- Add last-update age, disconnected/reconnecting state and a configurable stale-data timeout.
- On stale input, show unknown/stale. Do not silently report Normal or automatically clear a real alert merely because packets stopped.
- Check WebSocket reconnection and restart behaviour. Initialise the page from stored current state without duplicating zones.
- Escape text and keep the interface understandable on a phone as well as a laptop.

### 3. Strengthen API and alert handling

- Add integration tests for feature ingestion, status/history endpoints and WebSocket updates.
- Test Normal, Moderate, High Risk, Critical, alert clearance and repeated Critical requests. Current code may log repeated Critical events; agree on deduplication/rate limiting.
- Test multiple zone IDs. The current physical alarm follows the most recently ingested zone; for multi-zone use, aggregate the highest active severity or explicitly limit deployment to one zone.
- Make timestamp ordering and stale/replayed data handling explicit. Old packets must not overwrite fresh display or activate physical alarms as if current.
- Agree with Samrudh on the alarm behaviour when a source disconnects or the fog service restarts.
- Keep the prototype on a trusted LAN. There is no authentication/TLS implementation; do not expose port 8000 publicly.

### 4. Integrate the trained classifier when available

The supported command is:

```powershell
.\.venv\Scripts\python.exe -m fog.app --model models/risk_classifier.joblib --host 127.0.0.1 --port 8000
```

Run `python -m fog.app`, not plain `uvicorn fog.app:app`, when relying on the current `--model` argument. The CLI configures the model before starting the server.

- Load only a trusted model supplied by Karthikeyan; joblib files can execute code during loading.
- Confirm the exact order in `FEATURE_COLUMNS`: crowd_count, density, density_growth_rate, average_velocity, velocity_variance, direction_variance, congestion_index.
- Agree on units before integration. Offline extraction currently uses pixels/second while live edge code uses displacement per frame. This must be fixed by Karthikeyan before claiming live model validity.
- Show absent/unavailable model status clearly. Do not silently make heuristic output look like the trained model.
- Measure fog inference separately from API/network and dashboard-update latency.

## Required acceptance checks

| Check | Evidence to hand over |
| --- | --- |
| Valid input displays correct zone/count/features | API request, response and screenshot |
| Invalid input is rejected | Automated test and response code |
| Alerts and clearances appear live | Test sequence and browser recording/screenshots |
| History charts match ingested data | Known sequence and timestamp comparison |
| Disconnect/reconnect shows stale state correctly | Test log and screenshots |
| Trusted model is loaded with agreed feature order | Startup log and known test vector |
| Restart behaviour is documented | What persists and what is lost |

Record pass/fail and the tested software versions. Do not mark hardware or trained-model checks complete using simulation evidence.

## Handover to teammates

Give Samrudh: tested project copy, startup command, API schema, fog port, stale-data policy, alarm expectations and test report. Give Karthikeyan: final feature contract, model-interface requirements and any integration failures. Share changes through the team's agreed repository/file-transfer method; avoid independently overwriting the same files.

You do not own dataset licensing, risk-label creation or accuracy claims. You may help review clips, but coordinate labels with Karthikeyan.
