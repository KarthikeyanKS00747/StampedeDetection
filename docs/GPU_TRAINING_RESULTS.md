# GPU-assisted training experiment and next steps

Completed 14 October 2026 on the NVIDIA GeForce RTX 4070 Laptop GPU, with CUDA available in PyTorch 2.14.1+cu126. No Raspberry Pi was needed.

## What actually ran

The official pretrained YOLO11s detector processed 403 sampled video frames across L002, L003, L009 and L010 at an approximately 5 FPS sampling rate, using CUDA device 0 and image size 1024. ByteTrack supplied provisional track motion. The experiment extracted detected count, image-space bounding-box coverage, maximum count in a 4×4 image grid and track-speed proxies.

Two runs completed:

- Standard full-frame detection: about **25.2 seconds** of processing after model loading.
- Four overlapping image tiles, merged using NMS: about **65.0 seconds** after model loading.

Each run combined the GPU-extracted features with 70 existing motion windows and fitted a separate Isolation Forest baseline on the CPU. Each forest fitted 26 reference windows and calibrated on 14 later same-scene reference windows with a temporal gap. The saved models were reloaded and scored all 70 windows successfully.

**GPU usage here was real detector inference/feature extraction. YOLO weights were not fine-tuned.** The anomaly forest is a CPU model. There are no reviewed risk-class or bounding-box training labels yet. No four-level risk classifier was trained or connected to the alarm.

## Detection audit

These are detector outputs, not true crowd counts:

| Clip | Sampled frames | Median detections, full frame | Median detections, tiled |
| --- | ---: | ---: | ---: |
| L002: `2.mp4` | 140 | 41 | 72 |
| L003: `3.mp4` | 140 | 28 | 63 |
| L009: `Video Project 10.mp4` | 60 | 11 | 29 |
| L010: `Video Project 11.mp4` | 63 | 1 | 3 |

Full-frame detection produced fewer detections in the denser-looking corridor clip. The inspected overlays show missed distant/occluded people. Tiling produced additional boxes, but still misses much of the distant crowd and can leave overlapping/fragment boxes. A higher count is not proof of improved counting accuracy. L010's later very wide, low-quality view remains particularly unsuitable for person-detection counts.

Median synchronized GPU-assisted predict-call times were approximately 14–34 ms per full frame, versus 86–94 ms per tiled frame. These calls include preprocessing/postprocessing; they are not pure GPU kernel timings, end-to-end alarm latency or Raspberry Pi performance. The first invocation per clip was excluded from these timing medians.

## Anomaly-model diagnostics

| Probe | Full-frame augmented model | Tiled augmented model |
| --- | ---: | ---: |
| L009 | 1/12 windows above reference threshold | 4/12 |
| L010 | 2/12 | 4/12 |

These are **threshold exceedances, not accuracy, recall, risk probabilities or confirmed dangerous windows**. The probes have no reviewed ground truth. Both reference clips appear to share a scene; the temporal calibration is not independent-event validation. Scene and camera differences can drive scores. No variant is selected for deployment.

## Kaggle source check

The supplied [dataset](https://www.kaggle.com/datasets/ithikashr/stampede-model), owner `ithikashr`, version 1, lists **MIT** in its public metadata. The complete public file inventory contains **17 MP4 files**, about 446.7 MB in total. No CSV/JSON labels, YOLO TXT annotations, dataset YAML, class folders or documented train/test split were found in that inventory. The description is empty.

Evidence was saved in `data/review/kaggle_source_audit.json`, using the public [metadata endpoint](https://www.kaggle.com/api/v1/datasets/view/ithikashr/stampede-model) and [file-list endpoint](https://www.kaggle.com/api/v1/datasets/list/ithikashr/stampede-model). Local `2.mp4` and `3.mp4` match listed names and byte sizes, which supports the provenance statement but is not a remote hash verification. `Video Project 10.mp4` and `11.mp4` are not listed by those names; their original-file mapping is still unresolved.

The uploader's MIT selection does not independently establish rights to visible third-party news/Getty material. No clips were redownloaded. Getty-watermarked clips were excluded from these experiments, and upstream rights remain unverified. The earlier requested dataset link has now been received and checked; it should not be requested again.

## Files to keep

Completed runs are in:

- `outputs/training/gpu_person_2026_10_14_run02/`
- `outputs/training/gpu_person_2026_10_14_run03_tiled/`

Each contains person-frame features, combined window features, per-window scores, a report, 12 annotated audit images and `motion_person_anomaly.joblib`. These experimental bundles have `deployable=false`, no risk classes and no physical density. Do not pass them to the existing fog classifier. The failed initial GPU directory is retained separately; it contains no completed model.

Weights are cached at `models/yolo11s.pt`. The dependency constraint now keeps supervision below 0.31 because the current ByteTrack API is deprecated and scheduled for removal in 0.31. This avoids that known compatibility break; it is not a complete dependency lock.

26 focused tests pass. The full fog integration suite remains unverified following its previously stalled run. Original footage and teammates' fog changes were preserved.

## Terminal command

You do not need to rerun the completed experiments. For a new run on these same four clips, from the project root:

```powershell
python -m training.gpu_person_baseline --output-dir outputs/training/gpu_manual_run --device 0 --imgsz 1024 --tiled
```

Choose a fresh output directory each time; existing directories are deliberately protected. The manifest still points to the four local files and the first motion run; this command does not automatically ingest arbitrary new clips or train labelled risk classes.

Alternatively, the wrapper generates a timestamped output folder:

```powershell
.\scripts\run_gpu_baseline.ps1 -Tiled
```

If script execution is restricted, use the Python command rather than weakening system-wide execution policy. If CUDA runs out of memory, retry in a new output folder with `--imgsz 640`; expect poorer small-person detection and review it separately. Do not silently switch to CPU and call it a GPU result.

## What is required for meaningful supervised training

1. Review a small set of clean original frames and annotate real person/head boxes, or use a genuinely annotated, appropriately licensed dataset. Detector-produced boxes are proposals, not independent ground truth. With heads-only labels, the detector task/class must explicitly become head detection.
2. Inspect uninterrupted video windows and annotate observable behaviour plus uncertainty. Include quiet dense crowds, local congestion, running, cheering, flashes and camera motion. Do not label all files 'stampede' from the dataset title.
3. Group related crops and source events together. Add independent events for each target class; reserve a final event-level test before fitting or tuning.
4. First compare counting recall/error on annotated frames and simple supervised feature baselines. Only fine-tune a detector on the GPU when enough verified annotations exist. Keep temporal neural models optional until there is adequate independent data.

This is the main remaining data requirement. Additional epochs or a larger GPU cannot supply missing ground truth. The current overlays and diagnostics identify the failures to fix before spending hours on a training job.

Detector method references: [YOLO11](https://docs.ultralytics.com/models/yolo11/) and [prediction parameters](https://docs.ultralytics.com/modes/predict/).
