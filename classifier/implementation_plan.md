# YOLO Crowd Stampede Detection — Implementation Plan

## Dataset Summary

| Scene | Split | Frames | Labels |
|---|---|---|---|
| 1_Times_Square / View_1 | Train | 379 | ❌ None |
| 1_Times_Square / View_1 | Test | 1,026 | ❌ None |
| 1_Times_Square / View_2, View_3 | — | 0 | ❌ None |
| 2_Las_Vegas | Train | 4,347 | ❌ None |
| 2_Las_Vegas | Test_1–4 | 7,243 | ❌ None |
| 3_Love Parade | Train_1 | 131 | ❌ None |
| 3_Love Parade | Train_2 | 119 | ❌ None |
| **3_Love Parade / Test** | **Test** | **361** | **✅ YOLO Darknet + PASCAL VOC** |
| 4_Italy / View_1, View_2 | — | 0 (empty dirs) | ❌ None |
| **TOTAL** | | **13,606** | **361 labeled** |

**Key findings:**
- Only `3_Love Parade/Test/Label/YOLO_darknet/` contains ready-to-use YOLO labels.
- Class `0` = `Fight` (from PASCAL VOC XML: `<name>Fight</name>`), which maps to "stampede/panic" events.
- All other scenes have **no bounding-box labels** — they cannot be directly used for supervised YOLO object detection without annotation.
- The labeled set (361 frames) is small. We'll use pretrained YOLOv8 weights and heavy augmentation to compensate.

> [!IMPORTANT]
> The 361 labeled frames are the only ones usable for YOLO training as-is. The remaining 13,245 frames have **no bounding-box annotations**. Two options are discussed under "User Review Required".

---

## User Review Required

> [!CAUTION]
> **Critical design decision — please choose an approach:**
>
> **Option A (Implemented here):** Train only on the 361 labeled Love Parade frames (YOLO Darknet format, class = `Fight`). Use 80/20 train/val split. Use other scenes' unlabeled frames as background images (no-object frames) to reduce false positives.
>
> **Option B:** Auto-label the remaining 13,245 frames using a pretrained crowd/person detector (e.g. YOLOv8n-person), then manually review a sample before training on all data. This gives far more training data but requires additional annotation work.
>
> **I will implement Option A unless you specify otherwise.**

---

## Proposed Changes

### Data Preparation Script

#### [NEW] [prepare_dataset.py](file:///c:/Users/samri/OneDrive/Desktop/archive/prepare_dataset.py)

Reads the existing YOLO Darknet labels from `3_Love Parade/Test/Label/YOLO_darknet/` and the corresponding frames from `3_Love Parade/Test/`, then:
1. Splits them 80/20 into `dataset/images/train` + `dataset/images/val`.
2. Copies matching [.txt](file:///c:/Users/samri/OneDrive/Desktop/archive/Abnormal%20High-density%20Crowds/Read%20Me.txt) label files into `dataset/labels/train` + `dataset/labels/val`.
3. Optionally copies a random sample of **unlabeled frames** from other scenes into `dataset/images/train` as **background images** (no corresponding label file = YOLO skips them as negatives).

---

### Dataset YAML Config

#### [NEW] [dataset.yaml](file:///c:/Users/samri/OneDrive/Desktop/archive/dataset.yaml)

```yaml
path: ./dataset
train: images/train
val:   images/val

nc: 1
names: ['stampede']
```

---

### Training Script

#### [NEW] [train.py](file:///c:/Users/samri/OneDrive/Desktop/archive/train.py)

- Uses **Ultralytics YOLOv8** (`pip install ultralytics`)
- Loads `yolov8n.pt` (nano) as pretrained weights — good for limited data
- Key hyperparameters:
  - `epochs=100`, `imgsz=640`, `batch=16`
  - `augment=True`, `degrees=5`, `flipud=0.5`, `mosaic=1.0`
- Saves results to `runs/detect/stampede/`

---

### Inference Demo Script (optional)

#### [NEW] [inference.py](file:///c:/Users/samri/OneDrive/Desktop/archive/inference.py)

Runs the trained model on a video or image folder and renders bounding boxes with confidence scores. Shows a simple CLI interface.

---

## Verification Plan

### Automated Checks
```powershell
# 1. After prepare_dataset.py — verify counts
python prepare_dataset.py
# Expected output: "Train: ~288 images, Val: ~73 images"
Get-ChildItem .\dataset\images\train | Measure-Object   # ~288
Get-ChildItem .\dataset\labels\train | Measure-Object   # ~288 (same count)

# 2. Verify label format (spot check)
Get-Content .\dataset\labels\train\scene-00001.txt
# Expected: "0 <cx> <cy> <w> <h>"  all values 0–1

# 3. Run training (requires GPU/CPU)
python train.py
# Expected: training starts, metrics printed per epoch, model saved to runs/detect/stampede/weights/best.pt
```

### Manual Verification
1. After training completes, open `runs/detect/stampede/results.png` — check that loss curves decrease and mAP50 > 0.3 (given tiny dataset, this is reasonable).
2. Run `python inference.py --source "path/to/test/image.png"` and verify a bounding box appears on crowd panic frames.
