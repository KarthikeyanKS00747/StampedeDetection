# Local behaviour review

Updated 15 October 2026. This tool keeps videos on your computer: no uploads, remote assets or publishing.

## Actual dataset status

- 50 discovered source links, not 50 downloaded training videos.
- 1 catalogue download: V050, `data/raw/commons/GPD_RIOT_DRILL_2009.webm`. The [Commons source page](https://commons.wikimedia.org/wiki/File:GPD_RIOT_DRILL_2009.webm) lists SOFISINTOWN and CC BY 3.0. Attribution is retained in `data/raw/commons/ATTRIBUTION.md`. This is a controlled riot drill, not a confirmed stampede example.
- 10 user-supplied local clips; 4 used in the current experiment.
- 70 short review windows, 50 numerically quality-eligible and 20 indeterminate. Numerical eligibility is not proof of camera stability or accurate detections.
- No real reviewed observations or supervised behaviour/risk classifier yet.

Pexels' [terms](https://www.pexels.com/terms-of-service/) restrict unauthorised automatic ML extraction and systematic copying. Public YouTube availability alone does not establish training permission. Record a suitable source-specific licence or permission rather than assuming a free download grants all reuse rights.

## Launch

In a terminal in the project root, with `data/review/annotation_package.json`
present:

```powershell
python -m review.app --port 8011
```

Open `http://127.0.0.1:8011/`. Stop the server with Ctrl+C when finished. It binds only to loopback and serves only the four package-listed videos.

## Review each window

1. Press **Play this window**; repeat if necessary. Check what people actually do, not what an event title claims.
2. Choose stationary, walking, running, mixed movement or uncertain. Describe camera motion, lights, cuts and occlusion in the observation note.
3. Set Reviewed observation only when a definite behaviour is visible; enter your reviewer name and note. Otherwise use Uncertain or Exclude. Save explicitly before moving on.
4. Export reviewed data. Only definite reviewed behaviours with reviewer, note and eligible features are exported. Empty exports are expected until these requirements are met.

The form no longer asks repeated source/licensing questions. Saved records and
exports carry the user's 15 October 2026 statement that source permission is
held; the app has not independently verified it. Existing test fixtures and
training validation may still contain provenance fields.

Annotations persist in `data/review/behaviour_annotations.json`, created on the first save. Revision checks protect against silently overwriting another browser's edits. The source package is `data/review/annotation_package.json`. This video-window website does not accept or save still images; use the separate [head-point labeler](../classifier/DENSITY_MODEL.md#prepare-annotations) for audit-frame labels and marked-image downloads. The source package is not currently present in this workspace, so regenerate it before launching this video review site.

## Training boundary

The export schema is `reviewed_behaviour_features_v1`, with nine image-based features. It is **not** input for the existing seven-feature four-tier risk trainer, and no automatic conversion creates risk labels. The separate `training.train_behaviour` trainer is now implemented: see [project handoff](PROJECT_HANDOFF.md). Actual supervised fitting and independent event-level evaluation still require sufficient reviewed data. Adjacent windows and the two shared-corridor clips are not independent events.

Pretrained GPU person detection and experimental anomaly fits already exist; no YOLO fine-tuning or validated stampede classifier is claimed. Keep all experimental models disconnected from alarms.

## Verification

On 15 October 2026, 89 project tests passed, 4 skipped. These tests used synthetic labels; they do not verify real annotation accuracy or licensing.
