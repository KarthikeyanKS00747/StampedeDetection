# Local experimental artifact bundle

The local bundle is `outputs/handoff/experimental_ml_bundle.zip`, excluded from Git. Extract into the repository root to preserve the paths expected by the demonstration commands.

Contents:

- `models/yolo11s.pt`: downloaded official pretrained detector, not fine-tuned weights.
- `outputs/training/quality_motion_2026_10_14_run04/quality_joint.joblib`: fitted research anomaly model, not four-tier risk weights.
- `outputs/training/quality_motion_2026_10_14_run04/features_quality.json`: 70 cached windows for replay.
- `outputs/training/quality_motion_2026_10_14_run04/report.json`: original experiment metadata/limitations.
- `outputs/completion/integration_report_20261014.json`: inference/API verification, not accuracy.
- `ARTIFACT_README.md`: recorded environment and safety boundary.

No videos, local source paths or reviewer saves are intentionally packaged. Load only trusted joblib artifacts: they can execute code. Permissions for the reference footage remain unresolved; `deployable=false` and `rights_verified=false` still apply.
