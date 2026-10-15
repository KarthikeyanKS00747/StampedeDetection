"""
package_deployment.py
---------------------
Assembles the exact-matched deployment package for experimental integration.

Creates deploy/ containing:
  deploy/
    infer.py                  <- inference entry point
    extract_features.py       <- feature extractor (must match training exactly)
    yolov8n.pt                <- exact person detector weights
    model/
      classifier.pkl
      feature_order.txt
      metadata.json
    requirements.txt
    TRIGGER_POLICY.md
    INTEGRATION_NOTES.md
    SHA256SUMS.json

Usage
-----
  python package_deployment.py
  # Output: deploy/ folder ready to copy to integration environment
"""

import hashlib
import json
import shutil
import ast
from pathlib import Path

BASE = Path(__file__).parent
DEPLOY = BASE / "deploy"

FILES = [
    ("infer.py",            "infer.py"),
    ("extract_features.py", "extract_features.py"),
    ("yolov8n.pt",          "yolov8n.pt"),
    ("requirements.txt",    "requirements.txt"),
    ("TRIGGER_POLICY.md",   "TRIGGER_POLICY.md"),
    ("INTEGRATION_NOTES.md", "INTEGRATION_NOTES.md"),
]
MODEL_FILES = [
    "classifier.pkl",
    "feature_order.txt",
    "metadata.json",
]


def main():
    required = [BASE / src for src, _ in FILES]
    required += [BASE / "model" / name for name in MODEL_FILES]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing package inputs: " + ", ".join(missing))
    DEPLOY.mkdir(exist_ok=True)

    # Copy scripts
    for src_name, dst_name in FILES:
        src = BASE / src_name
        shutil.copy2(src, DEPLOY / dst_name)
        print(f"Copied {src_name}")

    # Copy model directory
    model_src = BASE / "model"
    model_dst = DEPLOY / "model"
    model_dst.mkdir(exist_ok=True)
    for f in MODEL_FILES:
        src = model_src / f
        shutil.copy2(src, model_dst / f)
        print(f"Copied model/{f}")

    # Verify feature_order matches the actual extractor declaration.
    feat_order = (model_dst / "feature_order.txt").read_text().strip().splitlines()
    extractor_ast = ast.parse((DEPLOY / "extract_features.py").read_text())
    extractor_features = next(
        ast.literal_eval(node.value) for node in extractor_ast.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "FEATURE_NAMES"
                for target in node.targets)
    )
    if feat_order != extractor_features:
        raise ValueError("Feature order differs from packaged extractor")
    print("Verified: model feature order matches packaged extractor")

    package_files = [DEPLOY / dst for _, dst in FILES]
    package_files += [model_dst / name for name in MODEL_FILES]
    checksums = {
        str(path.relative_to(DEPLOY)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest()
        for path in package_files
    }
    (DEPLOY / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2) + "\n")

    # Print metadata summary
    meta = json.loads((model_dst / "metadata.json").read_text())
    print(f"\nModel metadata:")
    print(f"  Trained at      : {meta.get('trained_at')}")
    print(f"  sklearn version : {meta.get('sklearn_version')}")
    print(f"  Training scenes : {meta.get('training_incidents')}")
    print(f"  Holdout         : {meta.get('holdout_incident')}")

    print(f"\nDeployment package ready -> {DEPLOY}/")
    print("Use this matched extractor/weight/model set for experimental integration only.")


if __name__ == "__main__":
    main()
