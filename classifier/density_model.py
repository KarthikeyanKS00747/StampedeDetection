"""Experimental, independent head-density estimator (not used by the alarm).

Each JSONL annotation is {"image": "path.jpg", "scene": "camera-or-video-id",
"points": [[head_x, head_y], ...]}. Coordinates are in the original image.
Train and validation manifests must contain different scenes and images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


IMAGE_SIZE = 512
OUTPUT_STRIDE = 4


class DensityNet(nn.Module):
    """Small stride-4 CNN; output cells represent estimated people, not boxes."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1), nn.Softplus(),
        )
        nn.init.constant_(self.features[-2].bias, -8.0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.features(image)


def letterbox(image: np.ndarray, size: int = IMAGE_SIZE) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    scaled_width = min(size, max(1, round(width * scale)))
    scaled_height = min(size, max(1, round(height * scale)))
    resized = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
    left = (size - scaled_width) // 2
    top = (size - scaled_height) // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top:top + scaled_height, left:left + scaled_width] = resized
    return canvas, scale, left, top


def density_target(points: list[list[float]], width: int, height: int,
                   scale: float, left: int, top: int) -> torch.Tensor:
    """Place a unit-mass Gaussian at each annotated head; sum equals count."""
    side = IMAGE_SIZE // OUTPUT_STRIDE
    target = np.zeros((side, side), dtype=np.float32)
    sigma = 1.5
    radius = 5
    for x, y in points:
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Head point {(x, y)} outside {width}x{height} image")
        cx = (x * scale + left) / OUTPUT_STRIDE
        cy = (y * scale + top) / OUTPUT_STRIDE
        xmin, xmax = max(0, int(cx) - radius), min(side, int(cx) + radius + 1)
        ymin, ymax = max(0, int(cy) - radius), min(side, int(cy) + radius + 1)
        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
        kernel = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()  # Preserve one person even near an image edge.
        target[ymin:ymax, xmin:xmax] += kernel
    return torch.from_numpy(target[None, :, :])


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("scene"), str) or not row["scene"]:
                raise ValueError(f"{path}:{line_number}: scene is required")
            if not isinstance(row.get("points"), list):
                raise ValueError(f"{path}:{line_number}: points list is required")
            image = (path.parent / row["image"]).resolve()
            if not image.is_file():
                raise FileNotFoundError(f"{path}:{line_number}: {image}")
            row["image"] = image
            rows.append(row)
    if not rows:
        raise ValueError(f"No annotations in {path}")
    return rows


class HeadPointDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = cv2.imread(str(row["image"]))
        if image is None:
            raise ValueError(f"Cannot read {row['image']}")
        height, width = image.shape[:2]
        canvas, scale, left, top = letterbox(image)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        inputs = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
        target = density_target(row["points"], width, height, scale, left, top)
        return inputs, target


@torch.no_grad()
def evaluate(model: DensityNet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    errors = []
    for images, targets in loader:
        predicted = model(images.to(device)).sum(dim=(1, 2, 3)).cpu()
        actual = targets.sum(dim=(1, 2, 3))
        errors.extend((predicted - actual).abs().tolist())
    return float(np.mean(errors))


def train(args: argparse.Namespace) -> None:
    train_rows = read_manifest(args.train)
    val_rows = read_manifest(args.val)
    train_scenes = {row["scene"] for row in train_rows}
    val_scenes = {row["scene"] for row in val_rows}
    if train_scenes & val_scenes:
        raise ValueError(f"Train/validation scenes overlap: {train_scenes & val_scenes}")
    if {row["image"] for row in train_rows} & {row["image"] for row in val_rows}:
        raise ValueError("Train/validation images overlap")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = DensityNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(HeadPointDataset(train_rows), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(HeadPointDataset(val_rows), batch_size=1, num_workers=0)
    best_mae = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            predicted = model(images)
            pixel_loss = F.mse_loss(predicted, targets, reduction="sum") / len(images)
            count_loss = F.mse_loss(predicted.sum(dim=(1, 2, 3)),
                                    targets.sum(dim=(1, 2, 3)))
            loss = pixel_loss + count_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_mae = evaluate(model, val_loader, device)
        print(f"epoch={epoch} train_loss={total_loss / len(train_loader):.3f} "
              f"val_count_mae={val_mae:.3f}")
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({"state_dict": model.state_dict(), "image_size": IMAGE_SIZE,
                        "stride": OUTPUT_STRIDE, "val_count_mae": val_mae,
                        "train_scenes": sorted(train_scenes),
                        "val_scenes": sorted(val_scenes)}, args.output)
    print(f"Best checkpoint: {args.output}; validation count MAE={best_mae:.3f}")


@torch.no_grad()
def predict(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    if checkpoint.get("image_size") != IMAGE_SIZE or checkpoint.get("stride") != OUTPUT_STRIDE:
        raise ValueError("Checkpoint preprocessing does not match this model")
    model = DensityNet()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"Cannot read {args.image}")
    canvas, _, _, _ = letterbox(image)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    inputs = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None] / 255.0
    density = model(inputs)[0, 0].numpy()
    count = float(density.sum())
    heat = cv2.resize(density, (IMAGE_SIZE, IMAGE_SIZE))
    heat = np.uint8(255 * heat / max(float(heat.max()), 1e-8))
    overlay = cv2.addWeighted(canvas, 0.65, cv2.applyColorMap(heat, cv2.COLORMAP_JET),
                              0.35, 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), overlay):
        raise RuntimeError(f"Cannot write {args.output}")
    print(json.dumps({"estimated_count": round(count, 2),
                      "validation_count_mae": checkpoint["val_count_mae"],
                      "overlay": str(args.output)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train", help="Train from manually annotated head points")
    training.add_argument("--train", type=Path, required=True)
    training.add_argument("--val", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--batch-size", type=int, default=2)
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", default="cpu")
    prediction = sub.add_parser("predict", help="Estimate count with a trained checkpoint")
    prediction.add_argument("--weights", type=Path, required=True)
    prediction.add_argument("--image", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
            parser.error("epochs, batch size, and learning rate must be positive")
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
