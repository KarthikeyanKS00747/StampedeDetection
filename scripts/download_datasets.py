from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already present: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "StampedeDetection academic-project/1.0"})
    with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        received = 0
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            received += len(chunk)
            print(f"Downloaded {received / 1_000_000:.1f} MB" + (f" / {total / 1_000_000:.1f} MB" if total else ""), end="\r")
    temporary.replace(destination)
    print(f"\nSaved {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download only datasets explicitly listed in the project manifest")
    parser.add_argument("--dataset", default="umn_crowd_activity")
    parser.add_argument("--manifest", default="data/datasets.json")
    args = parser.parse_args()
    items = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["datasets"]
    selected = next((item for item in items if item["id"] == args.dataset), None)
    if selected is None:
        raise SystemExit(f"Unknown dataset {args.dataset}")
    print(selected["display_name"])
    print(selected["use"])
    download(selected["url"], Path(selected["output"]))


if __name__ == "__main__":
    main()
