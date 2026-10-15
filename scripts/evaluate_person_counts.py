"""Compare candidate person-box counts with reviewed visible-head annotations.

This reports frame count error, not box precision/recall or true total people.
Uncountable/draft frames are excluded, never treated as zero people.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from review.head_labeler import DEFAULT_LABELS, DEFAULT_MANIFEST, HeadStore


METHODS = ("full640", "full1280", "tiled")


def evaluate(store: HeadStore, comparisons: list[Path]) -> list[dict]:
    frames = {(row["video"], row["frame"]): row for row in store.frames}
    labels = store.read()
    results = []
    seen = set()
    for comparison in comparisons:
        with comparison.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (row["video"], int(row["frame"]))
                if key in seen:
                    raise ValueError(f"Duplicate video/frame in comparison inputs: {key}")
                seen.add(key)
                frame = frames.get(key)
                if frame is None:
                    raise ValueError(f"Comparison frame not in audit manifest: {key}")
                annotation = labels.get(frame["id"], {})
                if annotation.get("status") != "reviewed":
                    continue
                actual = len(annotation["points"])
                result = {"video": key[0], "frame": key[1],
                          "reviewed_visible_heads": actual}
                for method in METHODS:
                    predicted = int(row[f"{method}_boxes"])
                    result[f"{method}_boxes"] = predicted
                    result[f"{method}_absolute_error"] = abs(predicted - actual)
                results.append(result)
    if not results:
        raise ValueError("No reviewed frames matched the supplied comparison CSVs")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/person_resolution/reviewed_count_errors.csv"))
    args = parser.parse_args()
    try:
        rows = evaluate(HeadStore(args.manifest, args.labels), args.comparison)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Cannot evaluate: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Reviewed frames: {len(rows)}")
    for method in METHODS:
        mae = sum(row[f"{method}_absolute_error"] for row in rows) / len(rows)
        print(f"{method} visible-head count MAE: {mae:.2f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
