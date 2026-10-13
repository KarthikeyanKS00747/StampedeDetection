from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Make human-review labels for extracted feature windows")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.features).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    with Path(args.output).open("w", newline="", encoding="utf-8") as target:
        columns = list(rows[0].keys()) + ["risk", "reviewer", "review_note"] if rows else ["risk", "reviewer", "review_note"]
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            row.update(risk="", reviewer="Karthikeyan", review_note="")
            writer.writerow(row)
    print(f"Wrote {len(rows)} label rows to {args.output}. Set risk to Normal, Moderate, High Risk, or Critical.")


if __name__ == "__main__":
    main()
