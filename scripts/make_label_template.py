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
        review_fields = ["risk", "event_id", "reviewer", "review_note", "rights_approved", "permission_evidence"]
        columns = list(dict.fromkeys([*(rows[0].keys() if rows else []), *review_fields]))
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            row.update(risk="", event_id="", reviewer="", review_note="", rights_approved="false", permission_evidence="")
            writer.writerow(row)
    print(f"Wrote {len(rows)} review rows. Complete reviewed risk/event_id/reviewer and verified permission evidence before training.")


if __name__ == "__main__":
    main()
