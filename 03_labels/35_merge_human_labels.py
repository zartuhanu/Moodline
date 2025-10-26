#!/usr/bin/env python3
"""
Merge human-provided labels into the current label table.

Inputs (under data/):
    data/labels_merged.csv   – current labels after auto merges
    data/human_labels.csv    – manual annotations (must have 'json_path' and
                               'human_mood' or 'human_label')

Output:
    data/labels_human_merged.csv
        - 'mood' overridden by human labels when provided
        - 'source' set to 'manual' for overridden rows
        - optional dropping of rows labeled 'no_bass'
"""

import csv
from pathlib import Path

LABELS_IN   = Path("data/labels_merged.csv")     # current merged labels file
HUMAN_IN    = Path("data/human_labels.csv")      # manual sheet
LABELS_OUT  = Path("data/labels_human_merged.csv")
DROP_NO_BASS = True   # set False to keep rows explicitly labeled 'no_bass'

def main():
    if not LABELS_IN.exists():
        raise SystemExit(f"Missing {LABELS_IN}")
    if not HUMAN_IN.exists():
        raise SystemExit(f"Missing {HUMAN_IN}")

    # Load human labels → map by json_path
    human_map = {}
    with HUMAN_IN.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        # Accept either 'human_mood' or 'human_label'
        col = "human_mood" if "human_mood" in (r.fieldnames or []) else "human_label"
        if col not in (r.fieldnames or []) or "json_path" not in (r.fieldnames or []):
            raise SystemExit("human_labels.csv must include 'json_path' and 'human_mood' (or 'human_label').")
        for row in r:
            jp = (row.get("json_path") or "").strip()
            hm = (row.get(col) or "").strip()
            if not jp or not hm:
                continue  # skip blanks
            human_map[jp] = hm

    # Read labels_merged and apply overrides
    out_rows = []
    dropped = 0
    with LABELS_IN.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        # Keep existing columns; ensure 'source' and 'human_mood' exist
        fieldnames = list(r.fieldnames or [])
        for extra in ("source", "human_mood"):
            if extra not in fieldnames:
                fieldnames.append(extra)

        for row in r:
            jp = row.get("json_path", "")
            if jp in human_map:
                hm = human_map[jp]
                # Optionally drop rows labeled 'no_bass'
                if DROP_NO_BASS and hm.lower() == "no_bass":
                    dropped += 1
                    continue
                # Override mood with human label
                row["mood"] = hm
                row["source"] = "manual"
                row["human_mood"] = hm
            else:
                # Keep original; ensure 'human_mood' key is present
                row["source"] = row.get("source", "")
                row.setdefault("human_mood", "")
            out_rows.append(row)

    # Write output
    with LABELS_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            # Ensure only expected fields are written
            clean = {k: row.get(k, "") for k in fieldnames}
            w.writerow(clean)

    print(f"Wrote {LABELS_OUT} (rows kept: {len(out_rows)})")
    if dropped:
        print(f" Dropped {dropped} row(s) labeled as no_bass")

if __name__ == "__main__":
    main()
