#!/usr/bin/env python3
"""
Clean existing train/val/test splits after label cleanup.

I/O
---
Inputs (under data/):
    - data/labels_human_merged.csv      (original label table, used for row indices)
    - data/labels_humans_cleaned_v1.csv (cleaned subset of labels)
    - data/splits.json                  (original splits based on old indices)

Outputs (under data/):
    - data/splits_cleaned.json          (same structure, filtered to valid rows)
"""

import json, csv
from pathlib import Path

# ================== I/O ==================
DATA_DIR = Path("data")
LABELS_ORIG = DATA_DIR / "labels_human_merged.csv"     # INPUT
LABELS_CLEAN = DATA_DIR / "labels_humans_cleaned_v1.csv"  # INPUT
SPLITS_ORIG = DATA_DIR / "splits.json"                 # INPUT
SPLITS_OUT  = DATA_DIR / "splits_cleaned.json"         # OUTPUT
# =========================================

def main():
    # Load all rows (preserve original index order)
    rows = []
    with LABELS_ORIG.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Build whitelist of JSON paths still present in the cleaned labels
    kept = set()
    with LABELS_CLEAN.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kept.add(row["json_path"])

    # Load original splits (by index)
    splits = json.loads(SPLITS_ORIG.read_text(encoding="utf-8"))["indices"]

    # Keep only indices corresponding to retained JSON paths
    def keep_idx(i):
        if i < 0 or i >= len(rows):
            return False
        return rows[i]["json_path"] in kept

    splits_clean = {k: [i for i in idxs if keep_idx(i)] for k, idxs in splits.items()}

    # Write cleaned splits
    out = {"indices": splits_clean}
    SPLITS_OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote cleaned splits to: {SPLITS_OUT}")

if __name__ == "__main__":
    main()
