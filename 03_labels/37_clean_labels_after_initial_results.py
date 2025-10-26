#!/usr/bin/env python3
"""
Clean and consolidate the final label table to improve dataset consistency.

I/O
---
Inputs (under data/):
    - data/labels_human_merged.csv
        Produced after human annotations and merges.

Outputs (under data/):
    - data/labels_humans_cleaned_v1.csv
        Cleaned and merged version for improved model training.

Function
--------
1. Filters out rows with low-confidence `predicted_mood` (below threshold).
2. Merges closely related moods:
       calm + easygoing        → calm_easygoing
       melancholic + wistful   → melancholic_wistful
3. Keeps neutral rows (training scripts can exclude them if desired).
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

# ================== I/O ==================
DATA_DIR = Path("data")
IN_CSV  = DATA_DIR / "labels_human_merged.csv"         # INPUT
OUT_CSV = DATA_DIR / "labels_humans_cleaned_v1.csv"    # OUTPUT
# =========================================

def parse_args():
    ap = argparse.ArgumentParser(description="Clean and merge mood labels for better consistency.")
    ap.add_argument("--in_csv", type=str, default=str(IN_CSV),
                    help="Input labels CSV (default: data/labels_human_merged.csv)")
    ap.add_argument("--out_csv", type=str, default=str(OUT_CSV),
                    help="Output cleaned CSV (default: data/labels_humans_cleaned_v1.csv)")
    ap.add_argument("--pred_conf_thresh", type=float, default=0.60,
                    help="Drop rows where predicted_mood is present and pred_confidence < this (default: 0.60)")
    return ap.parse_args()

def main():
    args = parse_args()
    inp = Path(args.in_csv)
    outp = Path(args.out_csv)

    if not inp.exists():
        print(f"Input not found: {inp}")
        sys.exit(1)

    df = pd.read_csv(inp)

    # Basic validation
    if "json_path" not in df.columns or "mood" not in df.columns:
        print("Input CSV must contain at least 'json_path' and 'mood' columns")
        sys.exit(1)

    print("=== BEFORE ===")
    print("rows:", len(df))
    print("mood counts:\n", df["mood"].value_counts(dropna=False).sort_index())

    # 1) Filter low-confidence predicted rows
    if "predicted_mood" in df.columns and "pred_confidence" in df.columns:
        mask_low_pred = df["predicted_mood"].notna() & (df["pred_confidence"].fillna(0) < args.pred_conf_thresh)
        dropped = int(mask_low_pred.sum())
        if dropped > 0:
            df = df.loc[~mask_low_pred].reset_index(drop=True)
        print(f"\nFiltered low-confidence predicted rows (< {args.pred_conf_thresh:.2f}): {dropped}")
    else:
        print("\n(predicted_mood/pred_confidence not found — skipping filter)")

    # 2) Merge semantically similar mood classes
    df["mood_raw"] = df["mood"]  # preserve original for traceability
    df.loc[df["mood"].isin(["calm", "easygoing"]), "mood"] = "calm_easygoing"
    df.loc[df["mood"].isin(["melancholic", "wistful"]), "mood"] = "melancholic_wistful"

    print("Merged classes:")
    print("  calm + easygoing → calm_easygoing")
    print("  melancholic + wistful → melancholic_wistful")

    # 3) Reorder columns for clarity and save
    front = ["json_path", "mood", "mood_raw"]
    cols = front + [c for c in df.columns if c not in front]
    df = df[cols]

    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False, encoding="utf-8")

    print("\n=== AFTER ===")
    print("rows:", len(df))
    print("mood counts:\n", df["mood"].value_counts(dropna=False).sort_index())
    print(f"\n✅ Wrote cleaned labels to: {outp}")

if __name__ == "__main__":
    main()
