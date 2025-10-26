#!/usr/bin/env python3
"""
Merge high-confidence predictions for 'neutral' songs into the label table.

Inputs (under data/):
    data/labels.csv               – original heuristic labels
    data/neutral_predictions.csv  – predictions for neutral rows with confidences

Output:
    data/labels_merged.csv        – labels with accepted high-confidence replacements
"""

import csv
from pathlib import Path

# ========= I/O =========
LABELS_CSV = Path("data/labels.csv")
PRED_CSV   = Path("data/neutral_predictions.csv")
OUT_CSV    = Path("data/labels_merged.csv")
CONF_THRESH = 0.68
# ========================================

def main():
    if not LABELS_CSV.exists(): raise SystemExit(f"Missing {LABELS_CSV}")
    if not PRED_CSV.exists():   raise SystemExit(f"Missing {PRED_CSV}")

    # Load predictions keyed by json_path
    preds = {}
    with PRED_CSV.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            jp = row["json_path"]
            preds[jp] = row  # contains predicted_mood, confidence, per-class probs

    # Merge into labels
    with LABELS_CSV.open("r", encoding="utf-8") as f_in, \
         OUT_CSV.open("w", newline="", encoding="utf-8") as f_out:
        r = csv.DictReader(f_in)
        fieldnames = list(r.fieldnames or [])
        if "mood" not in fieldnames:
            raise SystemExit("labels.csv must include a 'mood' column.")
        # Add merge columns if absent
        for extra in ["source", "predicted_mood", "pred_confidence"]:
            if extra not in fieldnames:
                fieldnames.append(extra)
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()

        changed = kept = 0
        for row in r:
            jp = row.get("json_path","")
            mood = (row.get("mood","") or "").lower()
            if mood == "neutral" and jp in preds:
                conf = float(preds[jp]["confidence"])
                if conf >= CONF_THRESH:
                    row["predicted_mood"] = preds[jp]["predicted_mood"]
                    row["pred_confidence"] = f"{conf:.6f}"
                    row["mood"] = preds[jp]["predicted_mood"]  # accept replacement
                    row["source"] = f"pred@≥{CONF_THRESH}"
                    changed += 1
                else:
                    row["predicted_mood"] = preds[jp]["predicted_mood"]
                    row["pred_confidence"] = f"{conf:.6f}"
                    row["source"] = "neutral"  # keep as neutral
                    kept += 1
            else:
                row["source"] = row.get("source") or "heuristic"
            w.writerow(row)

    print(f"Wrote {OUT_CSV}")
    print(f"Accepted high-confidence predictions: {changed}")
    print(f"Kept neutral (below threshold): {kept}")

if __name__ == "__main__":
    main()
