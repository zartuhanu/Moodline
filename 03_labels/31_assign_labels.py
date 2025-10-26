#!/usr/bin/env python3
"""
Assign heuristic mood labels from per-song features (tempo, mode, density).

Input:
    data/features.csv
        Columns (from build_tokens.py): json_path, tempo, mode, tonic, density

Output:
    data/labels.csv
        Columns: json_path, tempo, mode, tonic, density,
                 tempo_band, density_bucket, mood

Console diagnostics:
    - Label distribution
    - Tempo and density histograms
    - Tempo band × Mode crosstab (major/minor)
"""

import csv
from pathlib import Path
from collections import Counter, defaultdict
import math

# ---------- I/O locations ----------

FEATURES_CSV = Path("data/features.csv")   # input from build_tokens.py
OUT_DIR      = Path("data")                # where to save outputs
LABELS_CSV   = OUT_DIR / "labels.csv"

# ---- Heuristic thresholds (tweak and re-run) ----
# Tempo bands (bpm): slow if < SLOW_MAX, medium if [SLOW_MAX, MED_MAX), else fast
SLOW_MAX = 90.0
MED_MAX  = 110.0

# Density buckets (notes per bar)
LOW_DENS_MAX  = 2.5
HIGH_DENS_MIN = 3.5

# Label names (anchor words)
LABELS = {
    "fast_major_high": "joyful",
    "fast_minor_high": "angry",
    "slow_minor_low":  "melancholic",
    "slow_major_low":  "calm",
    "med_major":       "easygoing",   # aka content/uplifting
    "med_minor":       "wistful",     # aka moody/pensive
    "fallback":        "neutral",
}

# Histogram settings (for console plots)
TEMPO_BINS    = [0, 70, 80, 90, 100, 110, 120, 130, 140, 999]
DENSITY_BINS  = [0, 1, 2, 2.5, 3, 3.5, 4, 5, 6, 99]
BAR_WIDTH = 40  # characters
# ======================================================


# -------- helpers --------
def tempo_band(bpm: float) -> str:
    """Map BPM to 'slow' | 'med' | 'fast' using SLOW_MAX and MED_MAX."""
    if bpm < SLOW_MAX: return "slow"
    if bpm < MED_MAX:  return "med"
    return "fast"

def density_bucket(d: float) -> str:
    """Bucket notes-per-bar into 'low' | 'med' | 'high' using LOW_DENS_MAX/HIGH_DENS_MIN."""
    if d <= LOW_DENS_MAX:  return "low"
    if d >= HIGH_DENS_MIN: return "high"
    return "med"

def is_minor(mode: str) -> bool:
    """True if mode string denotes minor (prefix 'min'); robust to casing/whitespace."""
    return str(mode or "").strip().lower().startswith("min")

def map_to_label(bpm: float, mode: str, density: float) -> str:
    """Heuristic mapping from tempo band × density × mode → mood label."""
    t = tempo_band(bpm)
    db = density_bucket(density)
    minor = is_minor(mode)

    if t == "fast" and db == "high":
        return LABELS["fast_minor_high" if minor else "fast_major_high"]
    if t == "slow" and db == "low":
        return LABELS["slow_minor_low" if minor else "slow_major_low"]
    if t == "med":
        return LABELS["med_minor" if minor else "med_major"]
    return LABELS["fallback"]

def hist_counts(values, bins):
    """Count values into half-open intervals [b_i, b_{i+1}); values > last edge go to final bin."""
    counts = [0]*(len(bins)-1)
    for v in values:
        placed = False
        for i in range(len(bins)-1):
            lo, hi = bins[i], bins[i+1]
            if (v >= lo) and (v < hi):
                counts[i] += 1
                placed = True
                break
        if not placed and values:  # above last bound
            counts[-1] += 1
    return counts

def print_hist(title, bins, counts, total):
    """Print a fixed-width horizontal histogram with percentage annotations."""
    print(f"\n{title}")
    maxc = max(counts) if counts else 1
    for i, c in enumerate(counts):
        lo, hi = bins[i], bins[i+1]
        bar_len = 0 if maxc == 0 else int((c / maxc) * BAR_WIDTH)
        pct = (100*c/total) if total else 0.0
        rng = f"[{lo:.0f},{hi:.0f})"
        print(f"  {rng:>9} | {'█'*bar_len:<{BAR_WIDTH}} {c:5d}  ({pct:5.1f}%)")

def crosstab(rows, row_key, col_key):
    """Build contingency table: dict[row_val][col_val] -> count."""
    R = defaultdict(lambda: Counter())
    for r in rows:
        R[row_key(r)][col_key(r)] += 1
    return R


# -------- main --------
def main():
    if not FEATURES_CSV.exists():
        raise SystemExit(f"Features file not found: {FEATURES_CSV}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load features
    items = []
    with FEATURES_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                bpm = float(row["tempo"])
                dens = float(row["density"])
                md = row.get("mode","")
                tnc = row.get("tonic","")
                jpath = row.get("json_path","")
                items.append({"bpm": bpm, "density": dens, "mode": md, "tonic": tnc, "json_path": jpath})
            except Exception:
                # Skip malformed rows silently
                continue

    # Assign labels
    counts = Counter()
    with LABELS_CSV.open("w", newline="", encoding="utf-8") as f_out:
        fieldnames = ["json_path","tempo","mode","tonic","density","tempo_band","density_bucket","mood"]
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()
        for it in items:
            mood = map_to_label(it["bpm"], it["mode"], it["density"])
            tb = tempo_band(it["bpm"])
            db = density_bucket(it["density"])
            w.writerow({
                "json_path": it["json_path"],
                "tempo": it["bpm"],
                "mode": it["mode"],
                "tonic": it["tonic"],
                "density": it["density"],
                "tempo_band": tb,
                "density_bucket": db,
                "mood": mood,
            })
            counts[mood] += 1

    # Summary
    total = sum(counts.values())
    print(f"Wrote labels to: {LABELS_CSV}")
    print("\nLabel distribution:")
    for k, v in counts.most_common():
        pct = 100.0 * v / total if total else 0.0
        print(f"  {k:12s} {v:6d}  ({pct:5.1f}%)")

    # Histograms
    tempos    = [it["bpm"] for it in items]
    densities = [it["density"] for it in items]
    t_counts  = hist_counts(tempos, TEMPO_BINS)
    d_counts  = hist_counts(densities, DENSITY_BINS)
    print_hist("Tempo histogram (bpm)", TEMPO_BINS, t_counts, total)
    print_hist("Density histogram (notes/bar)", DENSITY_BINS, d_counts, total)

    # Crosstab: tempo_band × mode (major/minor)
    rows = [{
        "tb": tempo_band(it["bpm"]),
        "mm": "minor" if is_minor(it["mode"]) else "major"
    } for it in items]
    xt = crosstab(rows, lambda r: r["tb"], lambda r: r["mm"])
    print("\nTempo band × Mode crosstab:")
    all_rows = ["slow","med","fast"]
    all_cols = ["major","minor"]
    header = "            " + "".join([f"{c:>10s}" for c in all_cols]) + "   total"
    print(header)
    for r in all_rows:
        row_counts = [xt[r][c] for c in all_cols]
        print(f"{r:>10s} " + "".join([f"{n:10d}" for n in row_counts]) + f" {sum(row_counts):7d}")

if __name__ == "__main__":
    main()
