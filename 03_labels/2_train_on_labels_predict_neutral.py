#!/usr/bin/env python3
# Baseline: train on non-neutral, predict on neutral and export confidences.
"""
Train a simple classifier on non-neutral labels, then predict labels for
the 'neutral' rows and export per-class probabilities.

Input:
    data/labels.csv
        Columns: json_path, tempo, mode, tonic, density, tempo_band, density_bucket, mood

Outputs (under data/):
    data/neutral_predictions.csv   – predictions for the 'neutral' subset with confidences
    data/rf_baseline_model.json    – lightweight model metadata (not a pickle)

Console:
    - Validation report on a held-out split of the non-neutral set
    - Confusion matrix
    - Neutral-set prediction distribution and high-confidence count
"""

from pathlib import Path
import math
import csv
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# ------------ I/O ------------
LABELS_CSV = Path("data/labels.csv")          # produced by assign_labels.py
OUT_DIR     = Path("data")
PRED_CSV    = OUT_DIR / "neutral_predictions.csv"
MODEL_JSON  = OUT_DIR / "rf_baseline_model.json"  # stores basic metadata (not pickle)

CONFIDENCE_THRESHOLD = 0.70   # flag high-confidence predictions
RANDOM_SEED = 42
# ==========================================

# --- Helpers ---
TONIC_TO_PC = {
    # naturals
    "C":0, "D":2, "E":4, "F":5, "G":7, "A":9, "B":11,
    # sharps
    "C#":1, "CS":1, "D#":3, "DS":3, "F#":6, "FS":6, "G#":8, "GS":8, "A#":10, "AS":10,
    # flats
    "Db":1, "Eb":3, "Gb":6, "Ab":8, "Bb":10,
}

def normalize_tonic(s: str) -> str:
    """Normalize tonic spelling: accepts 'F#', 'Gb', 'Bb', or prior 'FS'."""
    if not s:
        return "C"
    s = s.strip()
    return s.replace("♯", "#").replace("♭", "b").replace("S", "#")

def tonic_to_unit_circle(tonic: str):
    """Return (cos, sin) for pitch class on the unit circle; defaults to C if unknown."""
    t = normalize_tonic(tonic)
    pc = TONIC_TO_PC.get(t, 0)
    ang = 2*math.pi*pc/12.0
    return math.cos(ang), math.sin(ang)

def is_minor(mode: str) -> int:
    """Binary minor flag from mode string (prefix 'min' → 1, else 0)."""
    return 1 if str(mode or "").lower().startswith("min") else 0

def build_features(df: pd.DataFrame):
    """Assemble simple numeric features for classification."""
    # Base numeric features
    tempo = df["tempo"].astype(float).to_numpy().reshape(-1,1)
    density = df["density"].astype(float).to_numpy().reshape(-1,1)
    minor = df["mode"].apply(is_minor).astype(int).to_numpy().reshape(-1,1)

    # Key on pitch-class circle (cos/sin)
    cos_list, sin_list = [], []
    for t in df["tonic"].fillna("C").astype(str):
        c, s = tonic_to_unit_circle(t)
        cos_list.append(c); sin_list.append(s)
    key_cos = np.array(cos_list).reshape(-1,1)
    key_sin = np.array(sin_list).reshape(-1,1)

    X = np.hstack([tempo, density, minor, key_cos, key_sin]).astype(float)
    feat_names = ["tempo", "density", "is_minor", "key_cos", "key_sin"]
    return X, feat_names

def main():
    if not LABELS_CSV.exists():
        raise SystemExit(f"File not found: {LABELS_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(LABELS_CSV)
    if "mood" not in df.columns:
        raise SystemExit("labels.csv must contain a 'mood' column.")

    # Split into train (non-neutral) and unlabeled/test (neutral)
    df_train = df[df["mood"].str.lower() != "neutral"].copy()
    df_neutral = df[df["mood"].str.lower() == "neutral"].copy()

    if df_train.empty:
        raise SystemExit("No non-neutral rows to train on. Adjust heuristics first.")

    # Build features
    X_all, feat_names = build_features(df_train)
    y_all = df_train["mood"].astype(str).to_numpy()

    # Standardize tempo/density only (tree models don't need it, but harmless if switching models later)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # Train/val split for a quick sanity check
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_scaled, y_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_all
    )

    # Baseline model
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    # Validation report
    y_hat = clf.predict(X_va)
    print("\n=== Validation (non-neutral) ===")
    print(classification_report(y_va, y_hat, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_va, y_hat))

    # Predict on the neutral set
    if not df_neutral.empty:
        X_neu, _ = build_features(df_neutral)
        X_neu = scaler.transform(X_neu)
        proba = clf.predict_proba(X_neu)
        preds = clf.classes_[np.argmax(proba, axis=1)]
        conf = np.max(proba, axis=1)

        # Write predictions with confidences and per-class probabilities
        cols = ["json_path","tempo","mode","tonic","density","predicted_mood","confidence"]
        prob_cols = [f"p_{c}" for c in clf.classes_]
        out_cols = cols + prob_cols

        with PRED_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(out_cols)
            for i, (_, row) in enumerate(df_neutral.iterrows()):
                w.writerow([
                    row.get("json_path",""),
                    row.get("tempo",""),
                    row.get("mode",""),
                    row.get("tonic",""),
                    row.get("density",""),
                    preds[i],
                    round(float(conf[i]), 6),
                    *[round(float(p), 6) for p in proba[i]]
                ])

        print(f"\n Wrote neutral predictions to: {PRED_CSV}")
        # quick summary
        pred_counts = pd.Series(preds).value_counts().sort_values(ascending=False)
        print("\nPredicted label distribution over NEUTRAL rows:")
        print(pred_counts.to_string())
        hi = (conf >= CONFIDENCE_THRESHOLD).sum()
        total = len(conf)
        print(f"\nHigh-confidence (≥{CONFIDENCE_THRESHOLD:.2f}) predictions: {hi}/{total} "
              f"({100.0*hi/max(1,total):.1f}%)")

    # Save basic model metadata (so you remember settings)
    MODEL_JSON.write_text(json.dumps({
        "model": "RandomForestClassifier",
        "n_estimators": clf.n_estimators,
        "class_weight": "balanced_subsample",
        "random_state": RANDOM_SEED,
        "features": feat_names,
        "classes_": clf.classes_.tolist(),
        "scaler": {"type": "StandardScaler", "fit_on": ["tempo","density","is_minor","key_cos","key_sin"]},
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }, indent=2))

    print(f"\n Saved model metadata to: {MODEL_JSON}")

if __name__ == "__main__":
    main()
