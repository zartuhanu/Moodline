#!/usr/bin/env python3
"""
Artist-level train/val/test split with a frozen test set.

I/O
---
Inputs (under data/):
    - data/labels_human_merged.csv
    - data/test_artists.txt    (optional; if absent, it will be created)

Outputs (under data/):
    - data/splits.json
    - data/test_artists.txt    (created if missing)
"""

import csv, json, hashlib, random
from math import ceil
from pathlib import Path
from collections import defaultdict, Counter

# ================== I/O ==================
OUT_DIR         = Path("data")
LABELS_CSV       = OUT_DIR / "labels_human_merged.csv"  # INPUT
SPLITS_JSON      = Path("data/splits.json")              # OUTPUT
TEST_ARTISTS_TXT = Path("data/test_artists.txt")         # INPUT/OUTPUT
# ==========================================================

# ============== SETTINGS ===============
TEST_RATIO             = 0.20     # ~20% of artists → TEST
VAL_FRACTION_OF_REST   = 0.125    # ≈10% overall (0.8 * 0.125)
HASH_SEED              = "v1_freeze"   # change ONLY to intentionally re-freeze test
MAX_TEST_SONGS_PER_ARTIST = 15    # cap per artist in TEST
TEST_ARTIST_SHARE_WARN = 0.05     # warn if any artist > 5% of test rows
# ==========================================================

def artist_from_path(p: str) -> str:
    parts = Path(p).parts
    return parts[-2] if len(parts) >= 2 else "UNKNOWN"

def stable_hash(obj: str) -> float:
    h = hashlib.md5((HASH_SEED + "@" + obj).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)

def summarize_class_balance(rows, indices):
    moods = sorted({r["mood"] for r in rows})
    total = len(rows)
    print("\n=== Overall distribution ===")
    overall = Counter(r["mood"] for r in rows)
    for m in moods:
        c = overall.get(m, 0); print(f"  {m:14s} {c:6d} ({c/max(1,total):6.1%})")

    print("\n=== Per-split distribution ===")
    for split in ("train","val","test"):
        idxs = set(indices[split]); n = len(idxs)
        counts = Counter(rows[i]["mood"] for i in idxs)
        print(f"\n[{split}] n={n}")
        for m in moods:
            c = counts.get(m, 0); print(f"  {m:14s} {c:6d} ({c/max(1,n):6.1%})")
        missing = [m for m in moods if counts.get(m,0)==0]
        if missing:
            print("  ⚠ Missing:", ", ".join(missing))

def warn_artist_dominance(by_artist, test_artists, test_indices_total):
    # Compute each test artist's contribution %
    contribs = []
    test_total = max(1, test_indices_total)
    for a in sorted(test_artists):
        n = len(by_artist[a])
        n_capped = min(n, MAX_TEST_SONGS_PER_ARTIST)  # reflect real cap
        contribs.append((a, n_capped, n_capped / test_total))
    # Print top contributors
    print("\n=== Test artist contributions (capped) ===")
    for a, n_cap, frac in sorted(contribs, key=lambda x: x[2], reverse=True)[:15]:
        print(f"  {a:28s} {n_cap:5d}  ({frac:6.2%})")
    # Warnings
    offenders = [(a, n_cap, frac) for a, n_cap, frac in contribs if frac > TEST_ARTIST_SHARE_WARN]
    if offenders:
        print("\n⚠ WARNING: Some artists exceed the per-artist share threshold "
              f"({TEST_ARTIST_SHARE_WARN:.0%}) of the test set. Consider lowering "
              f"MAX_TEST_SONGS_PER_ARTIST below {MAX_TEST_SONGS_PER_ARTIST}.")
        for a, n_cap, frac in offenders:
            print(f"   - {a}: {n_cap} songs → {frac:.2%} of test")

def main():
    if not LABELS_CSV.exists():
        raise SystemExit(f"Missing {LABELS_CSV}")

    # Load rows
    rows = []
    with LABELS_CSV.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            row["_idx"] = i
            row["_artist"] = artist_from_path(row["json_path"])
            rows.append(row)

    # Group by artist
    by_artist = defaultdict(list)
    for row in rows:
        by_artist[row["_artist"]].append(row)

    # Freeze or create test artists
    if TEST_ARTISTS_TXT.exists():
        test_artists = {a.strip() for a in TEST_ARTISTS_TXT.read_text().splitlines() if a.strip()}
        print(f"ℹ️ Using existing frozen test artists ({len(test_artists)}).")
    else:
        candidates = [a for a, lst in by_artist.items() if any(r["mood"].lower()!="neutral" for r in lst)]
        if not candidates:
            raise SystemExit("No artists with non-neutral labels found for test selection.")
        scored = sorted(candidates, key=lambda a: stable_hash(a))
        need_n = max(1, ceil(TEST_RATIO * len(candidates)))
        test_artists = set(scored[:need_n])
        TEST_ARTISTS_TXT.write_text("\n".join(sorted(test_artists)))
        print(f"✅ Wrote frozen test artists: {TEST_ARTISTS_TXT} (n_artists={len(test_artists)})")

    # Split remaining into train/val
    remaining = [a for a in by_artist if a not in test_artists]
    rng = random.Random(42)
    rng.shuffle(remaining)
    n_val = max(1, int(VAL_FRACTION_OF_REST * len(remaining)))
    val_artists = set(remaining[:n_val])
    train_artists = set(remaining[n_val:])

    # Build indices (apply cap only to TEST)
    def indices_for(arts, cap=False):
        idxs = []
        for a in arts:
            songs = by_artist[a]
            if cap and len(songs) > MAX_TEST_SONGS_PER_ARTIST:
                songs = sorted(songs, key=lambda r: stable_hash(a + "#" + r["json_path"]))[:MAX_TEST_SONGS_PER_ARTIST]
            idxs.extend(r["_idx"] for r in songs)
        return sorted(idxs)

    indices = {
        "test":  indices_for(test_artists, cap=True),
        "train": indices_for(train_artists),
        "val":   indices_for(val_artists),
    }

    # Save splits
    SPLITS_JSON.write_text(json.dumps({
        "artists": {
            "test_frozen": sorted(test_artists),
            "train":       sorted(train_artists),
            "val":         sorted(val_artists),
        },
        "indices": indices,
        "settings": {
            "hash_seed": HASH_SEED,
            "test_ratio": TEST_RATIO,
            "max_test_songs_per_artist": MAX_TEST_SONGS_PER_ARTIST,
            "val_fraction_of_rest": VAL_FRACTION_OF_REST,
            "warning_threshold_artist_share": TEST_ARTIST_SHARE_WARN
        }
    }, indent=2))

    # Console summary
    print(f"Wrote {SPLITS_JSON}")
    print(f"Artists → train:{len(train_artists)}  val:{len(val_artists)}  test:{len(test_artists)}")
    for k in ("train","val","test"):
        print(f"{k}: {len(indices[k])} rows")

    summarize_class_balance(rows, indices)
    warn_artist_dominance(by_artist, test_artists, len(indices["test"]))

if __name__ == "__main__":
    main()
