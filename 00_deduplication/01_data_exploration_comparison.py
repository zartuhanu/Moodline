#!/usr/bin/env python3
"""
Count how many MIDI files exist in a dataset organized as:
    dataset/artist/song.mid

Outputs:
- total number of .mid files
- number of artists
- file count per artist
"""

import os
from pathlib import Path

# ======== CONFIG ========
DATASET_DIR = Path("data/lmd_deduplicated")   # change this path if needed
# =========================

if not DATASET_DIR.exists():
    raise SystemExit(f"Error: folder '{DATASET_DIR}' does not exist.")

# dictionary: { artist_name: number_of_mid_files }
artist_counts = {}

# walk through each artist folder
for artist_folder in DATASET_DIR.iterdir():
    if artist_folder.is_dir():
        midi_files = list(artist_folder.rglob("*.mid"))
        artist_counts[artist_folder.name] = len(midi_files)

# total count
total_midis = sum(artist_counts.values())

# ======== OUTPUT ========
print(f" Dataset path: {DATASET_DIR.resolve()}")
print(f" Number of artists: {len(artist_counts)}")
print(f" Total MIDI files: {total_midis}")
print("-" * 40)



print("-" * 40)
print(f" Done. Total: {total_midis} MIDI files across {len(artist_counts)} artists.")
