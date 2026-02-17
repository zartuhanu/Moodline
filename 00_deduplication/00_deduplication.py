#!/usr/bin/env python3
"""
Deduplicate versioned MIDI files in the Lakh MIDI Dataset (LMD) or similar structure.

Removes duplicate song versions such as:
    Song.mid, Song.1.mid, Song.2.mid, ...
keeping only the first unique base name encountered.

Input:
    lmd_clean/           – original MIDI dataset
Output:
    lmd_deduplicated/    – deduplicated dataset (mirrors subfolder structure)
"""

import os
import shutil
from pathlib import Path

# ---------- Paths ----------
src_dir = Path("lmd_clean")          # source directory with raw MIDI files
dst_dir = Path("lmd_deduplicated")   # destination for deduplicated files
dst_dir.mkdir(parents=True, exist_ok=True)

# Track unique song names to avoid duplicates
seen_songs = set()

# ---------- Main loop ----------
for root, _, files in os.walk(src_dir):
    for file in files:
        if file.endswith(".mid"):
            # Remove numeric version suffixes like '.1.mid' or '.2.mid'
            base_name = file
            if '.' in file:
                parts = file.split('.')
                if len(parts) >= 3 and parts[-2].isdigit():
                    base_name = '.'.join(parts[:-2]) + '.mid'

            # Copy only the first occurrence of each unique song
            if base_name not in seen_songs:
                seen_songs.add(base_name)
                src_path = Path(root) / file
                relative_path = src_path.relative_to(src_dir)
                dst_subfolder = dst_dir / relative_path.parent
                dst_subfolder.mkdir(parents=True, exist_ok=True)
                dst_path = dst_subfolder / base_name
                shutil.copy(src_path, dst_path)
                print(f"Copied: {src_path} → {dst_path}")
