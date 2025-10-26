#!/usr/bin/env python3
"""
Shift all bassline note timings so that the first note begins at tick 0.

Use this to normalize per-song JSONs before tokenization or feature extraction.

Inputs:
    output_basslines/     – directory containing per-song JSONs from extract_basslines.py

Outputs:
    • In-place (default), overwriting original files
    • OR into a separate folder that mirrors the same substructure
"""

import os
import json
from pathlib import Path


def shift_bassline_file(json_path, in_place=True, out_dir=None):
    """
    Load one bassline JSON, subtract the first note's start_tick from every event,
    then save either in place or into a mirrored path under out_dir.

    Args:
        json_path: Path to a single bassline JSON file.
        in_place: If True, overwrite the file in place.
        out_dir:  Target root directory when in_place=False.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    events = data.get('bassline', [])
    if not events:
        return  # nothing to shift

    # Find earliest start tick (usually first note)
    min_start = min(ev['start_tick'] for ev in events)

    # Shift all events so first note starts at tick 0
    for ev in events:
        ev['start_tick'] = ev['start_tick'] - min_start

    data['bassline'] = events

    # Determine output path
    if in_place:
        out_path = json_path
    else:
        assert out_dir is not None, "If not in_place, out_dir must be provided"
        relative = json_path.relative_to(json_path.parents[1])
        out_path = Path(out_dir) / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save shifted file
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)


def shift_all_basslines(input_dir, in_place=True, output_dir=None):
    """
    Recursively walk input_dir for *.json files and shift each bassline so its
    first note starts at 0.
    """
    input_dir = Path(input_dir)
    if not in_place:
        assert output_dir is not None, "output_dir required when not in_place"
        output_dir = Path(output_dir)

    for root, _, files in os.walk(input_dir):
        for fn in files:
            if not fn.lower().endswith('.json'):
                continue
            full = Path(root) / fn
            shift_bassline_file(full, in_place=in_place, out_dir=output_dir)
            print(f"Shifted: {full}")


if __name__ == "__main__":
    # ——— configure these paths ———
    INPUT_DIR  = "output_basslines"   # input folder with bassline JSONs
    IN_PLACE   = True                 # set False to write into OUTPUT_DIR instead
    OUTPUT_DIR = "shifted_json"       # used only if IN_PLACE=False
    # —————————————————————————————————

    shift_all_basslines(INPUT_DIR, in_place=IN_PLACE, output_dir=OUTPUT_DIR)
