import os
import json
from pathlib import Path

def shift_bassline_file(json_path, in_place=True, out_dir=None):
    """
    Loads one bassline JSON, subtracts the first note's start_tick from every event,
    then saves back (in place or into out_dir preserving subfolders).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    events = data.get('bassline', [])
    if not events:
        return  # nothing to do

    # find the minimum start_tick (should be the first event)
    min_start = min(ev['start_tick'] for ev in events)

    # shift every event
    for ev in events:
        ev['start_tick'] = ev['start_tick'] - min_start

    # update the data
    data['bassline'] = events

    # determine output path
    if in_place:
        out_path = json_path
    else:
        assert out_dir is not None, "If not in_place, out_dir must be provided"
        relative = json_path.relative_to(json_path.parents[1])
        out_path = out_dir / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # save
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)

def shift_all_basslines(input_dir, in_place=True, output_dir=None):
    """
    Walk input_dir for *.json, and shift each bassline so first note starts at 0.
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
            shift_bassline_file(
                full,
                in_place=in_place,
                out_dir=output_dir
            )
            print(f"Shifted: {full}")

if __name__ == "__main__":
    # ——— configure these paths ———
    INPUT_DIR  = "output_basslines"
    IN_PLACE   = True           # set False to write into OUTPUT_DIR instead
    OUTPUT_DIR = "shifted_json"  # only used if IN_PLACE=False
    # —————————————————————————————————

    shift_all_basslines(INPUT_DIR, in_place=IN_PLACE, output_dir=OUTPUT_DIR)