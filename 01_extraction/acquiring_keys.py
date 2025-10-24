import os
import csv
import traceback
from music21 import converter

CSV_PATH   = 'midi_keys.csv'
DATA_ROOT  = 'lmd_deduplicated'
OUTPUT_CSV = 'midi_keys.csv'   # same CSV, we’ll append new entries

def load_done_set(csv_path):
    done = set()
    if os.path.isfile(csv_path):
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                # use the relative path (artist/file) as key
                done.add(os.path.join(r['artist_folder'], r['midi_file']))
    return done

def extract_key(midi_path):
    try:
        s   = converter.parse(midi_path)
        k   = s.analyze('key')
        return k.tonic.name, k.mode
    except Exception:
        traceback.print_exc()
        return None, None

def main():
    done_set = load_done_set(CSV_PATH)
    all_midis = []
    for root, _, files in os.walk(DATA_ROOT):
        for fn in files:
            if fn.lower().endswith(('.mid', '.midi')):
                rel = os.path.relpath(os.path.join(root, fn), DATA_ROOT)
                all_midis.append(rel)

    to_do = [p for p in all_midis if p not in done_set]
    print(f"{len(done_set)} already done, {len(to_do)} to go.")

    # open CSV in append mode
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        # only write header if file was empty
        if os.path.getsize(OUTPUT_CSV) == 0:
            writer.writerow(['artist_folder','midi_file','tonic','mode'])

        for idx, rel_path in enumerate(to_do, 1):
            abs_path = os.path.join(DATA_ROOT, rel_path)
            artist, fname = rel_path.split(os.sep,1)
            print(f"[{idx}/{len(to_do)}] Processing {rel_path}", flush=True)
            tonic, mode = extract_key(abs_path)
            t = tonic if tonic else 'error'
            m = mode if mode else 'error'
            writer.writerow([artist, fname, t, m])
            print(f"    → Key: {t} Mode: {m}")

    print("Done. Updated", OUTPUT_CSV)

if __name__ == '__main__':
    main()
