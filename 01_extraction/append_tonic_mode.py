#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Append tonic/mode from CSV into matching JSON files.")
    p.add_argument("--csv", required=True, help="Path to the CSV with columns: artist_folder,midi_file,tonic,mode")
    p.add_argument("--json-root", required=True, help="Root folder containing per-artist subfolders with .json files")
    p.add_argument("--dry-run", action="store_true", help="Report changes without writing files")
    p.add_argument("--backup", action="store_true", help="Write a .bak copy before modifying each JSON")
    p.add_argument("--encoding", default="utf-8", help="File encoding for reading/writing JSON (default: utf-8)")
    args = p.parse_args()

    csv_path = Path(args.csv)
    root = Path(args.json_root)

    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    if not root.exists():
        sys.exit(f"JSON root not found: {root}")

    # Stats
    total_rows = 0
    matched = 0
    updated = 0
    skipped_missing = []
    skipped_invalid = []
    already_set = 0

    # Read CSV
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"artist_folder", "midi_file", "tonic", "mode"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            sys.exit(f"CSV must have columns: {', '.join(sorted(required_cols))}")

        for row in reader:
            total_rows += 1
            artist = (row.get("artist_folder") or "").strip()
            midi_file = (row.get("midi_file") or "").strip()
            tonic = (row.get("tonic") or "").strip()
            mode = (row.get("mode") or "").strip()

            if not artist or not midi_file:
                skipped_invalid.append((artist, midi_file, "missing artist or midi_file"))
                continue

            # Expected JSON filename
            json_name = Path(midi_file).with_suffix(".json").name
            json_path = root / artist / json_name

            if not json_path.exists():
                # Try a couple of tolerant fallbacks (optional)
                candidates = [
                    json_path,
                    # Some datasets use space vs underscore differences—add light normalization attempts here if needed.
                    json_path.with_name(json_name.replace(" ", "_")),
                    json_path.with_name(json_name.replace("_", " ")),
                ]
                found = None
                for c in candidates:
                    if c.exists():
                        found = c
                        break
                if found is None:
                    skipped_missing.append(str(json_path))
                    continue
                json_path = found

            matched += 1

            # Load, update, write
            try:
                data = json.loads(json_path.read_text(encoding=args.encoding))
            except Exception as e:
                skipped_invalid.append((artist, midi_file, f"JSON read error: {e}"))
                continue

            # If fields already exist, decide whether to overwrite (here: overwrite if different)
            pre_tonic = data.get("tonic")
            pre_mode = data.get("mode")

            if pre_tonic == tonic and pre_mode == mode:
                already_set += 1
                continue

            data["tonic"] = tonic
            data["mode"] = mode

            if args.dry_run:
                print(f"[DRY] Would update: {json_path}  tonic={tonic} mode={mode}")
            else:
                if args.backup and not (json_path.parent / (json_path.name + ".bak")).exists():
                    (json_path.parent / (json_path.name + ".bak")).write_text(
                        json.dumps(data if pre_tonic is None and pre_mode is None else {**data, "tonic": pre_tonic, "mode": pre_mode}, ensure_ascii=False, indent=2),
                        encoding=args.encoding
                    )
                try:
                    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding=args.encoding)
                    updated += 1
                except Exception as e:
                    skipped_invalid.append((artist, midi_file, f"JSON write error: {e}"))

    # Report
    print("\n=== Summary ===")
    print(f"Rows in CSV:           {total_rows}")
    print(f"Matched JSON files:    {matched}")
    print(f"Updated JSON files:    {updated}")
    print(f"Already up-to-date:    {already_set}")
    print(f"Missing JSON files:    {len(skipped_missing)}")
    print(f"Invalid/other skips:   {len(skipped_invalid)}")

    if skipped_missing:
        print("\n-- Missing (first 50) --")
        for pth in skipped_missing[:50]:
            print(pth)

    if skipped_invalid:
        print("\n-- Invalid/Skipped detail (first 50) --")
        for a, m, why in skipped_invalid[:50]:
            print(f"{a}/{m} -> {why}")

if __name__ == "__main__":
    main()
