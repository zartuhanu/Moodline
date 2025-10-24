#!/usr/bin/env python3
"""
Shift bass note timings in JSON outputs so the first note starts at 0s.

Overwrites files in place. Assumes JSONs produced by extract_basslines.py.

Example:
    python shifting.py --root output_basslines
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def shift_file(json_path: Path) -> bool:
    """Shift a single JSON file in place. Returns True if modified."""
    try:
        data: Dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to read {json_path}: {e}")
        return False

    notes: List[Dict[str, Any]] = data.get("notes", [])
    if not notes:
        return False

    t0 = min(float(n.get("start", 0.0)) for n in notes)
    if t0 <= 0:
        return False

    for n in notes:
        n["start"] = float(n["start"]) - t0
        n["end"] = float(n["end"]) - t0

    try:
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        print(f"Failed to write {json_path}: {e}")
        return False


def process_tree(root: Path) -> None:
    json_files = list(root.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found under: {root}")
        return

    changed = 0
    for jp in json_files:
        if shift_file(jp):
            changed += 1
            print(f"Shifted: {jp}")

    print(f"Done. Shifted {changed} / {len(json_files)} files.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shift bass note timings to start at 0s (in place).")
    p.add_argument("--root", type=Path, default=Path("output_basslines"),
                   help="Root directory containing JSON outputs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    process_tree(args.root)


if __name__ == "__main__":
    main()
