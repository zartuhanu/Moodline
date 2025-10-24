#!/usr/bin/env python3
"""
Deduplicate versioned MIDI files (e.g., "Song.mid", "Song.1.mid", "Song.2.mid")
by copying a single canonical file per song into a mirrored destination tree.

Example:
    python lmd_deduplicated.py --src lmd_clean --dst lmd_deduplicated \
        --policy keep-shortest --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List


def normalize_filename(name: str) -> str:
    """Return canonical name by stripping trailing '.<int>.mid' version tags."""
    lname = name.lower()
    if not lname.endswith(".mid"):
        return name
    stem = name[:-4]  # strip ".mid"
    head, sep, tail = stem.rpartition(".")
    if sep and tail.isdigit():
        return f"{head}.mid"
    return name


def choose_file(candidates: List[Path], policy: str) -> Path:
    """Select one file among duplicates according to a simple policy.

    Args:
        candidates: Paths considered equivalent (same normalized name).
        policy: One of {"keep-first", "keep-shortest", "keep-largest", "mtime-newest"}.

    Returns:
        The chosen Path.
    """
    if len(candidates) == 1:
        return candidates[0]
    if policy == "keep-shortest":
        return min(candidates, key=lambda p: p.stat().st_size)
    if policy == "keep-largest":
        return max(candidates, key=lambda p: p.stat().st_size)
    if policy == "mtime-newest":
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return candidates[0]  # keep-first


def deduplicate(
    src: Path,
    dst: Path,
    policy: str = "keep-first",
    dry_run: bool = False,
) -> None:
    """Copy one MIDI per normalized name from src to dst.

    Notes:
        * Normalization collapses ".<int>.mid" suffixes only.
        * Destination directory mirrors the first-level folder and filename.
    """
    midi_paths = list(src.rglob("*.mid"))
    if not midi_paths:
        print(f"No MIDI files found under: {src}")
        return

    groups: dict[str, list[Path]] = {}
    for p in midi_paths:
        rel = p.relative_to(src)
        # Keep the first-level directory (e.g., artist/hash bucket) + normalized filename.
        bucket = rel.parts[0] if len(rel.parts) > 1 else ""
        key = f"{bucket}/{normalize_filename(p.name)}"
        groups.setdefault(key, []).append(p)

    kept = 0
    for key, paths in groups.items():
        chosen = choose_file(paths, policy)
        out = dst / key
        out.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            print(f"[DRY-RUN] {chosen} -> {out}")
        else:
            shutil.copy2(chosen, out)
        kept += 1

    print(f"Done. Considered: {len(midi_paths)} | Written: {kept} | Groups: {len(groups)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate versioned MIDI files by filename."
    )
    parser.add_argument("--src", type=Path, default=Path("lmd_clean"),
                        help="Source directory containing MIDI files.")
    parser.add_argument("--dst", type=Path, default=Path("lmd_deduplicated"),
                        help="Destination for deduplicated files.")
    parser.add_argument("--policy", type=str, default="keep-first",
                        choices=["keep-first", "keep-shortest", "keep-largest", "mtime-newest"],
                        help="How to pick among duplicates with the same normalized name.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without copying files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deduplicate(args.src, args.dst, args.policy, args.dry_run)


if __name__ == "__main__":
    main()
