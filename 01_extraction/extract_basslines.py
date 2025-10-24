#!/usr/bin/env python3
"""
Extract a bassline and a representative tempo from MIDI files.

- Tempo: scans all SetTempo events and chooses the BPM that dominates in time.
- Bassline: picks the non-drum instrument with the lowest median pitch and
  exports its notes as a monophonic (as-played) sequence—no forced quantization.

Example:
    python extract_basslines.py --src lmd_deduplicated --dst output_basslines --dry-run
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import mido
import pretty_midi


# --------------------------- Tempo extraction ---------------------------

def _tempo_segments(mid: mido.MidiFile) -> List[Tuple[float, int]]:
    """Return [(time_seconds_at_change, microseconds_per_beat), ...] sorted by time."""
    ticks_per_beat = mid.ticks_per_beat
    t = 0
    uspb_current = 500_000  # default 120 BPM
    segments: List[Tuple[float, int]] = [(0.0, uspb_current)]

    for track in mid.tracks:
        t_track = 0
        uspb = uspb_current
        for msg in track:
            t_track += msg.time
            if msg.type == "set_tempo":
                # absolute seconds for this track; keep earliest seen at each time
                sec = mido.tick2second(t_track, ticks_per_beat, uspb)
                segments.append((sec, msg.tempo))
                uspb = msg.tempo
    # Sort and coalesce by time (keep last tempo at same timestamp)
    segments.sort(key=lambda x: x[0])
    coalesced: List[Tuple[float, int]] = []
    for sec, uspb in segments:
        if not coalesced or sec > coalesced[-1][0]:
            coalesced.append((sec, uspb))
        else:
            coalesced[-1] = (sec, uspb)
    return coalesced


def extract_true_tempo(midi_path: Path, min_bpm: int = 40, max_bpm: int = 240) -> float:
    """Return the BPM whose tempo segment spans the largest total time.

    Args:
        midi_path: Path to a .mid file.
        min_bpm: Minimum plausible BPM (used to clamp outliers).
        max_bpm: Maximum plausible BPM.

    Returns:
        Dominant BPM as a float.

    Notes:
        If there are no tempo changes, defaults to 120 BPM.
    """
    mid = mido.MidiFile(str(midi_path))
    ticks_per_beat = mid.ticks_per_beat

    segments = _tempo_segments(mid)
    if not segments:
        return 120.0

    # Compute duration of each segment by looking ahead to the next change
    totals: dict[int, float] = {}
    for i, (t_s, uspb) in enumerate(segments):
        t_e = segments[i + 1][0] if i + 1 < len(segments) else mid.length
        dur = max(0.0, t_e - t_s)
        bpm = round(60_000_000 / max(1, uspb))
        bpm = int(min(max(bpm, min_bpm), max_bpm))
        totals[bpm] = totals.get(bpm, 0.0) + dur

    # Pick the BPM with the longest total duration; break ties by closest to 120
    bpm = max(sorted(totals.keys(), key=lambda x: abs(x - 120)), key=lambda b: totals[b])
    return float(bpm)


# --------------------------- Bassline extraction ---------------------------

def _choose_bass_instrument(pm: pretty_midi.PrettyMIDI) -> Optional[pretty_midi.Instrument]:
    """Heuristic: choose the non-drum instrument with the lowest median pitch."""
    candidates = [inst for inst in pm.instruments if not inst.is_drum and inst.notes]
    if not candidates:
        return None

    def median_pitch(inst: pretty_midi.Instrument) -> float:
        pitches = sorted(n.pitch for n in inst.notes)
        if not pitches:
            return 1e9
        m = len(pitches) // 2
        return (pitches[m - 1] + pitches[m]) / 2 if len(pitches) % 2 == 0 else pitches[m]

    # Prefer General MIDI bass family if available, else lowest-median instrument.
    bass_family_programs = set(range(32, 40))  # Fingered Bass .. Synth Bass 2
    gm_bass = [i for i in candidates if i.program in bass_family_programs]
    if gm_bass:
        return min(gm_bass, key=median_pitch)
    return min(candidates, key=median_pitch)


@dataclass
class BassNote:
    start: float  # seconds
    end: float    # seconds
    pitch: int
    velocity: int


@dataclass
class BasslineSample:
    midi_path: str
    bpm: float
    ppq: int
    time_signature: Optional[Tuple[int, int]]
    instrument_program: Optional[int]
    instrument_name: Optional[str]
    notes: List[BassNote]


def _first_time_signature(pm: pretty_midi.PrettyMIDI) -> Optional[Tuple[int, int]]:
    if pm.time_signature_changes:
        sig = pm.time_signature_changes[0]
        return int(sig.numerator), int(sig.denominator)
    return None


def extract_bassline_and_tempo(midi_path: Path) -> BasslineSample:
    """Parse `midi_path` and return bassline + representative tempo."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    bpm = extract_true_tempo(midi_path)
    ts = _first_time_signature(pm)
    inst = _choose_bass_instrument(pm)

    notes: List[BassNote] = []
    program = None
    name = None
    if inst:
        program = int(inst.program)
        try:
            name = pretty_midi.program_to_instrument_name(program)
        except Exception:
            name = None
        # Use notes as-is; no quantization, preserve durations
        notes = [
            BassNote(start=float(n.start), end=float(n.end), pitch=int(n.pitch), velocity=int(n.velocity))
            for n in sorted(inst.notes, key=lambda n: (n.start, n.pitch))
        ]

    return BasslineSample(
        midi_path=str(midi_path),
        bpm=float(bpm),
        ppq=int(pm.resolution),
        time_signature=ts,
        instrument_program=program,
        instrument_name=name,
        notes=notes,
    )


# --------------------------- I/O and CLI ---------------------------

def save_bassline_json(sample: BasslineSample, out_path: Path) -> None:
    """Write BasslineSample to JSON."""
    def encode(obj):
        if isinstance(obj, (BasslineSample, BassNote)):
            return asdict(obj)
        raise TypeError(f"Unserializable: {type(obj)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sample, f, default=encode, ensure_ascii=False, indent=2)


def process_tree(src: Path, dst: Path, dry_run: bool = False) -> None:
    """Walk `src` recursively, writing one JSON per MIDI into `dst`, mirroring structure."""
    midi_files = list(src.rglob("*.mid"))
    if not midi_files:
        print(f"No MIDI files found under: {src}")
        return

    for midi_path in midi_files:
        rel = midi_path.relative_to(src)
        out_json = (dst / rel.parent / (midi_path.stem + ".json")).with_suffix(".json")
        if dry_run:
            print(f"[DRY-RUN] {midi_path} -> {out_json}")
            continue
        try:
            sample = extract_bassline_and_tempo(midi_path)
        except Exception as e:
            print(f"Skipped {midi_path}: {e}")
            continue
        save_bassline_json(sample, out_json)
        print(f"Saved: {out_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract basslines and representative tempi from MIDI files."
    )
    parser.add_argument("--src", type=Path, default=Path("lmd_deduplicated"),
                        help="Source directory of MIDI files.")
    parser.add_argument("--dst", type=Path, default=Path("output_basslines"),
                        help="Destination directory for JSON outputs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned outputs without writing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_tree(args.src, args.dst, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
