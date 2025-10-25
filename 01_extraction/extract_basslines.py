import pretty_midi
import os
import json
from pathlib import Path

import mido
import math


def extract_true_tempo(midi_path, min_bpm=40, max_bpm=240):
    """
    Reads all set_tempo meta messages to find the BPM that lasts longest.
    """
    midi = mido.MidiFile(str(midi_path))  # ensure correct parsing
    events = []

    # Gather all (tick, microseconds_per_quarter) for set_tempo messages
    for track in midi.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'set_tempo':
                events.append((abs_tick, msg.tempo))  # tempo in µs/qn

    if not events:
        # No explicit tempo events => default 120 BPM
        return 120.0

    # Sort by tick and compute how long each tempo lasts
    events.sort(key=lambda x: x[0])
    end_tick = midi.length * midi.ticks_per_beat  # approximate end in ticks
    durations = []
    for i, (tick, us_per_qn) in enumerate(events):
        next_tick = events[i+1][0] if i+1 < len(events) else end_tick
        durations.append((us_per_qn, next_tick - tick))

    # Choose tempo with maximum span
    best_us_per_qn, _ = max(durations, key=lambda x: x[1])
    bpm = 60000000.0 / best_us_per_qn

    # Octave-correct into a musical range
    while bpm > max_bpm:
        bpm /= 2.0
    while bpm < min_bpm:
        bpm *= 2.0

    return round(bpm, 2)


def extract_bassline_and_tempo(midi_path, grid_subdiv=4):
    """
    Extracts the true tempo and bassline notes from a MIDI file,
    then quantizes start/duration to 16th-note ticks (grid_subdiv per quarter).
    Returns a dict: { "tempo": float, "grid_subdiv": int, "bassline": [ ... ] }
    """
    # 1) Extract the track's true tempo
    tempo = extract_true_tempo(midi_path)

    # 2) Load MIDI for note data
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))

    # 3) Identify bass notes (GM bass programs 33–40 → 32–39 zero-indexed)
    bass_notes = []
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        if 32 <= inst.program <= 39:
            bass_notes.extend(inst.notes)

    # Fallback: lowest-pitch heuristic
    if not bass_notes:
        all_notes = []
        for inst in midi_data.instruments:
            if not inst.is_drum:
                all_notes.extend(inst.notes)
        all_notes.sort(key=lambda n: n.pitch)
        bass_notes = all_notes[: len(all_notes)//4] if all_notes else []

    # 4) Quantize to 16th-note ticks
    sec_per_quarter = 60.0 / tempo
    ticks_per_quarter = grid_subdiv
    sec_per_tick = sec_per_quarter / ticks_per_quarter

    bassline = []
    for note in bass_notes:
        raw_start = note.start / sec_per_tick
        raw_dur   = (note.end - note.start) / sec_per_tick

        q_start = int(round(raw_start))
        q_dur   = max(1, int(round(raw_dur)))

        bassline.append({
            "start_tick":    q_start,
            "duration_tick": q_dur,
            "pitch":         note.pitch,
            "velocity":      note.velocity
        })

    return {
        "tempo":      tempo,
        "grid_subdiv": grid_subdiv,
        "bassline":   bassline
    }


def save_bassline_json(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    # ——— Update these two paths ———
    dataset_dir = Path("lmd_deduplicated")
    output_dir  = Path("output_basslines")
    # ————————————————————————
    output_dir.mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if not file.lower().endswith(".mid"):
                continue

            midi_path = Path(root) / file
            rel       = midi_path.relative_to(dataset_dir)
            out_sub   = output_dir / rel.parent
            out_sub.mkdir(parents=True, exist_ok=True)
            out_json  = out_sub / (midi_path.stem + ".json")

            try:
                data = extract_bassline_and_tempo(midi_path)
            except Exception as e:
                print(f"⚠️  Skipped {midi_path}: {e}")
                continue

            save_bassline_json(data, out_json)
            print(f"✅  Saved: {out_json}")

if __name__ == "__main__":
    main()