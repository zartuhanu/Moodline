#!/usr/bin/env python3
"""
Build a flat token corpus and per-song feature table from MIDI-derived bassline JSONs.

Each input JSON is expected to contain:
    {
      "tempo": 120.0,
      "grid_subdiv": 4,
      "bassline": [{"start_tick": 0, "duration_tick": 4, "pitch": 48, "velocity": 80}, ...],
      "tonic": "C",
      "mode": "major"
    }

Outputs (written under `data/`):
    data/tokens.txt      – one space-separated token sequence per song
    data/features.csv    – per-song metadata (tempo, mode, tonic, density)
"""

import json
import csv
from pathlib import Path

# ---------- I/O locations ----------
JSON_ROOT = Path("output_basslines")   # root folder containing artist subfolders of .json
OUT_DIR   = Path("data")               # where to save outputs
TOKENS_FILE   = OUT_DIR / "tokens.txt"
FEATURES_FILE = OUT_DIR / "features.csv"


# ---------- vocab helpers ----------
def tempo_bucket(bpm: float) -> str:
    """Return a coarse tempo bucket token based on BPM."""
    if bpm < 80:
        return "<TEMPO_SLOW>"
    if bpm < 100:
        return "<TEMPO_80_99>"
    if bpm < 120:
        return "<TEMPO_100_119>"
    if bpm < 140:
        return "<TEMPO_120_139>"
    return "<TEMPO_140P>"


def key_token(tonic: str, mode: str) -> str:
    """Return key token such as <KEY_C> or <KEY_Am>."""
    mode = (mode or "").lower()
    is_minor = mode.startswith("min")
    t = (tonic or "C").upper().replace("#", "S").replace("B", "b")  # C#, Bb → CS, Bb
    return f"<KEY_{t}{'m' if is_minor else ''}>"


def vel_bin(v: int, bins: int = 8) -> int:
    """Quantize MIDI velocity 0–127 into one of N bins."""
    step = 128 / bins
    return max(0, min(bins - 1, int(v // step)))


def ts_tokens(delta: int, max_ts: int = 32):
    """Return a list of TS_* tokens representing a time shift of `delta` ticks."""
    toks = []
    while delta > 0:
        step = min(delta, max_ts)
        toks.append(f"TS_{step}")
        delta -= step
    return toks


def dur_token(d: int) -> str:
    """Duration token."""
    return f"DUR_{d}"


def pitch_token(p: int) -> str:
    """Pitch token."""
    return f"PITCH_{p}"


def vel_token(vbin: int) -> str:
    """Velocity token."""
    return f"VEL_{vbin}"


# ---------- feature helpers ----------
def estimate_bars(events, grid_subdiv: int) -> float:
    """Estimate number of 4/4 bars given total ticks and grid resolution."""
    if not events:
        return 1.0
    end_tick = max(e["start_tick"] + e["duration_tick"] for e in events)
    ticks_per_bar = grid_subdiv * 4  # assume 4/4
    return max(1.0, end_tick / ticks_per_bar)


# ---------- tokenization ----------
def tokenize_song(json_path: Path):
    """Convert one bassline JSON into a token sequence and metadata row."""
    data = json.loads(json_path.read_text())
    tempo = float(data["tempo"])
    grid = int(data.get("grid_subdiv", 4))
    events = sorted(data["bassline"], key=lambda e: (e["start_tick"], e["pitch"]))
    tonic = data.get("tonic", "C")
    mode = data.get("mode", "major")

    tokens = ["<CLS>", key_token(tonic, mode), tempo_bucket(tempo), "<SONG_START>"]
    t = 0
    for e in events:
        start = int(e["start_tick"])
        dur = int(e["duration_tick"])
        pitch = int(e["pitch"])
        vel = int(e["velocity"])
        if start > t:
            tokens += ts_tokens(start - t)
        tokens += [pitch_token(pitch), dur_token(dur), vel_token(vel_bin(vel))]
        t = start + dur
    tokens.append("<SONG_END>")

    bars = estimate_bars(events, grid)
    density = len(events) / bars if bars > 0 else float(len(events))

    meta = {
        "json_path": str(json_path),
        "tempo": tempo,
        "mode": mode,
        "tonic": tonic,
        "density": round(density, 3),
    }
    return tokens, meta


# ---------- main ----------
def main():
    """Tokenize all bassline JSONs and write combined token and feature files."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokens_out = TOKENS_FILE.open("w", encoding="utf-8")
    features_out = FEATURES_FILE.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        features_out, fieldnames=["json_path", "tempo", "mode", "tonic", "density"]
    )
    writer.writeheader()

    count = 0
    for artist_dir in sorted([p for p in JSON_ROOT.iterdir() if p.is_dir()]):
        for jp in sorted(artist_dir.glob("*.json")):
            try:
                toks, meta = tokenize_song(jp)
            except Exception as e:
                print(f"[WARN] {jp}: {e}")
                continue
            tokens_out.write(" ".join(toks) + "\n")
            writer.writerow(meta)
            count += 1

    tokens_out.close()
    features_out.close()
    print(f"Tokenized {count} songs")
    print(f"   Tokens written to: {TOKENS_FILE}")
    print(f"   Features written to: {FEATURES_FILE}")


if __name__ == "__main__":
    main()
