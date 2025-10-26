#!/usr/bin/env python3
"""
Build a vocabulary (stoi/itos) from normalized token sequences and emit per-song ID lists.

I/O
---
Inputs (under data/):
    - data/tokens.txt
    - data/labels_humans_cleaned_v1.csv   (used only for json_path order/alignment)

Outputs (under data/):
    - data/vocab.json                     {"stoi": {...}, "itos": [...]}
    - data/ids.jsonl                      one JSON per line: {"json_path": ..., "ids": [...]}
"""

import json, csv, re, math
from pathlib import Path
from collections import Counter

# ================== I/O (edit as needed) ==================
TOKENS_TXT = Path("data/tokens.txt")                       # INPUT
LABELS_CSV = Path("data/labels_humans_cleaned_v1.csv")     # INPUT
OUT_DIR    = Path("data")                                  # OUTPUT ROOT
VOCAB_JSON = OUT_DIR / "vocab.json"                        # OUTPUT
IDS_JSONL  = OUT_DIR / "ids.jsonl"                         # OUTPUT
# ==========================================================

# Special tokens: keep CLS so the model can pool at position 0 if desired
SPECIALS = ["<PAD>", "<UNK>", "<CLS>"]

# Tokens to drop entirely (don’t help classification)
DROP_TOKENS = {"<SONG_START>", "<SONG_END>", "<BOS>", "<EOS>"}

# ---------- Normalization helpers ----------
_KEY_RE = re.compile(r"^<KEY_([^>]+)>$")        # e.g. <KEY_Eb>, <KEY_Am>, <KEY_E-m>, <KEY_b->
_DUR_RE = re.compile(r"^DUR_(\d+)$")            # e.g. DUR_523

# Valid roots and accidentals
_VALID_ROOTS = {"A","B","C","D","E","F","G"}
_ACCIDENTAL_MAP = {
    "#": "#", "♯": "#",    # sharps
    "b": "b", "♭": "b",    # flats
    "-": "b",              # some datasets use '-' for flat (E- -> Eb)
}

def _canonicalize_key_token(tok: str):
    """
    Map messy <KEY_*> tokens into canonical <KEY_[A-G][b/#]?m?>.
    Return canonical token string or None to drop if invalid.
    Examples:
      <KEY_E->    -> <KEY_Eb>
      <KEY_E-m>   -> <KEY_Ebm>
      <KEY_Am>    -> <KEY_Am>
      <KEY_error> -> None (drop)
      <KEY_b->    -> None (invalid)
    """
    m = _KEY_RE.match(tok)
    if not m:
        return tok  # not a key token; return as-is
    body = m.group(1).strip()

    # Obvious errors
    if body.lower() in {"error", "err", "unk", "unknown", "none", ""}:
        return None

    # Normalize casing and separators
    body = body.replace(" ", "").replace("_", "")
    if not body:
        return None

    # Extract root letter
    root = body[0].upper()
    if root not in _VALID_ROOTS:
        return None

    rest = body[1:]
    # minor flag if trailing 'm'
    is_minor = rest.lower().endswith("m")
    if is_minor:
        rest = rest[:-1]

    accidental = ""
    if rest:
        sym = rest[0]
        accidental = _ACCIDENTAL_MAP.get(sym, "")
        if accidental == "" and sym:
            return None  # unrecognized accidental/junk

    canon = f"<KEY_{root}{accidental}{'m' if is_minor else ''}>"
    return canon

def _bucket_duration(d: int) -> int:
    """Bucket duration integer to nearest power-of-two (reduces sparsity)."""
    if d <= 0:
        return 1
    return int(2 ** round(math.log2(d)))

def _normalize_token(tok: str):
    """Return normalized token or None to drop."""
    tok = tok.strip()
    if not tok or tok in DROP_TOKENS:
        return None

    # Normalize key tokens
    if tok.startswith("<KEY_"):
        return _canonicalize_key_token(tok)

    # Normalize duration bins
    m = _DUR_RE.match(tok)
    if m:
        val = int(m.group(1))
        bucket = _bucket_duration(val)
        return f"DUR_{bucket}"

    # Everything else: pass through
    return tok

# ------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load json_path order from labels (we align to this)
    json_paths = []
    with LABELS_CSV.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            json_paths.append(row["json_path"])

    lines = TOKENS_TXT.read_text(encoding="utf-8").splitlines()
    n_labels = len(json_paths)
    n_tokens = len(lines)
    if n_tokens != n_labels:
        print(f"tokens.txt lines ({n_tokens}) != labels rows ({n_labels})")
        if n_tokens > n_labels:
            print(f"→ Will use the first {n_labels} token lines and ignore {n_tokens - n_labels} extras.")
        else:
            print(f"→ Will use only {n_tokens} rows from labels (there are {n_labels - n_tokens} extra labels).")
    else:
        print(f"tokens and labels row counts match: {n_tokens}")

    min_len = min(n_tokens, n_labels)

    # First pass: normalize tokens and collect sequences
    token_seqs = []
    dur_before, dur_after = Counter(), Counter()
    key_before, key_after = Counter(), Counter()
    dropped_misc = 0

    for i in range(min_len):
        raw_toks = lines[i].strip().split()
        norm_toks = []
        for t in raw_toks:
            # stats before
            if t.startswith("DUR_"):
                m = _DUR_RE.match(t)
                if m:
                    dur_before[int(m.group(1))] += 1
            elif t.startswith("<KEY_"):
                key_before[t] += 1

            nt = _normalize_token(t)
            if nt is None:
                dropped_misc += 1
                continue
            norm_toks.append(nt)

            # stats after
            if nt.startswith("DUR_"):
                m2 = _DUR_RE.match(nt)
                if m2:
                    dur_after[int(m2.group(1))] += 1
            elif nt.startswith("<KEY_"):
                key_after[nt] += 1

        token_seqs.append(norm_toks)

    # Build vocab from normalized tokens (only what we'll actually use)
    counts = Counter()
    for toks in token_seqs:
        counts.update(toks)

    stoi = {}
    itos = []
    for tok in SPECIALS:
        if tok not in stoi:
            stoi[tok] = len(itos); itos.append(tok)
    for tok, _ in counts.most_common():
        if tok not in stoi:
            stoi[tok] = len(itos); itos.append(tok)

    VOCAB_JSON.write_text(json.dumps({"stoi": stoi, "itos": itos}, indent=2), encoding="utf-8")
    print(f"wrote {VOCAB_JSON} (size={len(stoi)})")

    # Write ids.jsonl (prepend <CLS> id)
    cls_id = stoi["<CLS>"]; unk = stoi["<UNK>"]
    with IDS_JSONL.open("w", encoding="utf-8") as f:
        for i in range(min_len):
            ids = [cls_id] + [stoi.get(t, unk) for t in token_seqs[i]]
            f.write(json.dumps({"json_path": json_paths[i], "ids": ids}) + "\n")
    print(f"wrote {IDS_JSONL} (entries={min_len})")

    # Safe alignment preview
    preview_idxs = [0,1,2, min_len-3, min_len-2, min_len-1]
    seen = set()
    print("\nAlignment preview (first/last up to 3):")
    for idx in preview_idxs:
        if 0 <= idx < min_len and idx not in seen:
            seen.add(idx)
            print(f"  [{idx}] {json_paths[idx]}  →  {len(token_seqs[idx])} tokens (normalized)")

    # Diagnostics
    if dropped_misc:
        print(f"\nDropped tokens during normalization: {dropped_misc}")
    if dur_before:
        def _topn(c, n=8):
            return ", ".join(f"{k}:{v}" for k,v in c.most_common(n))
        print(f"dur bins before (top): {_topn(dur_before)}")
        print(f"dur bins after  (top): {_topn(dur_after)}")
    if key_before:
        def _topn(c, n=12):
            return ", ".join(f"{k}:{v}" for k,v in c.most_common(n))
        print(f"key tokens before (top): {_topn(key_before)}")
        print(f"key tokens after  (top): {_topn(key_after)}")

if __name__ == "__main__":
    main()
