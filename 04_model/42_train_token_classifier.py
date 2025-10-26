#!/usr/bin/env python3
"""
Train a lightweight token-sequence classifier and export embeddings.

I/O
---
Inputs (under data/):
    - data/ids.jsonl                     (ids per song; from vocab/ids builder)
    - data/vocab.json                    (stoi/itos)
    - data/labels_human_merged.csv       (original label table → row order for indices)
    - data/labels_humans_cleaned_v1.csv  (cleaned labels overlay; optional but recommended)
    - data/splits_cleaned.json           (indices for train/val/test after cleanup)
        # If you didn't run the cleanup step, use data/splits.json instead.

Outputs:
    - data/artifacts_token_clf_lenghtier/
        - model.pt            (best checkpoint by validation macro-F1)
        - _best_metric.txt    (best validation macro-F1)
        - embeddings.npy      (pooled song embeddings for train+val+test)
        - songs.csv           (json_path, split) aligned with embeddings.npy
"""

import json, csv, math, random, sys
from pathlib import Path
from collections import Counter
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

# ================== PATHS / SETTINGS ==================
DATA_DIR    = Path("data")
IDS_JSONL   = DATA_DIR / "ids.jsonl"
VOCAB_JSON  = DATA_DIR / "vocab.json"

# ORIGINAL labels for row order; CLEANED overlays moods (and filters)
LABELS_CSV         = DATA_DIR / "labels_human_merged.csv"
CLEANED_LABELS_CSV = DATA_DIR / "labels_humans_cleaned_v1.csv"

# Use the cleaned splits produced after label cleanup
SPLITS_JSON = DATA_DIR / "splits_cleaned.json"

# Model/artifacts output
OUT_DIR     = DATA_DIR / "artifacts_token_clf_lenghtier"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ======================================================

# ================== Training config ===================
EXCLUDE_NEUTRAL = True          # drop neutral rows
MAX_LEN    = 1024
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-3
D_MODEL    = 256
DROP       = 0.1
USE_TINY_ATTENTION = True
SUBSAMPLE_FRAC = 1.0
SUBSAMPLE_SEED = 123
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 1337

# Balancing
USE_SAMPLER = True
USE_CLASS_WEIGHTS = False
LABEL_SMOOTH = 0.10
# ======================================================

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device_and_amp():
    if torch.cuda.is_available():
        return torch.device("cuda"), lambda: torch.amp.autocast(device_type="cuda", dtype=torch.float16), "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), lambda: torch.amp.autocast(device_type="mps", dtype=torch.bfloat16), "mps"
    return torch.device("cpu"), nullcontext, "cpu"

def read_ids():
    ids = {}
    with IDS_JSONL.open("r", encoding="utf-8") as f:
        for ln in f:
            obj = json.loads(ln)
            jp = obj["json_path"]; vec = obj["ids"]
            ids[jp] = vec
            ids[Path(jp).name] = vec
    return ids

def load_splits():
    obj = json.loads(SPLITS_JSON.read_text(encoding="utf-8"))
    return obj["indices"]

def read_labels_with_overlay():
    # original rows (row order for splits)
    base_rows = []
    with LABELS_CSV.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            base_rows.append({"json_path": row["json_path"], "mood": row["mood"].strip().lower()})

    # cleaned overlay (optional)
    cleaned_map = None
    if CLEANED_LABELS_CSV.exists():
        cleaned_map = {}
        with CLEANED_LABELS_CSV.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                cleaned_map[row["json_path"]] = row["mood"].strip().lower()
        mood_set = set(cleaned_map.values())
    else:
        mood_set = set(x["mood"] for x in base_rows)

    if EXCLUDE_NEUTRAL and "neutral" in mood_set:
        mood_set.remove("neutral")

    order = sorted(mood_set)
    label2id = {m:i for i,m in enumerate(order)}
    return base_rows, label2id, cleaned_map

class TokenDataset(Dataset):
    def __init__(self, items, pad_id, max_len):
        self.items = items; self.pad_id = pad_id; self.max_len = max_len
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        ids = self.items[i]["ids"]; label = self.items[i]["label"]
        L = len(ids)
        if L > self.max_len:
            start = random.randint(0, L - self.max_len)
            seq = ids[start:start+self.max_len]
        else:
            seq = ids[:]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class PadCollate:
    def __init__(self, pad_id, max_len):
        self.pad_id = pad_id; self.max_len = max_len
    def __call__(self, batch):
        seqs, labels = zip(*batch)
        maxlen = min(self.max_len, max(1, max(x.size(0) for x in seqs)))
        out = []
        for s in seqs:
            s = s[:maxlen]
            if s.size(0) < maxlen:
                pad = torch.full((maxlen - s.size(0),), self.pad_id, dtype=torch.long)
                s = torch.cat([s, pad], dim=0)
            out.append(s)
        return torch.stack(out, 0), torch.stack(labels, 0)

class MeanPooler(nn.Module):
    def __init__(self, pad_id): super().__init__(); self.pad_id = pad_id
    def forward(self, x, tokens):
        mask = (tokens != self.pad_id).float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom

class TinySelfAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model); self.v = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout); self.proj = nn.Linear(d_model, d_model)
    def forward(self, x, mask=None):
        Q, K, V = self.q(x), self.k(x), self.v(x)
        att = (Q @ K.transpose(1,2)) / math.sqrt(Q.size(-1))
        if mask is not None: att = att.masked_fill(mask.unsqueeze(1)==0, -1e4)
        return self.proj(self.drop(att.softmax(dim=-1)) @ V)

class FastTokenClassifier(nn.Module):
    def __init__(self, vocab, n_classes, pad_id, d_model=D_MODEL, dropout=DROP, use_tiny_attn=USE_TINY_ATTENTION):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab, d_model, padding_idx=pad_id)
        self.ln = nn.LayerNorm(d_model)
        self.use_tiny = use_tiny_attn
        if use_tiny_attn:
            self.tiny = TinySelfAttention(d_model, dropout)
        self.pool = MeanPooler(pad_id)
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )
    def forward(self, tokens):
        x = self.embed(tokens); x = self.ln(x)
        mask = (tokens != self.pad_id).int()
        if self.use_tiny: x = self.tiny(x, mask) + x
        pooled = self.pool(x, tokens)
        return self.proj(pooled), pooled

@torch.no_grad()
def evaluate(model, loader, device, n_classes, amp_ctx):
    model.eval(); preds, trues = [], []
    pbar = tqdm(loader, desc="Eval", leave=False)
    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        with amp_ctx(): logit, _ = model(xb)
        preds.extend(logit.argmax(1).cpu().tolist()); trues.extend(yb.cpu().tolist())
    preds, trues = np.array(preds), np.array(trues)
    acc = float((preds == trues).mean()) if len(preds) else 0.0
    per_class = {}
    for c in range(n_classes):
        tp = int(((preds==c)&(trues==c)).sum()); fp = int(((preds==c)&(trues!=c)).sum()); fn = int(((preds!=c)&(trues==c)).sum())
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0; rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        per_class[c] = {"precision":prec,"recall":rec,"f1":f1,"support":int((trues==c).sum())}
    macro_f1 = np.mean([v["f1"] for v in per_class.values()]) if per_class else 0.0
    return {"acc":acc, "macro_f1":macro_f1, "per_class":per_class}

@torch.no_grad()
def export_embeddings(model, loader, device, out_npy, song_ids, pad_id, amp_ctx):
    model.eval(); all_vecs = []
    pbar = tqdm(loader, desc="Export embeddings", leave=False)
    for xb, _ in pbar:
        xb = xb.to(device, non_blocking=True)
        with amp_ctx(): _, pooled = model(xb)
        all_vecs.append(pooled.cpu().numpy())
    arr = np.concatenate(all_vecs, axis=0); np.save(out_npy, arr)
    with open(OUT_DIR/"songs.csv","w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["json_path","split"])
        for s in song_ids: w.writerow(s)

def class_balance(items, id2label):
    ctr = Counter(id2label[it["label"]] for it in items)
    total = max(1, len(items))
    return {k: (v, v/total) for k,v in sorted(ctr.items())}

def build_items(rows, ids_map, label2id, idx_list, split_name, cleaned_map=None):
    items = []; n = len(rows)
    dropped_out_of_range = dropped_not_in_clean = dropped_missing_ids = 0
    for i in idx_list:
        if i < 0 or i >= n: dropped_out_of_range += 1; continue
        jp = rows[i]["json_path"]
        if cleaned_map is not None:
            if jp not in cleaned_map: dropped_not_in_clean += 1; continue
            mood = cleaned_map[jp]
        else:
            mood = rows[i]["mood"]
        if EXCLUDE_NEUTRAL and mood == "neutral": continue
        if mood not in label2id: continue
        key = jp if jp in ids_map else Path(jp).name
        if key not in ids_map: dropped_missing_ids += 1; continue
        items.append({"ids": ids_map[key], "label": label2id[mood], "json_path": jp, "split": split_name})
    if cleaned_map is not None:
        print(f"[{split_name}] dropped_not_in_clean={dropped_not_in_clean} | missing_ids={dropped_missing_ids} | bad_idx={dropped_out_of_range}")
    else:
        print(f"[{split_name}] missing_ids={dropped_missing_ids} | bad_idx={dropped_out_of_range}")
    return items

def subsample(items, frac, seed):
    if frac >= 1.0: return items
    rng = random.Random(seed); items = items[:]; rng.shuffle(items)
    return items[:max(1, int(len(items) * frac))]

def compute_class_weights(items, n_classes):
    counts = [0]*n_classes
    for it in items: counts[it["label"]] += 1
    counts = np.array(counts) + 1e-6
    weights = counts.sum() / (counts * n_classes)
    return torch.tensor(weights, dtype=torch.float32)

def main():
    set_seed(SEED)
    device, amp_ctx, devtag = get_device_and_amp()
    print("device:", device)

    vocab = json.loads(VOCAB_JSON.read_text(encoding="utf-8"))
    stoi = vocab["stoi"]; vocab_size = len(stoi)
    pad_id = stoi["<PAD>"]

    ids_map = read_ids()
    rows, label2id, cleaned_map = read_labels_with_overlay()
    id2label = {v:k for k,v in label2id.items()}
    splits = load_splits()

    items_train = build_items(rows, ids_map, label2id, splits["train"], "train", cleaned_map)
    items_val   = build_items(rows, ids_map, label2id, splits["val"],   "val",   cleaned_map)
    items_test  = build_items(rows, ids_map, label2id, splits["test"],  "test",  cleaned_map)

    items_train = subsample(items_train, SUBSAMPLE_FRAC, SUBSAMPLE_SEED)
    items_val   = subsample(items_val,   SUBSAMPLE_FRAC, SUBSAMPLE_SEED)

    print(f"train items: {len(items_train)} | val: {len(items_val)} | test: {len(items_test)}")
    print("label map:", label2id)
    def fmt_bal(d): return {k: {"n":v[0], "p":f"{v[1]*100:.1f}%"} for k,v in d.items()}
    print("class balance (train):", fmt_bal(class_balance(items_train, id2label)))
    print("class balance (val):  ", fmt_bal(class_balance(items_val,   id2label)))
    print("class balance (test): ", fmt_bal(class_balance(items_test,  id2label)))

    # DataLoaders
    collate = PadCollate(pad_id=pad_id, max_len=MAX_LEN)
    ds_tr = TokenDataset(items_train, pad_id, MAX_LEN)
    ds_va = TokenDataset(items_val,   pad_id, MAX_LEN)
    ds_te = TokenDataset(items_test,  pad_id, MAX_LEN)

    if USE_SAMPLER:
        labels_vec = [it["label"] for it in items_train]
        class_counts = np.bincount(labels_vec, minlength=len(label2id))
        weights_per_class = 1.0 / (class_counts + 1e-6)
        sample_weights = [float(weights_per_class[l]) for l in labels_vec]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate, num_workers=0, pin_memory=(devtag=="cuda"))
    else:
        dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=0, pin_memory=(devtag=="cuda"))

    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=(devtag=="cuda"))
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=(devtag=="cuda"))

    # model / optim
    n_classes = len(label2id); assert n_classes >= 2
    model = FastTokenClassifier(vocab_size, n_classes, pad_id).to(device)

    # Option A: no class weights, use label smoothing
    weights = compute_class_weights(items_train, n_classes).to(device) if USE_CLASS_WEIGHTS else None
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTH)

    # Bias init to class priors (helps calibration)
    with torch.no_grad():
        counts = np.bincount([it["label"] for it in items_train], minlength=n_classes)
        priors = torch.tensor(counts / counts.sum(), dtype=torch.float32, device=model.proj[-1].bias.device).clamp_min(1e-6)
        model.proj[-1].bias.copy_(priors.log())

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    step = 0; best_f1 = -1.0

    for ep in range(1, EPOCHS+1):
        model.train()
        total = correct = 0; loss_sum = 0.0
        pbar = tqdm(dl_tr, desc=f"Epoch {ep}/{EPOCHS} (train)")
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            if step < WARMUP_STEPS:
                lr = LR * (step+1) / max(1, WARMUP_STEPS)
                for g in opt.param_groups: g["lr"] = lr
            with (torch.amp.autocast(device_type="cuda", dtype=torch.float16) if device.type=="cuda" else nullcontext()):
                logit, _ = model(xb)
                loss = crit(logit, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP is not None: nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            loss_sum += float(loss.item()) * xb.size(0)
            correct += (logit.argmax(1) == yb).sum().item()
            total += xb.size(0); step += 1
            pbar.set_postfix(loss=loss_sum/max(1,total), acc=f"{(correct/max(1,total)):.3f}")

        # Eval on val
        if device.type == "cpu":
            amp_val = nullcontext
        else:
            amp_val = (lambda: torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type=="cuda" else torch.bfloat16))
        val_metrics = evaluate(model, dl_va, device, n_classes, amp_val)
        tr_acc = correct / max(1,total)
        print(f"Epoch {ep:02d} | train acc {tr_acc:.3f} | val macroF1 {val_metrics['macro_f1']:.3f}")

        # Distribution sanity check
        preds, trues = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                logit, _ = model(xb)
                preds.extend(logit.argmax(1).cpu().tolist()); trues.extend(yb.tolist())
        print("val true counts:", dict(Counter(trues)))
        print("val pred counts:", dict(Counter(preds)))

        # Save best
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            ckpt_path = OUT_DIR/"model.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "label2id": {k:int(v) for k,v in label2id.items()},
                "vocab_size": len(vocab["stoi"]),
                "config": {"D_MODEL":D_MODEL,"MAX_LEN":MAX_LEN,"EXCLUDE_NEUTRAL":EXCLUDE_NEUTRAL,"USE_TINY_ATTENTION":USE_TINY_ATTENTION}
            }, ckpt_path)
            (OUT_DIR/"_best_metric.txt").write_text(str(best_f1))
            print("saved best model")

    # Final test eval
    ckpt = torch.load(OUT_DIR/"model.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, dl_te, device, n_classes, nullcontext)
    print("\n=== TEST RESULTS ===")
    print(f"acc: {test_metrics['acc']:.3f} | macro-F1: {test_metrics['macro_f1']:.3f}")
    id2label_sorted = sorted(ckpt["label2id"].items(), key=lambda x: x[1])
    for lbl, idx in id2label_sorted:
        m = test_metrics["per_class"][idx]
        print(f"  {lbl:18s} F1={m['f1']:.3f}  (P={m['precision']:.3f}, R={m['recall']:.3f}, n={m['support']})")

    # Export embeddings (train + val + test)
    all_items = items_train + items_val + items_test
    dl_all = DataLoader(TokenDataset(all_items, pad_id, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=PadCollate(pad_id, MAX_LEN), num_workers=0, pin_memory=(device.type=="cuda"))
    song_ids = [(it["json_path"], it["split"]) for it in all_items]
    export_embeddings(model, dl_all, device, OUT_DIR/"embeddings.npy", song_ids, pad_id, nullcontext)
    print(f"\nsaved embeddings to {OUT_DIR/'embeddings.npy'} and song list to {OUT_DIR/'songs.csv'}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
