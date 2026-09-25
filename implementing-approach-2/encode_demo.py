"""One-shot demo encoding (India): combined 'core | addr' strings for S1/S2/S3 demo pools.

Caches to outputs/emb_*.npy (+ .ids.json) via src/embed.py; notebooks load cache instantly.
Run: python3 encode_demo.py   (~10-14 min CPU, one model load)
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
import torch

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
import config as C
from blocking import load_frame
from embed import ensure_model, encode_or_load

t0 = time.time()
model = ensure_model()
print("model loaded in %.0fs" % (time.time() - t0), flush=True)

jobs = [("train_s1_India.parquet", "s1", 3000),
        ("train_s2_India.parquet", "s2", None),
        ("train_s3_India.parquet", "s3", None)]
for fname, split, head in jobs:
    df = load_frame(C.PROC / fname)
    if head:
        df = df.head(head)
    texts = [(c or "") + " | " + (a or "") for c, a in
             zip(df["core_name"].to_list(), df["addr_norm"].to_list())]
    ids = df["entity_id"].to_list()
    t1 = time.time()
    emb, _ = encode_or_load(model, texts, ids, str(C.OUT), "India", split, batch=512)
    print(f"{split}: {len(ids)} in {time.time()-t1:.0f}s shape={emb.shape}", flush=True)
print("ALL ENCODED in %.0fs" % (time.time() - t0), flush=True)
