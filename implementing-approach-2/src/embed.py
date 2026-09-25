"""Dense retrieval + embedding similarity (Approach-4 first rung).

Bi-encoder (default paraphrase-multilingual-MiniLM-L12-v2, 118M) with
Qwen3-Embedding-0.6B as the Kaggle-GPU upgrade (MODEL_NAME env).
Embeddings cached per (model, country, split) so notebooks never re-encode.
"""
import os
from pathlib import Path

MODEL_NAME = os.environ.get("EMB_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")


def ensure_model(name=None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "sentence-transformers"])
        from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name or MODEL_NAME)


def cache_path(out_dir, country, split, model=None):
    safe = (model or MODEL_NAME).replace("/", "_")
    return Path(out_dir) / f"emb_{safe}_{country}_{split}.npy"


def encode_texts(model, texts, batch=256):
    import numpy as np
    emb = model.encode(texts, batch_size=batch, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(emb, dtype="float32")


def encode_or_load(model, texts, ids, out_dir, country, split, batch=256):
    """Encode `texts` or load cache. Returns (emb matrix, ids). Order = input order."""
    import numpy as np
    cp = cache_path(out_dir, country, split)
    ids_p = Path(str(cp) + ".ids.json")
    if cp.exists() and ids_p.exists():
        import json
        if json.load(open(ids_p)) == ids:
            return np.load(str(cp)), ids
    emb = encode_texts(model, texts, batch=batch)
    np.save(str(cp), emb)
    import json
    json.dump(ids, open(ids_p, "w"))
    return emb, ids


def topk_cosine(q_emb, q_ids, d_emb, d_ids, topk, chunk=1000):
    """Normalized embeddings -> inner product = cosine. Chunked over queries."""
    import numpy as np
    import polars as pl
    o1, om, os = [], [], []
    for off in range(0, len(q_ids), chunk):
        sim = q_emb[off:off + chunk] @ d_emb.T
        for i in range(sim.shape[0]):
            row = sim[i]
            if len(row) > topk:
                idx = np.argpartition(row, -topk)[-topk:]
            else:
                idx = np.arange(len(row))
            for j in idx:
                o1.append(q_ids[off + i])
                om.append(d_ids[j])
                os.append(round(float(row[j]), 4))
    return pl.DataFrame({"s1": o1, "mid": om, "dense_score": os,
                         "from_dense": [1] * len(o1)})
