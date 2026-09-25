"""Generate 03a_dense_blocking.ipynb: MiniLM dense pass + recall delta (India first)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

c00 = nbf.v4.new_markdown_cell("""# 03a — Dense Retrieval Pass (Approach 4, first rung)

paraphrase-multilingual-MiniLM-L12-v2 embeddings (cached by `encode_demo.py`) add a dense
top-K pass over the India demo pools. Measures marginal recall vs the lexical union.
Qwen3-Embedding-0.6B is the drop-in upgrade (`EMB_MODEL` env) for Kaggle-GPU.""")

c01 = nbf.v4.new_code_cell(r"""import sys, json
from pathlib import Path
HERE = Path.cwd()
SRC = HERE / "src" if (HERE / "src").exists() else HERE / "implementing-approach-2" / "src"
sys.path.insert(0, str(SRC))
import config as C
import polars as pl
import numpy as np
from embed import cache_path
from metrics import blocking_recall
SPL = json.load(open(C.OUT / "splits_gtin.json"))
CAND = {c: pl.read_parquet(str(C.OUT / f"candidates_{c}.parquet")) for c in C.COUNTRIES_TRAIN}
print("candidates loaded:", {c: len(CAND[c]) for c in CAND})
""")

c02 = nbf.v4.new_code_cell(r"""# ---- Dense top-K retrieval (India; cached embeddings, no re-encode) ----
from embed import topk_cosine
c = "India"
E, I = {}, {}
for split in ["s1", "s2", "s3"]:
    E[split] = np.load(str(cache_path(str(C.OUT), c, split)))
    I[split] = json.load(open(str(cache_path(str(C.OUT), c, split)) + ".ids.json"))
print({k: v.shape for k, v in E.items()})
# demo S1s = first 3000 (02a DEMO_S1 head) — align query subset to those ids
demo_ids = pl.read_parquet(str(C.PROC / f"train_s1_{c}.parquet")).head(3000)["entity_id"].to_list()
qidx = [I["s1"].index(x) for x in demo_ids]
Q, QIDS = E["s1"][qidx], demo_ids
D = np.concatenate([E["s2"], E["s3"]])
DIDS = I["s2"] + I["s3"]
dense = topk_cosine(Q, QIDS, D, DIDS, topk=20)
dense.write_parquet(str(C.OUT / "dense_India.parquet"))
print("dense pairs:", len(dense))
""")

c03 = nbf.v4.new_code_cell(r"""# ---- Recall delta: lexical union vs +dense ----
import json as _j
c = "India"
s1ids = set(CAND[c]["s1"].to_list())
gtin = {s: set(SPL[c]["gtin"].get(s, [])) for s in s1ids}
base = CAND[c].select(["s1", "mid"])
both = base.vstack(dense.select(["s1", "mid"])).unique()
r0 = blocking_recall(base, gtin, s1ids)
r1 = blocking_recall(both, gtin, s1ids)
new = 0
got0 = {}
for s, m in zip(base["s1"].to_list(), base["mid"].to_list()):
    got0.setdefault(s, set()).add(m)
for s in s1ids:
    dmids = set(dense.filter(pl.col("s1") == s)["mid"].to_list())
    new += len((gtin.get(s, set()) - got0.get(s, set())) & dmids)
print(f"recall lexical={r0:.4f} +dense={r1:.4f} newly_retrieved_true={new}")
both.write_parquet(str(C.OUT / "candidates_India_dense.parquet"))
_j.dump({"recall_lexical": round(r0, 4), "recall_plus_dense": round(r1, 4),
         "new_true": int(new), "dense_pairs": len(dense)},
        open(C.OUT / "metrics_03a.json", "w"), indent=1)
print("saved dense pairs + metrics_03a.json")
""")

nb.cells = [c00, c01, c02, c03]
nbf.write(nb, "03a_dense_blocking.ipynb")
print("wrote 03a_dense_blocking.ipynb")
