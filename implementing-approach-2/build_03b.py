"""Generate 03b_boost.ipynb: cosine features + retrains, Ditto augmentation, honest report."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

c00 = nbf.v4.new_markdown_cell("""# 03b — Boost: Cosine Features + Augmentation (India)

V0 baseline → V1 +emb_cosine → V2 +Ditto-augmentation → V3 both. Primary metric: pair F0.5;
entity macro reported with in-sample scope label. All India-only (the recall gap).""")

c01 = nbf.v4.new_code_cell(r"""import sys, json
from pathlib import Path
HERE = Path.cwd()
SRC = HERE / "src" if (HERE / "src").exists() else HERE / "implementing-approach-2" / "src"
sys.path.insert(0, str(SRC))
import config as C
import polars as pl
import numpy as np
SPL = json.load(open(C.OUT / "splits_gtin.json"))
FEAT = pl.read_parquet(str(C.OUT / "features_India.parquet"))
CAND = pl.read_parquet(str(C.OUT / "candidates_India.parquet"))
DENSE = pl.read_parquet(str(C.OUT / "dense_India.parquet"))
FCOLS = [c for c in FEAT.columns if c not in ("s1", "mid", "label")]
print("feat pairs:", len(FEAT), "dense pairs:", len(DENSE), "ncols:", len(FCOLS))
""")

c02 = nbf.v4.new_code_cell(r"""# ---- Cosine features for union pairs (cached embeddings) ----
from embed import cache_path
E, I = {}, {}
for split in ["s1", "s2", "s3"]:
    E[split] = np.load(str(cache_path(str(C.OUT), "India", split)))
    I[split] = json.load(open(str(cache_path(str(C.OUT), "India", split)) + ".ids.json"))
pos = {sid: i for i, sid in enumerate(I["s1"])}
pid = {}
for split in ["s2", "s3"]:
    for i, sid in enumerate(I[split]):
        pid[sid] = (split, i)
U = CAND.select(["s1", "mid"]).vstack(DENSE.select(["s1", "mid"])).unique()
s1l, midl = U["s1"].to_list(), U["mid"].to_list()
cos = np.empty(len(s1l), dtype="float32")
B = 20000
for off in range(0, len(s1l), B):
    sl = s1l[off:off + B]
    Q = np.stack([E["s1"][pos[s]] for s in sl])
    D = np.stack([E[pid[m][0]][pid[m][1]] for m in midl[off:off + B]])
    cos[off:off + B] = (Q * D).sum(axis=1)
U = U.with_columns(pl.Series("emb_cosine", cos))
U.write_parquet(str(C.OUT / "union_India_dense.parquet"))
print("union pairs:", len(U), "cos mean:", round(float(cos.mean()), 3))
""")

c03 = nbf.v4.new_code_cell(r"""# ---- Retrains: V0 baseline / V1 +emb / V2 +aug / V3 both ----
from model import train, predict
from metrics import pair_prf
from features import build_lookup
from blocking import load_frame
from augment import augment_positives
import polars as _pl
va = set(SPL["India"]["va_ids"])
base = FEAT.join(U.select(["s1", "mid", "emb_cosine"]), on=["s1", "mid"], how="left")
base = base.with_columns(pl.col("emb_cosine").fill_null(0.0))
s1l = base["s1"].to_list()
tr_idx = [i for i, s in enumerate(s1l) if s not in va]
va_idx = [i for i, s in enumerate(s1l) if s in va]
y = base["label"].to_numpy()
RES = {}
def run_variant(name, cols, extra=None):
    F = base
    tr, vv = list(tr_idx), list(va_idx)
    if extra is not None:
        F = _pl.concat([base, extra], how="diagonal")
        s2 = F["s1"].to_list()
        tr = [i for i, s in enumerate(s2) if s not in va]
        vv = [i for i, s in enumerate(s2) if s in va]
    X = F.select(cols).to_numpy()
    yy = F["label"].to_numpy()
    m = train(X[tr], yy[tr], X[vv], yy[vv], C.LGBM_PARAMS, C.LGBM_ROUNDS, C.LGBM_EARLY)
    pv = predict(m, X[vv])
    p, r, f = pair_prf(yy[vv], (pv >= 0.5).astype(int))
    RES[name] = {"P": round(p, 4), "R": round(r, 4), "F0.5": round(f, 4), "iter": m.best_iteration}
    print(f"{name}: P={p:.3f} R={r:.3f} F0.5={f:.3f} iter={m.best_iteration}")
    return m
FC = FCOLS + ["emb_cosine"]
run_variant("V0_baseline", FCOLS)
run_variant("V1_+emb", FC)
# augmentation on train positives (Ditto ops: del/swap/attr_del x3)
s1f = load_frame(C.PROC / "train_s1_India.parquet").head(3000)
s2f = load_frame(C.PROC / "train_s2_India.parquet")
s3f = load_frame(C.PROC / "train_s3_India.parquet")
cd = _pl.concat([s2f, s3f], how="diagonal")
s1map, sc = build_lookup(s1f)
cdmap, cc = build_lookup(cd)
pos = base.filter((pl.col("s1").is_in([s for s in base["s1"].to_list() if s not in va])) &
                  (pl.col("label") == 1))
AUG = augment_positives(pos, s1map, cdmap, sc, cc, reps=3, seed=C.SEED)
_cos = {(s, m): c for s, m, c in zip(base["s1"].to_list(), base["mid"].to_list(),
                                    base["emb_cosine"].to_list())}
AUG = AUG.with_columns(pl.Series("emb_cosine",
    [_cos.get((r["s1"], r["mid"].replace("#aug", "")), 0.0) for r in AUG.to_dicts()]))
for _c in base.columns:
    if AUG[_c].dtype != base.schema[_c]:
        AUG = AUG.with_columns(pl.col(_c).cast(base.schema[_c]))
print("augmented positives:", len(AUG))
run_variant("V2_+aug", FCOLS, extra=AUG.select(FC + ["label", "s1", "mid"]))
run_variant("V3_both", FC, extra=AUG.select(FC + ["label", "s1", "mid"]))
json.dump(RES, open(C.OUT / "metrics_03b.json", "w"), indent=1)
""")

c04 = nbf.v4.new_code_cell(r"""# ---- Honest report ----
print("pair-F0.5 table (India, retrieved pairs):")
for k, v in RES.items():
    print(f"  {k:12s} P={v['P']} R={v['R']} F0.5={v['F0.5']}")
print("entity macro-F0.5 on demo universe remains ~0.97+ but is pseudo-singleton-inflated;")
print("blocking recall: lexical 0.844 -> +dense 0.938 (metrics_03a.json).")
print("TRUE 0.97 macro requires FULL_RUN (real recall ceiling + real thresholds).")
""")

nb.cells = [c00, c01, c02, c03, c04]
nbf.write(nb, "03b_boost.ipynb")
print("wrote 03b_boost.ipynb")
