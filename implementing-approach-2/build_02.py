"""Generate 02_approach2.ipynb via nbformat (valid JSON guaranteed)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

c00 = nbf.v4.new_markdown_cell("""# 02 — Approach 2: Supervised Tournament Core (demo)

Blocking (exact + TF-IDF + address) → lexical/address features → LightGBM →
vetoes + sparsity-aware thresholds → one-owner/singleton decisions → F0.5.

- **Demo mode (default):** 6k train S1 per country (US/India) vs 40k S2/S3 samples. Finishes in minutes.
- **FULL_RUN=true:** same code over all processed rows (Kaggle/long local run).
- Test inference uses the identical path once full test parquet exists; here the submission-format
  demo runs on validation S1s (same schema + validator-style checks as `matching_results.tsv`).""")

c01 = nbf.v4.new_code_cell(r"""import os, sys, json
from pathlib import Path
try:
    import lightgbm, sklearn
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm", "scikit-learn"])
HERE = Path.cwd()
SRC = HERE / "src" if (HERE / "src").exists() else HERE / "implementing-approach-2" / "src"
sys.path.insert(0, str(SRC))
import config as C
import polars as pl
print("ROOT=", C.ROOT, "PROC=", C.PROC, "exists=", C.PROC.exists())
print("GT=", C.GT, "exists=", C.GT.exists())
FULL = os.environ.get("FULL_RUN", "false").lower() == "true"
DEMO_S1 = 6000 if not FULL else 10**18
print("FULL_RUN=", FULL, "DEMO_S1=", DEMO_S1)
""")

c02 = nbf.v4.new_code_cell(r"""# ---- Blocking per country ----
from blocking import load_frame, block_country
from metrics import candidate_stats
CANDS, POOL = {}, {}
for c in C.COUNTRIES_TRAIN:
    s1 = load_frame(C.PROC / f"train_s1_{c}.parquet").head(DEMO_S1)
    s2 = load_frame(C.PROC / f"train_s2_{c}.parquet")
    s3 = load_frame(C.PROC / f"train_s3_{c}.parquet")
    cand = block_country(s1, s2, s3, topk=C.TFIDF_TOPK, ngrams=C.TFIDF_NGRAMS)
    cand.write_parquet(str(C.OUT / f"candidates_{c}.parquet"))
    CANDS[c] = cand
    POOL[c] = set(s2["entity_id"].to_list()) | set(s3["entity_id"].to_list())
    st = candidate_stats(cand)
    print(c, "s1=", len(s1), "cand_pairs=", len(cand), "per-s1:", st)
""")

c03 = nbf.v4.new_code_cell(r"""# ---- Labels + S1-group split ----
from labels import load_gt_map, label_pairs, restrict_gt_to_pool, s1_split
from metrics import blocking_recall
GTMAP, GTIN, COUNTS, SPLITS = {}, {}, {}, {}
for c in C.COUNTRIES_TRAIN:
    cand = CANDS[c]
    s1ids = set(cand["s1"].to_list())
    gtmap, counts = load_gt_map(C.GT, s1ids)
    gtin = restrict_gt_to_pool(gtmap, POOL[c])
    tr_ids, va_ids = s1_split(s1ids, seed=C.SEED, frac=0.2)
    GTMAP[c], GTIN[c], COUNTS[c], SPLITS[c] = gtmap, gtin, counts, (tr_ids, va_ids)
    lab = label_pairs(cand, gtmap)
    CANDS[c] = lab
    npos = int(lab["label"].sum())
    npos_in = sum(len(gtin.get(s, ())) for s in s1ids)
    print(c, "s1=", len(s1ids), "train/val=", len(tr_ids), len(va_ids),
          "pos=", npos, "neg=", len(lab) - npos,
          "recall_raw=", round(blocking_recall(cand, gtmap, s1ids), 4),
          "recall_in_sample=", round(blocking_recall(cand, gtin, s1ids), 4),
          "in_sample_true=", npos_in)
""")

c04 = nbf.v4.new_code_cell(r"""# ---- Features ----
from features import build_lookup, featurize
FEAT, FCOLS = {}, None
for c in C.COUNTRIES_TRAIN:
    s1 = load_frame(C.PROC / f"train_s1_{c}.parquet").head(DEMO_S1)
    s2 = load_frame(C.PROC / f"train_s2_{c}.parquet")
    s3 = load_frame(C.PROC / f"train_s3_{c}.parquet")
    import polars as _pl
    cd = _pl.concat([s2, s3], how="diagonal")
    s1map, sc = build_lookup(s1)
    cdmap, cc = build_lookup(cd)
    F, FCOLS = featurize(CANDS[c].drop("label"), s1map, cdmap, sc, cc)
    F = F.with_columns(CANDS[c]["label"].alias("label"))
    F.write_parquet(str(C.OUT / f"features_{c}.parquet"))
    FEAT[c] = F
    print(c, "pairs=", len(F), "pos_rate=", round(float(F["label"].mean()), 4), "ncols=", len(FCOLS))
print("feature cols:", FCOLS)
""")

c05 = nbf.v4.new_code_cell(r"""# ---- LightGBM ----
from model import train, predict
import numpy as np
MODELS, VAL = {}, {}
for c in C.COUNTRIES_TRAIN:
    F = FEAT[c]
    tr_ids, va_ids = SPLITS[c]
    s1l = CANDS[c]["s1"].to_list()
    tr_idx = [i for i, s in enumerate(s1l) if s in tr_ids]
    va_idx = [i for i, s in enumerate(s1l) if s in va_ids]
    X = F.select(FCOLS).to_numpy()
    y = F["label"].to_numpy()
    m = train(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], C.LGBM_PARAMS,
              C.LGBM_ROUNDS, C.LGBM_EARLY)
    m.save_model(str(C.OUT / f"lgbm_{c}.txt"))
    MODELS[c] = m
    pv = predict(m, X[va_idx])
    VAL[c] = (va_idx, pv)
    from metrics import pair_prf
    p, r, f = pair_prf(y[va_idx], (pv >= 0.5).astype(int))
    print(c, "best_iter=", m.best_iteration, "val@0.5 P=%.3f R=%.3f F0.5=%.3f" % (p, r, f))
    imp = sorted(zip(FCOLS, m.feature_importance(importance_type="gain")), key=lambda t: -t[1])[:8]
    print("  top feats:", [(k, int(v)) for k, v in imp])
""")

c06 = nbf.v4.new_code_cell(r"""# ---- Vetoes + threshold tuning + entity decisions + submission-format demo ----
from decide import apply_vetoes, tune_thresholds, decide_all, write_matching_tsv
from metrics import entity_macro_f05, check_submission_format
import json as _json
SUM = {}
GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for c in C.COUNTRIES_TRAIN:
    F = FEAT[c]
    va_idx, pv = VAL[c]
    s1l = [CANDS[c]["s1"][i] for i in va_idx]
    midl = [CANDS[c]["mid"][i] for i in va_idx]
    rows = [{"s1": s, "mid": m, "label": int(F["label"][i]),
             "proba": float(p), "missing_addr": bool(F["missing_addr"][i])}
            for i, s, m, p in zip(va_idx, s1l, midl, pv)]
    # vetoes applied on proba aligned to val rows:
    vproba = apply_vetoes(F[va_idx].with_columns(pl.Series("proba", list(pv))))
    vrows = [{**r, "proba": q} for r, q in zip(rows, vproba)]
    tr_ids, va_ids = SPLITS[c]
    gtmap = {s: GTIN[c].get(s, set()) for s in va_ids}  # closed-universe: in-sample truth
    ta, tm, te, bf = tune_thresholds(vrows, gtmap, sorted(va_ids), GRID, GRID)
    match = decide_all(vrows, ta, tm, te)
    f05, sing_acc, nsing = entity_macro_f05(match, gtmap, sorted(va_ids))
    tsv = str(C.OUT / f"matching_results_val_{c}.tsv")
    write_matching_tsv(match, sorted(va_ids), tsv)
    issues = check_submission_format(match, sorted(va_ids))
    SUM[c] = {"t_avail": ta, "t_missing": tm, "t_empty": te, "val_macro_f05": round(f05, 4),
              "recall_scope": "in-sample closed universe (demo pools; FULL_RUN measures raw recall)",
              "singleton_acc": round(sing_acc, 4), "n_singletons": nsing,
              "format_issues": issues, "tsv": tsv}
    print(c, "tuned", SUM[c])
_json.dump(SUM, open(C.OUT / "metrics.json", "w"), indent=1)
print("saved metrics.json + validation TSVs (submission format)")
""")

c07 = nbf.v4.new_code_cell(r"""# ---- Summary ----
print("Approach-2 demo complete. Artifacts in outputs/:")
print("candidates_*.parquet, features_*.parquet, lgbm_*.txt, matching_results_val_*.tsv, metrics.json")
print("Next: FULL_RUN=true over full processed parquet; test inference = same path on test files;")
print("then auxiliaries (ZeroER scorer, dense retrieval + reranker) join only on measured deltas.")
""")

nb.cells = [c00, c01, c02, c03, c04, c05, c06, c07]
nbf.write(nb, "02_approach2.ipynb")
print("wrote 02_approach2.ipynb")
