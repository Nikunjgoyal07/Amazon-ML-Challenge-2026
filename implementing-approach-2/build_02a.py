"""Generate 02a_approach2_core.ipynb: blocking(+ablation) -> features -> train -> decide."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

c00 = nbf.v4.new_markdown_cell("""# 02a — Approach 2 Core (extended)

Upgraded blocking (exact + phonetic + address + dual TF-IDF, train-fit scope) with per-pass
recall diagnostics → features → LightGBM → vetoes + tuned thresholds → submission-format TSV.
Demo: 3k train S1/country. Companion `02b` holds ablations, test inference, one-owner, FULL_RUN.""")

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
print("ROOT=", C.ROOT, "PROC=", C.PROC.exists(), "GT=", C.GT.exists())
FULL = os.environ.get("FULL_RUN", "false").lower() == "true"
DEMO_S1 = 3000 if not FULL else 10**18
print("FULL_RUN=", FULL, "DEMO_S1=", DEMO_S1)
""")

c02 = nbf.v4.new_code_cell(r"""# ---- Blocking per country (return parts for ablation) ----
from blocking import load_frame, block_country
from metrics import candidate_stats
CANDS, PARTS, POOL = {}, {}, {}
for c in C.COUNTRIES_TRAIN:
    s1 = load_frame(C.PROC / f"train_s1_{c}.parquet").head(DEMO_S1)
    s2 = load_frame(C.PROC / f"train_s2_{c}.parquet")
    s3 = load_frame(C.PROC / f"train_s3_{c}.parquet")
    fit = s1["core_name"].to_list()  # planner: fit TF-IDF on train names
    union, parts = block_country(s1, s2, s3, topk=20, ngrams=C.TFIDF_NGRAMS,
                                 fit_names=fit, return_parts=True)
    union.write_parquet(str(C.OUT / f"candidates_{c}.parquet"))
    CANDS[c], PARTS[c] = union, parts
    POOL[c] = set(s2["entity_id"].to_list()) | set(s3["entity_id"].to_list())
    print(c, "s1=", len(s1), "pairs=", len(union), candidate_stats(union))
    for k, df in parts.items():
        print("   pass", k, "pairs=", len(df))
""")

c03 = nbf.v4.new_code_cell(r"""# ---- Labels + splits + recall (raw vs in-sample) ----
from labels import load_gt_map, label_pairs, restrict_gt_to_pool, s1_split
from metrics import blocking_recall
GTMAP, GTIN, SPLITS = {}, {}, {}
for c in C.COUNTRIES_TRAIN:
    cand = CANDS[c]
    s1ids = set(cand["s1"].to_list())
    gtmap, _ = load_gt_map(C.GT, s1ids)
    gtin = restrict_gt_to_pool(gtmap, POOL[c])
    tr_ids, va_ids = s1_split(s1ids, seed=C.SEED, frac=0.2)
    GTMAP[c], GTIN[c], SPLITS[c] = gtmap, gtin, (tr_ids, va_ids)
    CANDS[c] = label_pairs(cand, gtmap)
    print(c, "pos=", int(CANDS[c]["label"].sum()),
          "recall_raw=", round(blocking_recall(cand, gtmap, s1ids), 4),
          "recall_in_sample=", round(blocking_recall(cand, gtin, s1ids), 4))
""")

c04 = nbf.v4.new_code_cell(r"""# ---- Per-pass recall ablation (which pass earns its keep?) ----
from metrics import blocking_recall
for c in C.COUNTRIES_TRAIN:
    s1ids = set(CANDS[c]["s1"].to_list())
    print("==", c)
    for k, df in PARTS[c].items():
        r = blocking_recall(df.select(["s1", "mid"]), GTIN[c], s1ids)
        print(f"  {k:10s} pairs={len(df):7d} in-sample-recall={r:.4f}")
""")

c05 = nbf.v4.new_code_cell(r"""# ---- Features ----
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
    F = F.with_columns([CANDS[c]["s1"].alias("s1"), CANDS[c]["mid"].alias("mid")])
    F.write_parquet(str(C.OUT / f"features_{c}.parquet"))
    FEAT[c] = F
    print(c, "pairs=", len(F), "pos_rate=", round(float(F["label"].mean()), 4))
print("ncols=", len(FCOLS))
""")

c06 = nbf.v4.new_code_cell(r"""# ---- LightGBM baselines + val predictions ----
from model import train, predict
from metrics import pair_prf
import numpy as np
MODELS, VALP = {}, {}
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
    p, r, f = pair_prf(y[va_idx], (pv >= 0.5).astype(int))
    print(c, "iter=", m.best_iteration, "val@0.5 P=%.3f R=%.3f F0.5=%.3f" % (p, r, f))
    imp = sorted(zip(FCOLS, m.feature_importance(importance_type="gain")),
                 key=lambda t: -t[1])[:10]
    print("  top:", [(k, int(v)) for k, v in imp])
    vp = pl.DataFrame({"s1": [s1l[i] for i in va_idx],
                       "mid": [CANDS[c]["mid"][i] for i in va_idx],
                       "label": [int(y[i]) for i in va_idx],
                       "proba": [float(x) for x in pv],
                       "missing_addr": [bool(F["missing_addr"][i]) for i in va_idx],
                       "name_set": [float(F["name_set"][i]) for i in va_idx],
                       "house_tri": [int(F["house_tri"][i]) for i in va_idx],
                       "pin_tri": [int(F["pin_tri"][i]) for i in va_idx]})
    vp.write_parquet(str(C.OUT / f"valpred_{c}.parquet"))
    VALP[c] = vp
print("saved models + valpred frames")
""")

c07 = nbf.v4.new_code_cell(r"""# ---- Decisions: vetoes + tuned 2-bin thresholds + TSV + honest metrics ----
from decide import apply_vetoes, tune_thresholds, decide_all, write_matching_tsv
from metrics import entity_macro_f05, check_submission_format
import json as _json
SUM, GRID = {}, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for c in C.COUNTRIES_TRAIN:
    vp = VALP[c].with_columns(pl.Series("proba", apply_vetoes(VALP[c])))
    rows = vp.to_dicts()
    _, va_ids = SPLITS[c]
    gtmap = {s: GTIN[c].get(s, set()) for s in va_ids}
    ta, tm, te, bf = tune_thresholds(rows, gtmap, sorted(va_ids), GRID, GRID)
    match = decide_all(rows, ta, tm, te)
    f05, sacc, ns = entity_macro_f05(match, gtmap, sorted(va_ids))
    tsv = str(C.OUT / f"matching_results_val_{c}.tsv")
    write_matching_tsv(match, sorted(va_ids), tsv)
    SUM[c] = {"t_avail": ta, "t_missing": tm, "t_empty": te,
              "val_macro_f05_in_sample": round(f05, 4),
              "singleton_acc": round(sacc, 4), "n_singletons": ns,
              "format_issues": check_submission_format(match, sorted(va_ids)), "tsv": tsv}
    print(c, SUM[c])
_json.dump(SUM, open(C.OUT / "metrics_02a.json", "w"), indent=1)
_json.dump({c: {"va_ids": sorted(SPLITS[c][1]),
                "gtin": {s: sorted(GTIN[c].get(s, ())) for s in SPLITS[c][1]}}
             for c in C.COUNTRIES_TRAIN}, open(C.OUT / "splits_gtin.json", "w"))
print("note: entity scores are in-sample (closed universe); raw-recall ceiling measured at FULL_RUN")
""")

c08 = nbf.v4.new_code_cell(r"""# ---- Summary ----
print("02a complete: blocking(+per-pass recall), features, LightGBM, tuned decisions, TSVs.")
print("02b continues: veto/bin/NEG/group ablations, calibration, test+France inference,")
print("one-owner resolution, FULL_RUN sharded path + scale estimate.")
""")

nb.cells = [c00, c01, c02, c03, c04, c05, c06, c07, c08]
nbf.write(nb, "02a_approach2_core.ipynb")
print("wrote 02a_approach2_core.ipynb")
