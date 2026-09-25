"""Generate 02b_approach2_scale.ipynb: ablations, test+France inference, one-owner, FULL_RUN."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

c00 = nbf.v4.new_markdown_cell("""# 02b — Approach 2 Scale & Ablations

Decision ablations (veto, 1/2/4-bin) → NEG-ratio + group retrains → calibration →
test inference incl. **France** → one-owner resolution → FULL_RUN sharded dry-run + scale estimate.
Loads 02a artifacts; no recomputation of blocking/features.""")

c01 = nbf.v4.new_code_cell(r"""import os, sys, json, time
from pathlib import Path
HERE = Path.cwd()
SRC = HERE / "src" if (HERE / "src").exists() else HERE / "implementing-approach-2" / "src"
sys.path.insert(0, str(SRC))
import config as C
import polars as pl
import lightgbm as lgb
SPL = json.load(open(C.OUT / "splits_gtin.json"))
FEAT = {c: pl.read_parquet(str(C.OUT / f"features_{c}.parquet")) for c in C.COUNTRIES_TRAIN}
VALP = {c: pl.read_parquet(str(C.OUT / f"valpred_{c}.parquet")) for c in C.COUNTRIES_TRAIN}
MODELS = {c: lgb.Booster(model_file=str(C.OUT / f"lgbm_{c}.txt")) for c in C.COUNTRIES_TRAIN}
FCOLS = [c for c in FEAT["US"].columns if c not in ("s1", "mid", "label")]
print("loaded:", {c: len(FEAT[c]) for c in FEAT}, "ncols=", len(FCOLS))
""")

c02 = nbf.v4.new_code_cell(r"""# ---- Veto ablation: same tuned thresholds, vetoes on vs off ----
from decide import apply_vetoes, tune_thresholds, decide_all
from metrics import entity_macro_f05
GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for c in C.COUNTRIES_TRAIN:
    base = VALP[c].to_dicts()
    vetoed = VALP[c].with_columns(pl.Series("proba", apply_vetoes(VALP[c]))).to_dicts()
    va_ids = sorted(SPL[c]["va_ids"])
    gtmap = {s: set(SPL[c]["gtin"].get(s, [])) for s in va_ids}
    for name, rows in [("veto_on", vetoed), ("veto_off", base)]:
        ta, tm, te, bf = tune_thresholds(rows, gtmap, va_ids, GRID, GRID)
        m = decide_all(rows, ta, tm, te)
        f, _, _ = entity_macro_f05(m, gtmap, va_ids)
        print(f"{c} {name}: t=({ta},{tm},{te}) macroF05={f:.4f}  delta={(bf - f):+.4f} (tuned-val)")
""")

c03 = nbf.v4.new_code_cell(r"""# ---- Bin ablation: 1-bin global vs 2-bin (missing) vs 4-bin (missing x name_set>=0.8) ----
from decide import tune_binned, decide_all
from metrics import entity_macro_f05
BINS = {"1bin": lambda r: 0,
        "2bin": lambda r: 1 if r["missing_addr"] else 0,
        "4bin": lambda r: (1 if r["missing_addr"] else 0) * 2 + (1 if r["name_set"] >= 0.8 else 0)}
for c in C.COUNTRIES_TRAIN:
    rows = VALP[c].with_columns(pl.Series("proba", apply_vetoes(VALP[c]))).to_dicts()
    va_ids = sorted(SPL[c]["va_ids"])
    gtmap = {s: set(SPL[c]["gtin"].get(s, [])) for s in va_ids}
    for name, fn in BINS.items():
        bt, te, f = tune_binned(rows, fn, gtmap, va_ids, GRID, GRID)
        print(f"{c} {name}: bins={bt} t_empty={te} macroF05={f:.4f}")
""")

c04 = nbf.v4.new_code_cell(r"""# ---- NEG-ratio (US): all retrieved negatives vs 1:10 subsample ----
from model import train, predict
from metrics import pair_prf
import numpy as np
F = FEAT["US"]
y = F["label"].to_numpy()
s1l = F["s1"].to_list()
va = set(SPL["US"]["va_ids"])
tr_idx = [i for i, s in enumerate(s1l) if s not in va]
va_idx = [i for i, s in enumerate(s1l) if s in va]
X = F.select(FCOLS).to_numpy()
rng = np.random.default_rng(C.SEED)
pos_tr = [i for i in tr_idx if y[i] == 1]
neg_tr = [i for i in tr_idx if y[i] == 0]
keep_neg = set(rng.choice(neg_tr, size=min(len(neg_tr), 10 * len(pos_tr)), replace=False).tolist())
sub_idx = sorted(set(pos_tr) | keep_neg)
m10 = train(X[sub_idx], y[sub_idx], X[va_idx], y[va_idx], C.LGBM_PARAMS, C.LGBM_ROUNDS, C.LGBM_EARLY)
for name, m in [("all-neg", MODELS["US"]), ("1:10", m10)]:
    pv = predict(m, X[va_idx])
    p, r, f = pair_prf(y[va_idx], (pv >= 0.5).astype(int))
    print(f"US {name}: P={p:.3f} R={r:.3f} F0.5={f:.3f}")
m10.save_model(str(C.OUT / "lgbm_US_neg10.txt"))
""")

c05 = nbf.v4.new_code_cell(r"""# ---- Group ablation (US): lexical-only vs full ----
LEX = ["name_fuzz", "name_set", "name_jw", "name_jac", "name_len_diff", "core_exact",
       "legal_agree", "domain_exact", "phonetic_match", "tfidf_score"]
from model import train, predict
from metrics import pair_prf
F = FEAT["US"]
y = F["label"].to_numpy()
s1l = F["s1"].to_list()
va = set(SPL["US"]["va_ids"])
tr_idx = [i for i, s in enumerate(s1l) if s not in va]
va_idx = [i for i, s in enumerate(s1l) if s in va]
Xl = F.select(LEX).to_numpy()
Xv = F.select(LEX).to_numpy()[va_idx]
ml = train(Xl[tr_idx], y[tr_idx], Xv, y[va_idx], C.LGBM_PARAMS, C.LGBM_ROUNDS, C.LGBM_EARLY)
pv = predict(ml, Xv)
p, r, f = pair_prf(y[va_idx], (pv >= 0.5).astype(int))
print(f"US lex-only: P={p:.3f} R={r:.3f} F0.5={f:.3f}  (vs full-model cell in 02a)")
""")

c06 = nbf.v4.new_code_cell(r"""# ---- Calibration (US baseline): 10 bins, mean proba vs pos rate ----
from model import predict
import numpy as np
F = FEAT["US"]
va = set(SPL["US"]["va_ids"])
va_idx = [i for i, s in enumerate(F["s1"].to_list()) if s in va]
pv = predict(MODELS["US"], F.select(FCOLS).to_numpy()[va_idx])
y = F["label"].to_numpy()[va_idx]
for b in range(10):
    m = (pv >= b / 10) & (pv < (b + 1) / 10)
    if m.sum():
        print(f"[{b/10:.1f},{(b+1)/10:.1f}) n={int(m.sum()):6d} mean_p={pv[m].mean():.3f} pos_rate={y[m].mean():.3f}")
""")

c07 = nbf.v4.new_code_cell(r"""# ---- Test inference incl. France ----
from normalize import normalize_frame
from blocking import block_country
from features import build_lookup, featurize
from model import train, predict
from decide import apply_vetoes, decide_all, write_matching_tsv
from metrics import check_submission_format
TEST_C = ["US", "India", "France"]
# normalize small test S2/S3 pools on the fly (vendored notebook-01 normalizer)
for c in TEST_C:
    for src, stem in [("test_source2.tsv", "test_s2_sample"), ("test_source3.tsv", "test_s3_sample")]:
        dst = C.OUT / f"{stem}_{c}.parquet"
        if dst.exists():
            continue
        raw = pl.scan_csv(str(C.DATA / "test" / src), separator="\t",
                          infer_schema_length=10000).filter(
            pl.col("country") == c).head(6000).collect()
        normalize_frame(raw).write_parquet(str(dst))
        print("wrote", dst.name, len(raw))
# combined US+India model for France (no French labels exist)
import polars as _pl
FB = _pl.concat([FEAT["US"], FEAT["India"]], how="diagonal")
yB = FB["label"].to_numpy()
XB = FB.select(FCOLS).to_numpy()
sB = FB["s1"].to_list()
vaB = set(SPL["US"]["va_ids"]) | set(SPL["India"]["va_ids"])
trB = [i for i, s in enumerate(sB) if s not in vaB]
vaBidx = [i for i, s in enumerate(sB) if s in vaB]
mC = train(XB[trB], yB[trB], XB[vaBidx], yB[vaBidx], C.LGBM_PARAMS, C.LGBM_ROUNDS, C.LGBM_EARLY)
mC.save_model(str(C.OUT / "lgbm_combined.txt"))
print("combined model trained")
for c in TEST_C:
    s1t = pl.read_parquet(str(Path(C.PROC) / f"test_s1_{c}.parquet")).head(1500)
    s2t = pl.read_parquet(str(C.OUT / f"test_s2_sample_{c}.parquet"))
    s3t = pl.read_parquet(str(C.OUT / f"test_s3_sample_{c}.parquet"))
    cand = block_country(s1t, s2t, s3t, topk=20, ngrams=C.TFIDF_NGRAMS)
    cd = _pl.concat([s2t, s3t], how="diagonal")
    s1map, sc = build_lookup(s1t)
    cdmap, cc = build_lookup(cd)
    F, _ = featurize(cand, s1map, cdmap, sc, cc)
    model = MODELS[c] if c in MODELS else mC
    pv = predict(model, F.select(FCOLS).to_numpy())
    rows = [{"s1": s, "mid": m, "label": -1, "proba": float(p),
             "missing_addr": bool(F["missing_addr"][i])}
            for i, (s, m, p) in enumerate(zip(cand["s1"].to_list(),
                                              cand["mid"].to_list(), pv))]
    rows = [{**r, "proba": q} for r, q in
            zip(rows, apply_vetoes(F.with_columns(pl.Series("proba", list(pv)))))]
    match = decide_all(rows, 0.5, 0.5, 0.5)
    tsv = str(C.OUT / f"matching_results_test_sample_{c}.tsv")
    write_matching_tsv(match, s1t["entity_id"].to_list(), tsv)
    issues = check_submission_format(match, s1t["entity_id"].to_list())
    nmat = sum(1 for v in match.values() if v)
    print(f"{c}: s1={len(s1t)} cand={len(cand)} matched_s1={nmat} issues={issues or 'none'} -> {tsv}")
""")

c08 = nbf.v4.new_code_cell(r"""# ---- One-owner resolution demo (US val: conflicts across S1s sharing the pool) ----
from decide import resolve_one_owner
from metrics import entity_macro_f05
rows = VALP["US"].with_columns(
    pl.Series("proba", apply_vetoes(VALP["US"]))).to_dicts()
va_ids = sorted(SPL["US"]["va_ids"])
gtmap = {s: set(SPL["US"]["gtin"].get(s, [])) for s in va_ids}
own = resolve_one_owner(rows, t_pair=0.5)
indep = {}
for r in rows:
    if r["proba"] >= 0.5:
        indep.setdefault(r["s1"], []).append(r["mid"])
for s in va_ids:
    indep.setdefault(s, [])
    own.setdefault(s, [])
nconf = sum(1 for r in rows if r["proba"] >= 0.5)
print("above-threshold pairs:", nconf, "| unique mids kept:", sum(len(v) for v in own.values()))
for name, m in [("independent@0.5", indep), ("one-owner@0.5", own)]:
    f, sa, ns = entity_macro_f05({s: m.get(s, []) for s in va_ids}, gtmap, va_ids)
    print(f"{name}: macroF05={f:.4f} singleton_acc={sa:.4f}")
""")

c09 = nbf.v4.new_code_cell(r"""# ---- FULL_RUN sharded path: dry-run equality + scale estimate ----
import time
from blocking import block_country_sharded, load_frame
t0 = time.time()
s1 = load_frame(C.PROC / "train_s1_US.parquet").head(1500)
s2 = load_frame(C.PROC / "train_s2_US.parquet").head(20000)
s3 = load_frame(C.PROC / "train_s3_US.parquet").head(20000)
s1.write_parquet(str(C.OUT / "_dry_s1.parquet"))
s2.write_parquet(str(C.OUT / "_dry_s2.parquet"))
s3.write_parquet(str(C.OUT / "_dry_s3.parquet"))
from blocking import block_country as _bc
t1 = time.time()
ref = _bc(s1, s2, s3, topk=20, ngrams=C.TFIDF_NGRAMS,
          fit_names=s1["core_name"].to_list())
t2 = time.time()
n = block_country_sharded(str(C.OUT / "_dry_s1.parquet"),
                          [str(C.OUT / "_dry_s2.parquet"), str(C.OUT / "_dry_s3.parquet")],
                          str(C.OUT / "_dry_out.parquet"), topk=20, ngrams=C.TFIDF_NGRAMS,
                          s1_chunk=500, pool_shard=20000, fit_sample=20000,
                          fit_names=s1["core_name"].to_list())
t3 = time.time()
got = pl.read_parquet(str(C.OUT / "_dry_out.parquet"))
rs = {(a, b) for a, b in zip(ref["s1"].to_list(), ref["mid"].to_list())}
gs = {(a, b) for a, b in zip(got["s1"].to_list(), got["mid"].to_list())}
print(f"unsharded={len(rs)} sharded={len(gs)} equal={rs == gs} t_ref={t2-t1:.0f}s t_sharded={t3-t2:.0f}s")
assert rs == gs, "sharded path diverged from unsharded reference"
print("dry-run equality PROVEN; FULL_RUN: per-country block_country_sharded, features chunked, train, decide")
for f in ["_dry_s1.parquet", "_dry_s2.parquet", "_dry_s3.parquet", "_dry_out.parquet"]:
    (C.OUT / f).unlink()
# scale estimate (linear in S1 x pool-shard work; TF-IDF fit sublinear via fit_sample cap)
import json as _j
est = {"measured": {"s1": 1500, "pool": 40000}, "full": {"s1_US": 1323633, "pool_US": 6186873},
       "note": ("TF-IDF fit capped at fit_sample; per-(S1chunk x poolshard) work scales ~linearly; "
                "run sharded FULL_RUN on Kaggle (30GB RAM) with s1_chunk=200k, pool_shard=1M.")}
_j.dump(est, open(C.OUT / "scale_estimate.json", "w"), indent=1)
""")

c10 = nbf.v4.new_code_cell(r"""# ---- Summary ----
print("02b complete: veto/bin/NEG/group ablations, calibration, test+France inference,")
print("one-owner demo, FULL_RUN dry-run equality. Promote winners into the FULL_RUN config,")
print("then auxiliaries (ZeroER scorer, dense retrieval) join only on measured deltas.")
""")

nb.cells = [c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c10]
nbf.write(nb, "02b_approach2_scale.ipynb")
print("wrote 02b_approach2_scale.ipynb")
