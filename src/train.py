"""Train LightGBM matcher on 150k S1 sample. Batched blocking+featurize, low RAM."""
import sys, time, pickle, random
sys.path.insert(0, "src")
import polars as pl
import lightgbm as lgb
from common import add_keys, block_batch, featurize, FEATURE_COLS, load_stops, ensure_phon

CAP = 30
SAMPLE = 150_000
CHUNK = 25_000
SEED = 42

t0 = time.time()
stops = load_stops("train")
tok_stops = pickle.load(open("/tmp/opencode/tokstops_train.pkl", "rb"))
phon_stops = pickle.load(open("/tmp/opencode/phonstops_train.pkl", "rb"))
idx = ensure_phon(pl.read_parquet("/tmp/opencode/keys_s23_train.parquet"))
tok_idx = pl.concat([
    pl.read_parquet("/tmp/opencode/tok_train_India.parquet").with_columns(pl.lit("India").alias("country")),
    pl.read_parquet("/tmp/opencode/tok_train_US.parquet").with_columns(pl.lit("US").alias("country")),
])
print(f"index loaded {time.time()-t0:.0f}s idx={len(idx)} tok={len(tok_idx)}", flush=True)

# stratified sample of S1 ids (by country)
ids = pl.scan_csv("dataset/train/train_source1.tsv", separator="\t").select(["entity_id", "country"]).collect()
random.seed(SEED)
keep = []
for c in ids["country"].unique().to_list():
    sub = ids.filter(pl.col("country") == c)["entity_id"].to_list()
    n = int(SAMPLE * len(sub) / len(ids))
    keep.extend(random.sample(sub, n))
keep = set(keep)
print(f"sampled {len(keep)} S1", flush=True)

gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
gt = gt.filter(pl.col("source1_entity_id").is_in(list(keep)))
truth = {}
for r in gt.to_dicts():
    g = [] if r["matched_entity_ids"] is None else [x for x in str(r["matched_entity_ids"]).split(",") if x]
    truth[r["source1_entity_id"]] = set(g)

s1_all = pl.scan_csv("dataset/train/train_source1.tsv", separator="\t").select(
    ["entity_id", "business_name", "business_address", "country"]).filter(
    pl.col("entity_id").is_in(list(keep))).collect()
s1_all = add_keys(s1_all)
print(f"s1 sample collected: {len(s1_all)} {time.time()-t0:.0f}s", flush=True)

feat_frames, label_list = [], []
ids_list = list(keep)
for i in range(0, len(s1_all), CHUNK):
    tc = time.time()
    batch = s1_all.slice(i, CHUNK)
    cands = block_batch(batch, idx, cap=CAP, stops=stops, tok_idx=tok_idx, tok_stops=tok_stops, phon_stops=phon_stops)
    if not len(cands):
        continue
    # attach strings for featurize
    pairs = (cands.join(batch.select(["entity_id", "name_clean", "addr_clean", "pin", "hnum", "first_tok"]),
                        on="entity_id", how="left")
             .join(idx.select(["entity_id", "name_clean", "addr_clean", "pin", "hnum", "first_tok"]).rename(
                 {c: c + "_right" for c in ["name_clean", "addr_clean", "pin", "hnum", "first_tok"]} | {"entity_id": "cand_id"}),
                 on="cand_id", how="left"))
    F = featurize(pairs)
    y = [1 if c in truth.get(s, set()) else 0 for s, c in zip(F["entity_id"].to_list(), F["cand_id"].to_list())]
    feat_frames.append(F.with_columns(pl.Series("label", y, dtype=pl.Int8)))
    print(f"chunk {i//CHUNK}: cands={len(cands)} pos={sum(y)} {time.time()-tc:.0f}s", flush=True)

data = pl.concat(feat_frames)
print(f"train pairs: {len(data)} pos={data['label'].sum()} {time.time()-t0:.0f}s", flush=True)
data.write_parquet("/tmp/opencode/train_pairs.parquet")

# downsample negatives to 2:1 for speed
pos = data.filter(pl.col("label") == 1)
neg = data.filter(pl.col("label") == 0).sample(min(len(data.filter(pl.col('label')==0)), 2 * len(pos)), seed=SEED)
tr = pl.concat([pos, neg]).sample(len(pos) + len(neg), seed=SEED)
print(f"fit rows: {len(tr)}", flush=True)
X = tr.select(FEATURE_COLS).to_numpy()
y = tr["label"].to_numpy()
dtr = lgb.Dataset(X, label=y)
params = {"objective": "binary", "metric": "binary_logloss", "num_leaves": 63, "min_data_in_leaf": 200,
          "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1, "verbosity": -1, "num_threads": 4}
booster = lgb.train(params, dtr, num_boost_round=200)
booster.save_model("/tmp/opencode/lgbm.txt")

# per-country threshold tuning for F0.5 on this same sample (quick grid)
import numpy as np
tr2 = tr.with_columns(pl.Series("p", booster.predict(X)))
best = {}
for c in s1_all["country"].unique().to_list():
    s1c = set(s1_all.filter(pl.col("country") == c)["entity_id"].to_list())
    sub = tr2.filter(pl.col("entity_id").is_in(list(s1c)))
    # group probs/labels per S1
    best_t, best_f = 0.5, -1
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        # per-S1 F0.5 with threshold t (predict all above t)
        g = sub.group_by("entity_id").agg([pl.col("p"), pl.col("label")])
        fs = []
        for r in g.to_dicts():
            pp, ll = r["p"], r["label"]
            pred = [i for i, p in enumerate(pp) if p >= t]
            tp = sum(ll[i] for i in pred)
            prec = tp / max(len(pred), 1)
            rec = tp / max(sum(ll), 1)
            if not pred and sum(ll) == 0:
                fs.append(1.0)
            elif prec + rec == 0:
                fs.append(0.0)
            else:
                fs.append(1.25 * prec * rec / (0.25 * prec + rec))
        # include S1 with no candidates as singleton-correct if GT empty
        f = sum(fs) / len(fs) if fs else 0
        if f > best_f:
            best_f, best_t = f, t
    best[c] = best_t
    print(f"country {c}: thr={best_t} F05={best_f:.3f}", flush=True)
pickle.dump(best, open("/tmp/opencode/thresholds.pkl", "wb"))
print(f"DONE total {time.time()-t0:.0f}s", flush=True)
