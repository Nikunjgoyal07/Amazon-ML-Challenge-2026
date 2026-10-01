"""Batched test inference with append-mode TSV outputs. Constant memory."""
import sys, time, pickle
sys.path.insert(0, "src")
import polars as pl
import lightgbm as lgb
from common import add_keys, block_batch, featurize, FEATURE_COLS, load_stops, ensure_phon

CAP = 30
BATCH = 50_000

t0 = time.time()
stops = load_stops("test")
tok_stops = pickle.load(open("/tmp/opencode/tokstops_test.pkl", "rb"))
phon_stops = pickle.load(open("/tmp/opencode/phonstops_test.pkl", "rb"))
idx_all = ensure_phon(pl.read_parquet("/tmp/opencode/keys_s23_test.parquet"))
tok_all = []
for c in ["US", "India", "France"]:
    try:
        tok_all.append(pl.read_parquet(f"/tmp/opencode/tok_test_{c}.parquet").with_columns(pl.lit(c).alias("country")))
    except Exception as e:
        print("tok shard missing", c, e, flush=True)
tok_all = pl.concat(tok_all)
# partition once by country -> smaller hash tables per join
idx_by_c = {c: idx_all.filter(pl.col("country") == c) for c in ["US", "India", "France"]}
tok_by_c = {c: tok_all.filter(pl.col("country") == c) for c in ["US", "India", "France"]}
del idx_all, tok_all
import gc; gc.collect()
booster = lgb.Booster(model_file="/tmp/opencode/lgbm.txt")
thrs = pickle.load(open("/tmp/opencode/thresholds.pkl", "rb"))
print(f"loaded {time.time()-t0:.0f}s idx={[len(v) for v in idx_by_c.values()]} tok={[len(v) for v in tok_by_c.values()]} thrs={thrs}", flush=True)

s1_ids = pl.scan_csv("dataset/test/test_source1.tsv", separator="\t").select(["entity_id"]).collect()["entity_id"].to_list()
N = len(s1_ids)
print(f"test S1: {N}", flush=True)

m_path, c_path = "output/matching_results.tsv", "output/candidate_pairs.tsv"
first = True
for start in range(0, N, BATCH):
    tc = time.time()
    chunk_ids = set(s1_ids[start:start + BATCH])
    batch = pl.scan_csv("dataset/test/test_source1.tsv", separator="\t").select(
        ["entity_id", "business_name", "business_address", "country"]).filter(
        pl.col("entity_id").is_in(list(chunk_ids))).collect()
    batch = add_keys(batch)
    # per-country blocking against country shards (smaller hashes, less peak RAM)
    cands_list = []
    for c in batch["country"].unique().to_list():
        bc = batch.filter(pl.col("country") == c)
        ref = idx_by_c.get(c)
        tok = tok_by_c.get(c)
        if ref is None:  # open-set country never seen: fall back to all shards
            ref = pl.concat(list(idx_by_c.values()))
            tok = pl.concat(list(tok_by_c.values()))
        cc = block_batch(bc, ref, cap=CAP, stops=stops,
                         tok_idx=tok, tok_stops=tok_stops, phon_stops=phon_stops)
        if len(cc):
            cands_list.append(cc)
    import gc; gc.collect()
    cands = pl.concat(cands_list) if cands_list else pl.DataFrame(
        schema={"entity_id": pl.String, "cand_id": pl.String})
    n_pairs = len(cands)
    # candidate output rows (all S1 in batch, empty if none)
    if len(cands):
        cg = cands.group_by("entity_id").agg(pl.col("cand_id").alias("c")) \
            .with_columns(pl.col("c").list.join(",").alias("candidate_entity_ids")).select(["entity_id", "candidate_entity_ids"])
    else:
        cg = pl.DataFrame(schema={"entity_id": pl.String, "candidate_entity_ids": pl.String})
    # predict (per-country string lookup -> bounded joins)
    match_map = {}
    if len(cands):
        cands = cands.join(batch.select(["entity_id", "country"]), on="entity_id", how="left")
        scols = ["entity_id", "name_clean", "addr_clean", "pin", "hnum", "first_tok"]
        bstr = batch.select(["entity_id", "country"] + ["name_clean", "addr_clean", "pin", "hnum", "first_tok"])
        F_list = []
        for c in cands["country"].unique().to_list():
            cc = cands.filter(pl.col("country") == c)
            ref = idx_by_c.get(c, pl.concat(list(idx_by_c.values())))
            pairs = (cc.join(bstr.filter(pl.col("country") == c), on=["entity_id", "country"], how="left")
                     .join(ref.select(scols).rename(
                         {x: x + "_right" for x in ["name_clean", "addr_clean", "pin", "hnum", "first_tok"]} | {"entity_id": "cand_id"}),
                         on="cand_id", how="left"))
            F_list.append(featurize(pairs))
            del pairs
        del cands; gc.collect()
        F = pl.concat(F_list); del F_list; gc.collect()
        probs = booster.predict(F.select(FEATURE_COLS).to_numpy())
        F = F.with_columns(pl.Series("p", probs))
        countries = dict(zip(batch["entity_id"].to_list(), batch["country"].to_list()))
        for s, c, p in zip(F["entity_id"].to_list(), F["cand_id"].to_list(), probs):
            t = thrs.get(countries.get(s, ""), 0.5)
            if p >= t:
                match_map.setdefault(s, []).append((c, float(p)))
        del F; gc.collect()
    # build rows for every S1 in batch (required: one row each)
    mrows, crows = [], []
    cand_map = {r["entity_id"]: r["candidate_entity_ids"] for r in cg.to_dicts()}
    for s in batch["entity_id"].to_list():
        ms = sorted(match_map.get(s, []), key=lambda x: -x[1])
        mrows.append((s, ",".join(c for c, _ in ms)))
        crows.append((s, cand_map.get(s, "")))
    mdf = pl.DataFrame({"source1_entity_id": [r[0] for r in mrows], "matched_entity_ids": [r[1] for r in mrows]})
    cdf = pl.DataFrame({"source1_entity_id": [r[0] for r in crows], "candidate_entity_ids": [r[1] for r in crows]})
    import io
    if first:
        mdf.write_csv(m_path, separator="\t", quote_style="never", include_header=True)
        cdf.write_csv(c_path, separator="\t", quote_style="never", include_header=True)
        first = False
    else:
        buf = io.StringIO()
        mdf.write_csv(buf, separator="\t", quote_style="never", include_header=False)
        with open(m_path, "a", encoding="utf-8") as f:
            f.write(buf.getvalue())
        buf2 = io.StringIO()
        cdf.write_csv(buf2, separator="\t", quote_style="never", include_header=False)
        with open(c_path, "a", encoding="utf-8") as f:
            f.write(buf2.getvalue())
    print(f"batch {start//BATCH+1}/{(N+BATCH-1)//BATCH}: S1={len(batch)} cands={n_pairs} matched_S1={len(match_map)} {time.time()-tc:.0f}s total={time.time()-t0:.0f}s", flush=True)

print(f"DONE total {time.time()-t0:.0f}s", flush=True)
