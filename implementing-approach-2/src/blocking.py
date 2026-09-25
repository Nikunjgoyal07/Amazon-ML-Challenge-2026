"""Per-country blocking: exact keys + char TF-IDF + address keys, union with provenance."""
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer


def load_frame(path):
    return pl.read_parquet(str(path))


def _key_map(df, key):
    """key value -> list of entity_ids (non-empty keys only)."""
    m = {}
    for eid, k in zip(df["entity_id"].to_list(), df[key].to_list()):
        if k:
            m.setdefault(k, []).append(eid)
    return m


def exact_block(s1, cd, key, tag, cap_per_s1=200):
    """Join S1 to candidates on exact normalized key. Returns (s1, mid, tag)."""
    cmap = _key_map(cd, key)
    rows, counts = [], {}
    for eid, k in zip(s1["entity_id"].to_list(), s1[key].to_list()):
        if k and k in cmap:
            for mid in cmap[k]:
                if counts.get(eid, 0) >= cap_per_s1:
                    break
                rows.append((eid, mid))
                counts[eid] = counts.get(eid, 0) + 1
    if not rows:
        return pl.DataFrame({"s1": [], "mid": [], tag: []},
                            schema={"s1": pl.String, "mid": pl.String, tag: pl.Int8})
    return pl.DataFrame({"s1": [r[0] for r in rows], "mid": [r[1] for r in rows],
                         tag: [1] * len(rows)})


def tfidf_block(s1_ids, s1_names, cd_ids, cd_names, topk, ngrams=(2, 5), chunk=2000,
              fit_names=None, tag="tfidf_score"):
    """Char n-gram TF-IDF cosine top-K per S1. fit_names overrides the fit corpus
    (planner: fit on train names, transform everything)."""
    import numpy as np
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngrams)
    vec.fit(fit_names if fit_names is not None else cd_names)
    C = vec.transform(cd_names)
    out_s1, out_mid, out_sc = [], [], []
    for off in range(0, len(s1_ids), chunk):
        S = vec.transform(s1_names[off:off + chunk])
        sim = (S @ C.T).tocsr()
        for i in range(sim.shape[0]):
            row = sim.getrow(i)
            if row.nnz == 0:
                continue
            idx = row.indices
            dat = row.data
            if len(idx) > topk:
                sel = np.argpartition(dat, -topk)[-topk:]
                idx, dat = idx[sel], dat[sel]
            for j, s in zip(idx, dat):
                out_s1.append(s1_ids[off + i])
                out_mid.append(cd_ids[j])
                out_sc.append(round(float(s), 4))
    return pl.DataFrame({"s1": out_s1, "mid": out_mid, tag: out_sc})


def union_blocks(dfs):
    """Outer union on (s1, mid); max provenance flags and tfidf_score."""
    if not dfs:
        return pl.DataFrame(schema={"s1": pl.String, "mid": pl.String})
    tags = [c for df in dfs for c in df.columns if c not in ("s1", "mid")]
    scores = {c for c in tags if c.endswith("_score")}
    all_cols = ["s1", "mid"] + sorted(set(tags))
    normed = []
    for df in dfs:
        for c in all_cols:
            if c not in df.columns:
                df = df.with_columns(pl.lit(0).alias(c) if c not in scores
                                     else pl.lit(0.0).alias(c))
        df = df.with_columns(
            [(pl.col(c).cast(pl.Float64) if c in scores
              else pl.col(c).cast(pl.Int8)) for c in all_cols if c not in ("s1", "mid")])
        normed.append(df.select(all_cols))
    return pl.concat(normed, how="diagonal").group_by(["s1", "mid"]).max()


def block_country(s1, s2, s3, topk=15, ngrams=(2, 5), fit_names=None, return_parts=False,
                  auto_shard_at=500000, pool_shard=1000000, s1_chunk=200000,
                  fit_sample=500000):
    """Full multi-pass blocking for one country (s2+s3 pooled). Passes: exact name_norm,
    exact core_name, exact phonetic_code, address keys, TF-IDF on core_name AND name_norm.
    Pools larger than auto_shard_at spill transparently through block_country_sharded
    (exact for global top-K); then return_parts comes back None."""
    import os
    import shutil
    import tempfile
    cd = pl.concat([s2, s3], how="diagonal")
    if len(cd) > auto_shard_at:
        tmpdir = tempfile.mkdtemp(prefix="blk_")
        try:
            s1p = os.path.join(tmpdir, "s1.parquet")
            s2p = os.path.join(tmpdir, "s2.parquet")
            s3p = os.path.join(tmpdir, "s3.parquet")
            outp = os.path.join(tmpdir, "union.parquet")
            s1.write_parquet(s1p)
            s2.write_parquet(s2p)
            s3.write_parquet(s3p)
            block_country_sharded(s1p, [s2p, s3p], outp, topk=topk, ngrams=ngrams,
                                  # bounded transients for 5M+ pools: small S1 chunks,
                                  # 500k pool shards, fit corpus capped at pool/12
                                  s1_chunk=min(s1_chunk, 5000),
                                  pool_shard=min(pool_shard, 500000),
                                  fit_sample=min(fit_sample,
                                               max(50000, (len(s2) + len(s3)) // 12)),
                                  fit_names=fit_names)
            union = pl.read_parquet(outp)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        if return_parts:
            print("sharded full-pool run: per-pass frames unavailable, ablation skipped")
            return union, None
        return union
    fit = fit_names if fit_names is not None else None
    parts = {
        "exact": exact_block(s1, cd, "name_norm", "from_exact"),
        "core": exact_block(s1, cd, "core_name", "from_core"),
        "phon": exact_block(s1, cd, "phonetic_code", "from_phon"),
        "addr": address_block(s1, cd),
        "tfidf_core": tfidf_block(
            s1["entity_id"].to_list(), s1["core_name"].to_list(),
            cd["entity_id"].to_list(), cd["core_name"].to_list(),
            topk, ngrams, fit_names=fit,
            tag="tfidf_score").with_columns(pl.lit(1).alias("from_tfidf")),
        "tfidf_name": tfidf_block(
            s1["entity_id"].to_list(), s1["name_norm"].to_list(),
            cd["entity_id"].to_list(), cd["name_norm"].to_list(),
            topk, ngrams, fit_names=fit,
            tag="tfidf_name_score").with_columns(pl.lit(1).alias("from_tfidf2")),
    }
    union = union_blocks(list(parts.values()))
    # merge the two tfidf score columns into one max score
    if "tfidf_name_score" in union.columns:
        union = union.with_columns(
            pl.max_horizontal("tfidf_score", "tfidf_name_score").alias("tfidf_score")
        ).drop("tfidf_name_score")
    if return_parts:
        return union, parts
    return union


def block_country_sharded(s1_path, cd_paths, out_path, topk=15, ngrams=(2, 5),
                          s1_chunk=200000, pool_shard=1000000, fit_sample=500000, seed=42,
                          fit_names=None):
    """FULL_RUN path: streams S1 (chunks) x pool (shards); exact for global top-K.
    s1_path: S1 parquet; cd_paths: [s2 parquet, s3 parquet]. Writes union parquet.
    Returns n_pairs. Memory peaks: one pool shard + one S1 chunk + shard matrices."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from pathlib import Path as _P
    out_path, tmp = _P(out_path), _P(str(out_path) + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    s1_total = pl.scan_parquet(str(s1_path)).select(pl.len()).collect().item()
    pool_ns = [pl.scan_parquet(str(p)).select(pl.len()).collect().item() for p in cd_paths]
    # --- fit vectorizer: explicit corpus wins (equality with block_country), else strided pool sample ---
    if fit_names is None:
        stride = max(1, sum(pool_ns) // fit_sample)
        fit_names = []
        for p in cd_paths:
            lf = pl.scan_parquet(str(p)).select("core_name")
            fit_names += lf.gather_every(stride).collect()["core_name"].to_list()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngrams)
    vec.fit([x or "" for x in fit_names])
    # --- S1 key map (narrow) for exact/address passes ---
    s1keys = pl.scan_parquet(str(s1_path)).select(
        ["entity_id", "name_norm", "core_name", "phonetic_code",
         "zip_pin_cp", "city"]).collect()
    # --- stream pool shards ---
    shard_files, off = [], [0] * len(cd_paths)
    keyed_pairs = []  # exact/address pairs accumulated per shard (small)
    totals = [0] * len(cd_paths)
    # materialize pool shards lazily per pass to bound memory
    n_shards = max(1, max((n + pool_shard - 1) // pool_shard for n in pool_ns))
    for si in range(n_shards):
        shard = []
        for pi, p in enumerate(cd_paths):
            n = pool_ns[pi]
            a = si * pool_shard
            if a >= n:
                continue
            part = pl.scan_parquet(str(p)).slice(a, pool_shard).collect()
            shard.append(part)
        if not shard:
            continue
        pool = pl.concat(shard, how="diagonal")
        kp = _key_pairs(s1keys, pool)  # exact/address/phonetic vs full S1 map
        if len(kp):
            keyed_pairs.append(kp)
        C = None
        cids = pool["entity_id"].to_list()
        pool_core = [(x or "") for x in pool["core_name"].to_list()]
        pool_name = [(x or "") for x in pool["name_norm"].to_list()]
        # S1 chunks x this pool shard -> top-K, both TF-IDF fields (fld 0=core, 1=name)
        for off in range(0, s1_total, s1_chunk):
            s1c = pl.scan_parquet(str(s1_path)).slice(off, s1_chunk).collect()
            ids = s1c["entity_id"].to_list()
            o1, om, os, of = [], [], [], []
            for fld, qnames, dnames in [
                    (0, [(x or "") for x in s1c["core_name"].to_list()], pool_core),
                    (1, [(x or "") for x in s1c["name_norm"].to_list()], pool_name)]:
                S = vec.transform(qnames)
                C = vec.transform(dnames)
                sim = (S @ C.T).tocsr()
                for i in range(sim.shape[0]):
                    row = sim.getrow(i)
                    if row.nnz == 0:
                        continue
                    idx, dat = row.indices, row.data
                    if len(idx) > topk:
                        sel = np.argpartition(dat, -topk)[-topk:]
                        idx, dat = idx[sel], dat[sel]
                    for j, s in zip(idx, dat):
                        o1.append(ids[i])
                        om.append(cids[j])
                        os.append(round(float(s), 4))
                        of.append(fld)
            if o1:
                pl.DataFrame({"s1": o1, "mid": om, "tfidf_score": os, "fld": of,
                              "shard": si, "off": off}).write_parquet(
                    str(tmp / f"tf_{si}_{off}.parquet"))
                shard_files.append((si, off))
        del pool
    # --- merge TF-IDF chunk files: global top-K per S1 (exact merge) ---
    import collections
    per_sf = collections.defaultdict(list)
    for si, off in shard_files:
        df = pl.read_parquet(str(tmp / f"tf_{si}_{off}.parquet"))
        for s1, mid, sc, fld in zip(df["s1"].to_list(), df["mid"].to_list(),
                                    df["tfidf_score"].to_list(), df["fld"].to_list()):
            per_sf[(s1, fld)].append((mid, sc))
    best = {}  # (s1, mid) -> [core_score, name_score]
    for (s1, fld), lst in per_sf.items():
        lst.sort(key=lambda t: -t[1])
        seen = set()
        for mid, sc in lst:
            if mid in seen:
                continue
            seen.add(mid)
            if len(seen) > topk:
                break
            best.setdefault((s1, mid), [0.0, 0.0])[fld] = sc
    o1, om, os, of1, of2 = [], [], [], [], []
    for (s1, mid), (c0, c1) in best.items():
        o1.append(s1)
        om.append(mid)
        os.append(max(c0, c1))
        of1.append(int(c0 > 0))
        of2.append(int(c1 > 0))
    tf = pl.DataFrame({"s1": o1, "mid": om, "tfidf_score": os,
                       "from_tfidf": of1, "from_tfidf2": of2})
    parts = [tf] + keyed_pairs
    union = union_blocks(parts)
    union.write_parquet(str(out_path))
    for f in tmp.glob("*.parquet"):
        f.unlink()
    tmp.rmdir()
    return len(union)


def _key_pairs(s1keys, pool):
    """Exact/address/phonetic joins of a pool shard vs full S1 key map."""
    out = []
    sids = s1keys["entity_id"].to_list()
    for key, tag in [("name_norm", "from_exact"), ("core_name", "from_core"),
                      ("phonetic_code", "from_phon"), ("zip_pin_cp", "from_addr"),
                      ("city", "from_addr")]:
        cmap = {}
        for eid, k in zip(pool["entity_id"].to_list(), pool[key].to_list()):
            if k:
                cmap.setdefault(k, []).append(eid)
        rows = []
        for eid, k in zip(sids, s1keys[key].to_list()):
            if k and k in cmap:
                for mid in cmap[k][:200]:
                    rows.append((eid, mid))
        if rows:
            out.append(pl.DataFrame({"s1": [r[0] for r in rows],
                                     "mid": [r[1] for r in rows],
                                     tag: [1] * len(rows)}))
    return union_blocks(out) if out else pl.DataFrame(schema={"s1": pl.String,
    "mid": pl.String})


def address_block(s1, cd):
    """Candidates sharing non-empty zip_pin_cp or city (exact_block skips empties)."""
    zb = exact_block(s1, cd, "zip_pin_cp", "from_addr")
    cb = exact_block(s1, cd, "city", "from_addr")
    return union_blocks([zb, cb])
