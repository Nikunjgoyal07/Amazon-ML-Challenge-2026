"""Ground-truth loading, pair labeling, S1-group splits."""
import polars as pl


def load_gt_map(gt_path, keep_ids=None):
    """Returns (gtmap: s1 -> set(mid), counts: s1 -> match_count)."""
    gt = pl.read_csv(str(gt_path), separator="\t", infer_schema_length=10000,
                     columns=["source1_entity_id", "matched_entity_ids"])
    if keep_ids is not None:
        gt = gt.filter(pl.col("source1_entity_id").is_in(list(keep_ids)))
    gt = gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.strip_chars().alias("m"))
    gtmap, counts = {}, {}
    for s1, m in zip(gt["source1_entity_id"].to_list(), gt["m"].to_list()):
        s = {x.strip() for x in m.split(",") if x.strip()} if m else set()
        gtmap[s1] = s
        counts[s1] = len(s)
    return gtmap, counts


def label_pairs(pairs, gtmap):
    return pairs.with_columns(
        pl.struct(["s1", "mid"]).map_elements(
            lambda r: 1 if r["mid"] in gtmap.get(r["s1"], set()) else 0,
            return_dtype=pl.Int8).alias("label"))


def restrict_gt_to_pool(gtmap, pool_ids):
    """Keep only GT matches present in the candidate pool (closed-universe eval on samples)."""
    return {s1: (mids & pool_ids) for s1, mids in gtmap.items()}


def s1_split(s1_ids, seed=42, frac=0.2):
    import random
    ids = sorted(set(s1_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * frac)
    return set(ids[n_val:]), set(ids[:n_val])
