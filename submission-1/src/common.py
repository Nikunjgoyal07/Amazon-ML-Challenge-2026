"""Shared normalize + blocking + featurize helpers (Polars, CPU-efficient)."""
import polars as pl

CLEAN_EXPR_NAME = (
    pl.col("business_name").str.to_lowercase()
    .str.replace_all(r"[^\p{L}\p{M}\p{N} ]", " ")
    .str.replace_all(r"\s+", " ")
    .str.strip_chars()
)
CLEAN_EXPR_ADDR = (
    pl.col("business_address").str.to_lowercase()
    .str.replace_all(r"[^\p{L}\p{M}\p{N} ]", " ")
    .str.replace_all(r"\s+", " ")
    .str.strip_chars()
)

def add_keys(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        CLEAN_EXPR_NAME.alias("name_clean"),
        CLEAN_EXPR_ADDR.alias("addr_clean"),
    ]).with_columns([
        pl.col("name_clean").str.extract(r"(\b\d{5,6}\b)", group_index=1).alias("pin"),
        # first number anywhere (covers 'KH NO 570', '5 bis Rue', '1795 Westchester')
        pl.col("addr_clean").str.extract(r"(\d{1,6})", group_index=1).alias("hnum"),
        pl.col("name_clean").str.split(" ").list.first().alias("first_tok"),
        pl.col("name_clean").str.replace_all(" ", "").str.slice(0, 4).alias("nprefix"),
        pl.col("name_clean").str.replace_all(r"[aeiou]", "").str.replace_all(" ", "").str.slice(0, 8).alias("phon"),
    ]).with_columns([
        # null out junk keys so joins don't explode on empty strings
        pl.when(pl.col("first_tok").str.len_chars() >= 3).then(pl.col("first_tok")).otherwise(None).alias("first_tok"),
        pl.when(pl.col("nprefix").str.len_chars() == 4).then(pl.col("nprefix")).otherwise(None).alias("nprefix"),
        pl.when(pl.col("hnum").is_not_null() & (pl.col("hnum").str.len_chars() <= 6)).then(pl.col("hnum")).otherwise(None).alias("hnum"),
    ])

import pickle
_STOPS = {}
def load_stops(split: str):
    global _STOPS
    if split not in _STOPS:
        with open(f"/tmp/opencode/stops_{split}.pkl", "rb") as f:
            _STOPS[split] = pickle.load(f)
    return _STOPS[split]

def _mask_stops(df: pl.DataFrame, stops: dict) -> pl.DataFrame:
    for k in ["hnum", "first_tok", "nprefix"]:
        s = stops.get(k, set())
        if not s:
            continue
        # vectorized mask via anti-join against small stop frame
        stop_df = pl.DataFrame({"country": [c for c, v in s], k: [v for c, v in s]}).with_columns(pl.lit(True).alias("__stop"))
        df = df.join(stop_df, on=["country", k], how="left").with_columns(
            pl.when(pl.col("__stop").fill_null(False)).then(None).otherwise(pl.col(k)).alias(k)
        ).drop("__stop")
    return df

def ensure_phon(df: pl.DataFrame) -> pl.DataFrame:
    if "phon" in df.columns:
        return df
    return df.with_columns(
        pl.col("name_clean").str.replace_all(r"[aeiou]", "").str.replace_all(" ", "").str.slice(0, 8).alias("phon"))

def block_batch(s1: pl.DataFrame, s23_idx: pl.DataFrame, cap: int = 25, stops: dict | None = None,
                tok_idx: pl.DataFrame | None = None, tok_stops: set | None = None,
                phon_stops: set | None = None) -> pl.DataFrame:
    if stops:
        s1 = _mask_stops(s1, stops)
    if phon_stops:
        sdf = pl.DataFrame({"country": [c for c, _ in phon_stops], "phon": [t for _, t in phon_stops]})
        s1 = s1.join(sdf, on=["country", "phon"], how="anti")
    """Hash-join candidates per country on lean keys. Returns s1_id, cand_id pairs."""
    frames = []
    s23 = s23_idx
    # K1 pin (high precision)
    j = s1.join(s23, left_on=["country", "pin"], right_on=["country", "pin"], how="inner", nulls_equal=False)
    if len(j):
        j = j.with_columns(pl.lit(0).alias("_prio")).select(["entity_id", "entity_id_right", "_prio"])
        frames.append(j)
    # K2 hnum
    j = s1.join(s23, left_on=["country", "hnum"], right_on=["country", "hnum"], how="inner", nulls_equal=False)
    if len(j):
        j = j.with_columns(pl.lit(1).alias("_prio")).select(["entity_id", "entity_id_right", "_prio"])
        frames.append(j)
    # K3 first token
    j = s1.join(s23, left_on=["country", "first_tok"], right_on=["country", "first_tok"], how="inner", nulls_equal=False)
    if len(j):
        j = j.with_columns(pl.lit(2).alias("_prio")).select(["entity_id", "entity_id_right", "_prio"])
        frames.append(j)
    # K4 nprefix
    j = s1.join(s23, left_on=["country", "nprefix"], right_on=["country", "nprefix"], how="inner", nulls_equal=False)
    if len(j):
        j = j.with_columns(pl.lit(3).alias("_prio")).select(["entity_id", "entity_id_right", "_prio"])
        frames.append(j)
    # K6 consonant-skeleton prefix (typo/transliteration tolerant)
    j = s1.join(s23_idx, left_on=["country", "phon"], right_on=["country", "phon"], how="inner", nulls_equal=False)
    if len(j):
        j = j.with_columns(pl.lit(3).alias("_prio")).select(["entity_id", "entity_id_right", "_prio"])
        frames.append(j)
    # K5 any significant name token (catches reorder/affix/DBA variants)
    if tok_idx is not None:
        s1t = s1.select(["entity_id", "country", "name_clean"]).with_columns(
            pl.col("name_clean").str.split(" ").alias("tok")).explode("tok").filter(
            pl.col("tok").str.len_chars() >= 4)
        if tok_stops:
            sdf = pl.DataFrame({"country": [c for c, _ in tok_stops], "tok": [t for _, t in tok_stops]})
            s1t = s1t.join(sdf, on=["country", "tok"], how="anti")
        j = s1t.join(tok_idx, left_on=["country", "tok"], right_on=["country", "tok"],
                     how="inner", nulls_equal=False)
        if len(j):
            j = j.with_columns(pl.lit(2).alias("_prio")).select(["entity_id", "entity_id_right", "_prio"])
            frames.append(j)
    if not frames:
        return pl.DataFrame({"entity_id": [], "entity_id_right": []}, schema={"entity_id": pl.String, "entity_id_right": pl.String})
    allc = pl.concat(frames)
    # rank by multi-key agreement: true matches usually share 2+ keys
    allc = (
        allc.group_by(["entity_id", "entity_id_right"], maintain_order=False)
        .agg([pl.len().alias("nkeys"), pl.col("_prio").min().alias("best_prio")])
        .sort(["entity_id", "nkeys", "best_prio"], descending=[False, True, False])
        .group_by("entity_id", maintain_order=True).head(cap)
    )
    return allc.rename({"entity_id_right": "cand_id"}).drop("best_prio", "nkeys")


def featurize(pairs: pl.DataFrame) -> pl.DataFrame:
    """pairs must have name_clean, addr_clean, pin, hnum, first_tok (+_right suffix for cand).
    Returns numeric feature frame. RapidFuzz 1-to-1 via zip loop in chunks."""
    from rapidfuzz import fuzz
    a_name = pairs["name_clean"].to_list()
    b_name = pairs["name_clean_right"].to_list()
    a_addr = pairs["addr_clean"].to_list()
    b_addr = pairs["addr_clean_right"].to_list()
    n = len(pairs)
    f_nr = [0.0] * n
    f_nts = [0.0] * n
    f_ar = [0.0] * n
    for i in range(n):
        an, bn, aa, ba = a_name[i], b_name[i], a_addr[i], b_addr[i]
        f_nr[i] = fuzz.ratio(an, bn) / 100.0
        f_nts[i] = fuzz.token_set_ratio(an, bn) / 100.0
        f_ar[i] = fuzz.ratio(aa, ba) / 100.0
    out = pairs.with_columns([
        pl.Series("f_name_ratio", f_nr, dtype=pl.Float32),
        pl.Series("f_name_tset", f_nts, dtype=pl.Float32),
        pl.Series("f_addr_ratio", f_ar, dtype=pl.Float32),
        (pl.col("pin") == pl.col("pin_right")).fill_null(False).cast(pl.Int8).alias("f_pin_match"),
        (pl.col("hnum") == pl.col("hnum_right")).fill_null(False).cast(pl.Int8).alias("f_hnum_match"),
        (pl.col("first_tok") == pl.col("first_tok_right")).fill_null(False).cast(pl.Int8).alias("f_first_match"),
        (pl.col("name_clean").str.len_chars() - pl.col("name_clean_right").str.len_chars()).abs().alias("f_name_lendiff"),
        (pl.col("addr_clean").str.len_chars() - pl.col("addr_clean_right").str.len_chars()).abs().alias("f_addr_lendiff"),
    ]).select(["entity_id", "cand_id"] + FEATURE_COLS)
    return out

FEATURE_COLS = ["f_name_ratio", "f_name_tset", "f_addr_ratio", "f_pin_match",
                "f_hnum_match", "f_first_match", "f_name_lendiff", "f_addr_lendiff"]
