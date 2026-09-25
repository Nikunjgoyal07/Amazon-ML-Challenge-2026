"""Regression check: block_country_sharded == block_country on demo scale. Run: python3 verify_sharded.py"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src"))
import config as C
import polars as pl
from blocking import block_country, block_country_sharded, load_frame

s1 = load_frame(C.PROC / "train_s1_US.parquet").head(1500)
s2 = load_frame(C.PROC / "train_s2_US.parquet").head(20000)
s3 = load_frame(C.PROC / "train_s3_US.parquet").head(20000)
for df, n in [(s1, "_v_s1"), (s2, "_v_s2"), (s3, "_v_s3")]:
    df.write_parquet(str(C.OUT / f"{n}.parquet"))

t1 = time.time()
ref = block_country(s1, s2, s3, topk=20, ngrams=C.TFIDF_NGRAMS,
                    fit_names=s1["core_name"].to_list())
t2 = time.time()
n = block_country_sharded(str(C.OUT / "_v_s1.parquet"),
                          [str(C.OUT / "_v_s2.parquet"), str(C.OUT / "_v_s3.parquet")],
                          str(C.OUT / "_v_out.parquet"), topk=20, ngrams=C.TFIDF_NGRAMS,
                          s1_chunk=500, pool_shard=20000, fit_sample=20000,
                          fit_names=s1["core_name"].to_list())
t3 = time.time()
got = pl.read_parquet(str(C.OUT / "_v_out.parquet"))
rs = {(a, b) for a, b in zip(ref["s1"].to_list(), ref["mid"].to_list())}
gs = {(a, b) for a, b in zip(got["s1"].to_list(), got["mid"].to_list())}
print(f"unsharded={len(rs)} sharded={len(gs)} equal={rs == gs} t_ref={t2-t1:.0f}s t_sharded={t3-t2:.0f}s")
assert rs == gs, "sharded path diverged!"
for f in ["_v_s1.parquet", "_v_s2.parquet", "_v_s3.parquet", "_v_out.parquet"]:
    (C.OUT / f).unlink()
print("SHARDED PATH EQUALITY PROVEN")
