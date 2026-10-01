"""Build S2/S3 key index -> parquet (once). Streams via Polars, low RAM."""
import sys, polars as pl
sys.path.insert(0, "src")
from common import add_keys

def build(inputs: list[str], out: str):
    frames = []
    for f in inputs:
        df = pl.scan_csv(f, separator="\t").select(["entity_id", "business_name", "business_address", "country"]).collect()
        df = add_keys(df)
        frames.append(df.select(["entity_id", "country", "pin", "hnum", "first_tok", "nprefix",
                                 "name_clean", "addr_clean"]))
        print(f"indexed {f}: {len(df)}", flush=True)
    idx = pl.concat(frames)
    idx.write_parquet(out)
    print(f"wrote {out}: {len(idx)} rows", flush=True)

if __name__ == "__main__":
    mode = sys.argv[1]  # train | test
    if mode == "train":
        build(["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"], "/tmp/opencode/keys_s23_train.parquet")
    else:
        build(["dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"], "/tmp/opencode/keys_s23_test.parquet")
