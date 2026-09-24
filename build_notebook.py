"""Build 01_eda_preprocessing.ipynb via nbformat (guarantees valid JSON)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

c00 = nbf.v4.new_markdown_cell("""# 01 — EDA + Preprocessing (from scratch, Kaggle-ready)

Business Entity Resolution: `S1 -> S2/S3`, macro-F0.5. Run top-to-bottom locally or on Kaggle.

- **Local**: data auto-found at `../6ab10eb3b23ba_student_resource/student_resource/dataset` (or set `DATA_ROOT` env).
- **Kaggle**: upload the TSVs as a dataset, set `KAGGLE_MODE=true` (or `DATA_ROOT=/kaggle/input/<your-dataset>`).
- `FULL_RUN=true` processes every row in chunks and writes full parquet. Default (`false`) runs EDA fully (fast lazy scans) + validates preprocessing on a 100k sample so the notebook finishes in minutes anywhere.""")

c01 = nbf.v4.new_code_cell(r"""import os, re, json, unicodedata
from pathlib import Path

import polars as pl

REPO = Path.cwd()
_env_root = os.environ.get("DATA_ROOT", "").strip()
DATA_ROOT = Path(_env_root) if _env_root else None
if DATA_ROOT is None or not DATA_ROOT.exists():
    cands = [REPO / "../6ab10eb3b23ba_student_resource/student_resource/dataset",
             REPO.parent / "6ab10eb3b23ba_student_resource/student_resource/dataset",
             Path("/kaggle/input/ml-challenge-2026")]
    DATA_ROOT = next((p.resolve() for p in cands if p.exists()), cands[0].resolve())
assert Path(DATA_ROOT).exists(), f"DATA_ROOT not found: {DATA_ROOT}"
FULL_RUN = os.environ.get("FULL_RUN", "false").lower() == "true"
OUT = REPO / "processed"
OUT.mkdir(parents=True, exist_ok=True)
SEP, SAMPLE_N, CHUNK = "\t", 100000, 200000
P = {"s1tr": DATA_ROOT/"train/train_source1.tsv", "s2tr": DATA_ROOT/"train/train_source2.tsv",
     "s3tr": DATA_ROOT/"train/train_source3.tsv", "gt": DATA_ROOT/"train/train_ground_truth.tsv",
     "s1te": DATA_ROOT/"test/test_source1.tsv", "s2te": DATA_ROOT/"test/test_source2.tsv",
     "s3te": DATA_ROOT/"test/test_source3.tsv"}
print("REPO=", str(REPO))
print("DATA_ROOT=", str(DATA_ROOT), "exists=", DATA_ROOT.exists())
print("FULL_RUN=", FULL_RUN)
print("missing=", [k for k, v in P.items() if not Path(v).exists()] or "none")

def safe(x, n=120):
    s = repr(x)
    return s[:n]
""")

c02 = nbf.v4.new_code_cell(r"""# ---- EDA (lazy scans: fast even on 2M+ rows) ----
eda = {}
for k in ["s1tr", "s2tr", "s3tr", "s1te", "s2te", "s3te"]:
    n = pl.scan_csv(str(P[k]), separator="\t", infer_schema_length=10000).select(pl.len()).collect().item()
    eda[k + "_rows"] = n
    print(k, "rows=", n)
for k in ["s1tr", "s1te"]:
    d = pl.scan_csv(str(P[k]), separator="\t", infer_schema_length=10000).group_by("country").len().collect().sort("country")
    eda[k + "_country"] = {r["country"]: r["len"] for r in d.to_dicts()}
    print(k, eda[k + "_country"])
for k in ["s1tr", "s2tr", "s3tr"]:
    d = pl.scan_csv(str(P[k]), separator="\t", infer_schema_length=10000).select(
        [pl.col(c).is_null().sum().alias(c) for c in ["business_name", "business_address"]]).collect().to_dicts()[0]
    eda[k + "_nulls"] = d
    print(k, "nulls=", d)
gt = pl.read_csv(str(P["gt"]), separator="\t", infer_schema_length=10000).with_columns(
    pl.col("matched_entity_ids").fill_null("").str.strip_chars().alias("m"))
gt = gt.with_columns(
    pl.when(pl.col("m") == "").then(0).otherwise(pl.col("m").str.split(",").list.len()).alias("mc"))
mc = gt["mc"]
eda["gt_mean_mc"] = round(float(mc.mean()), 3)
eda["gt_max_mc"] = int(mc.max())
eda["gt_singleton_pct"] = round(float((mc == 0).mean() * 100), 2)
eda["gt_dist"] = {int(r["mc"]): int(r["len"]) for r in gt.group_by("mc").len().sort("mc").head(12).to_dicts()}
print("mean_mc=", eda["gt_mean_mc"], "max=", eda["gt_max_mc"], "singleton%=", eda["gt_singleton_pct"])
print("dist=", eda["gt_dist"])
s = pl.scan_csv(str(P["s1tr"]), separator="\t", infer_schema_length=10000).select(
    ["business_name", "business_address"]).head(50000).collect()
s = s.with_columns(pl.col("business_name").fill_null("").str.len_chars().alias("nl"),
                   pl.col("business_address").fill_null("").str.len_chars().alias("al"))
eda["name_len_median_50k"] = float(s["nl"].median())
eda["addr_len_median_50k"] = float(s["al"].median())
eda["empty_addr_in_50k"] = int((s["al"] == 0).sum())
print("name_median=", eda["name_len_median_50k"], "addr_median=", eda["addr_len_median_50k"],
      "empty_in_50k=", eda["empty_addr_in_50k"])
print("--- raw noise samples (train S1) ---")
for r in pl.scan_csv(str(P["s1tr"]), separator="\t", infer_schema_length=10000).head(5).collect().to_dicts():
    print(safe(r["business_name"]), "|", safe(r["business_address"]), "|", r["country"])
json.dump(eda, open(OUT / "eda_summary.json", "w"), indent=1)
print("saved eda_summary.json")
""")

c03 = nbf.v4.new_code_cell(r"""# ---- Normalization (MIT-safe: stdlib only, no unidecode/GPL) ----
JUNK_RE = re.compile(r"(\*+|>>|--|<<|##|\[services\]|\(id:\s*\d+\)|<null>)", re.I)
HON_RE = re.compile(r"^\s*(smt|shri|mr|mrs|miss|dr)\.?\s+", re.I)
SPACE_RE = re.compile(r"\s+")
DOMAIN_RE = re.compile(r"^([\w\-]+)\.(com|in|fr|co\.in|net|org)(\.\w+)?$", re.I)
HOUSE_RE = re.compile(r"\b(h\.?\s*no\.?|house|plot|door)\s*\.?\s*#?\s*(\d+[a-z]?)", re.I)
PO_RE = re.compile(r"\b(p\.?o\.?\s*box|unit\s*\d+|hn\b|door\s*no)\b[^,]*,?", re.I)
PIN5_RE = re.compile(r"(?<!\d)(\d{5})(?!\d)")
PIN6_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
LEGAL = {"pvt", "private", "ltd", "limited", "corp", "corporation", "inc", "llc", "llp",
         "co", "enterprises", "sarl", "sas", "sa", "eurl", "sci", "groupe", "s.a.s"}
ABBR = {"rd": "road", "st": "street", "cir": "circle", "ave": "avenue", "av": "avenue",
        "blvd": "boulevard", "bd": "boulevard", "r": "rue"}

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def base_clean(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = strip_accents(s).lower()
    s = JUNK_RE.sub(" ", s)
    return SPACE_RE.sub(" ", s).strip(" ,.-")

def normalize_name(raw):
    s = base_clean("" if raw is None else str(raw))
    s = HON_RE.sub("", s)
    m = DOMAIN_RE.match(s.replace(" ", ""))
    stem = m.group(1) if m else ""
    toks = [t.strip(".") for t in s.split(" ") if t]
    return {"name_norm": s,
            "core_name": " ".join(t for t in toks if t not in LEGAL),
            "legal_form": " ".join(t for t in toks if t in LEGAL),
            "domain_stem": stem}

def normalize_address(raw):
    s = base_clean("" if raw is None else str(raw))
    mh = HOUSE_RE.search(s)
    house = mh.group(2).lower() if mh else ""
    m6, m5 = PIN6_RE.search(s), PIN5_RE.search(s)
    pin = m6.group(1) if m6 else (m5.group(1) if m5 else "")
    toks = [ABBR.get(t.strip("."), t.strip(".")) for t in PO_RE.sub(" ", s).replace(",", " ").split()]
    norm = SPACE_RE.sub(" ", " ".join(toks)).strip()
    return {"addr_norm": norm, "house_no": house, "zip_pin_cp": pin,
            "empty_addr": int(norm == ""), "missing_pin": int(pin == "")}

assert normalize_name("GURU INFOTECH PVT. LTD.")["core_name"] == "guru infotech"
assert normalize_name("projectsmanipalindia.com")["domain_stem"] == "projectsmanipalindia"
assert normalize_address("H.no 780, Voltaire, 75011 Paris")["house_no"] == "780"
assert normalize_address("H.no 780, Voltaire, 75011 Paris")["zip_pin_cp"] == "75011"
assert normalize_address("")["empty_addr"] == 1
print("all normalize unit tests PASS")
""")

c04 = nbf.v4.new_code_cell(r"""# ---- Run preprocessing ----
def normalize_frame(df):
    names = [normalize_name(x) for x in df["business_name"].to_list()]
    addrs = [normalize_address(x) for x in df["business_address"].to_list()]
    return df.with_columns([
        pl.Series("name_norm", [n["name_norm"] for n in names]),
        pl.Series("core_name", [n["core_name"] for n in names]),
        pl.Series("legal_form", [n["legal_form"] for n in names]),
        pl.Series("domain_stem", [n["domain_stem"] for n in names]),
        pl.Series("addr_norm", [a["addr_norm"] for a in addrs]),
        pl.Series("house_no", [a["house_no"] for a in addrs]),
        pl.Series("zip_pin_cp", [a["zip_pin_cp"] for a in addrs]),
        pl.Series("empty_addr", [a["empty_addr"] for a in addrs]),
        pl.Series("missing_pin", [a["missing_pin"] for a in addrs])])

def process_full(src, dst, chunk=CHUNK):
    total = pl.scan_csv(str(src), separator="\t", infer_schema_length=10000).select(pl.len()).collect().item()
    parts = []
    for off in range(0, total, chunk):
        df = pl.scan_csv(str(src), separator="\t", infer_schema_length=10000).slice(off, chunk).collect()
        parts.append(normalize_frame(df))
        print(f"{Path(src).name} {off + len(parts[-1])}/{total}")
    out = pl.concat(parts, how="diagonal")
    out.write_parquet(str(dst))
    print(f"WROTE {dst.name} rows={len(out)}")
    return len(out)

counts = {}
if FULL_RUN:
    counts["s1tr"] = process_full(P["s1tr"], OUT / "train_s1_processed.parquet")
    counts["s1te"] = process_full(P["s1te"], OUT / "test_s1_processed.parquet")
else:
    for k, name in [("s1tr", "train_s1_sample_processed.parquet"),
                    ("s1te", "test_s1_sample_processed.parquet"),
                    ("s2tr", "train_s2_sample_processed.parquet"),
                    ("s3tr", "train_s3_sample_processed.parquet")]:
        df = normalize_frame(pl.scan_csv(str(P[k]), separator="\t", infer_schema_length=10000).head(SAMPLE_N).collect())
        df.write_parquet(str(OUT / name))
        counts[k] = len(df)
        print(f"WROTE {name} rows={len(df)} (sample; set FULL_RUN=true for full)")
json.dump(counts, open(OUT / "processed_counts.json", "w"), indent=1)
print("DONE:", counts)
""")

c05 = nbf.v4.new_code_cell(r"""# ---- Verify ----
sample = "train_s1_sample_processed.parquet" if not FULL_RUN else "train_s1_processed.parquet"
df = pl.read_parquet(str(OUT / sample)).select(
    ["business_name", "core_name", "legal_form", "business_address",
     "addr_norm", "house_no", "zip_pin_cp"]).head(5)
for r in df.to_dicts():
    print(safe(r["business_name"]), "->", safe(r["core_name"]), "| legal=", safe(r["legal_form"]))
    print("   ", safe(r["business_address"]), "->", safe(r["addr_norm"]),
          "| house=", r["house_no"], "pin=", r["zip_pin_cp"])
print("verify OK")
""")

nb.cells = [c00, c01, c02, c03, c04, c05]
nbf.write(nb, "01_eda_preprocessing.ipynb")
print("wrote 01_eda_preprocessing.ipynb")
