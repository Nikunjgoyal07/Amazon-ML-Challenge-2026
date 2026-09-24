"""Build 01_eda_preprocessing.ipynb via nbformat (guarantees valid JSON). Full coverage of preprocessing.md."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

c00 = nbf.v4.new_markdown_cell("""# 01 — EDA + Preprocessing (full preprocessing.md coverage, Kaggle-ready)

Business Entity Resolution: `S1 -> S2/S3`, macro-F0.5. Run top-to-bottom locally or on Kaggle.

- **Local**: data auto-found at `../6ab10eb3b23ba_student_resource/student_resource/dataset` (or `DATA_ROOT` env).
- **Kaggle**: upload TSVs as a dataset, set `DATA_ROOT=/kaggle/input/<dataset>`.
- `FULL_RUN=true` processes every row chunked into **per-country parquet** (`preprocessing.md` §4 contract). Default (`false`) runs full EDA + 100k sample validation.
- Covers preprocessing.md §2 (EDA checklist), §3 (normalization), §3.3 (native-script), §5 (done definition).""")

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
print("DATA_ROOT=", str(DATA_ROOT))
print("FULL_RUN=", FULL_RUN)
print("missing=", [k for k, v in P.items() if not Path(v).exists()] or "none")

def safe(x, n=120):
    s = repr(x)
    return s[:n]
""")

c02 = nbf.v4.new_code_cell(r"""# ---- EDA basics: shapes, schema, prefixes, nulls, ground truth ----
eda = {}
FILES = ["s1tr", "s2tr", "s3tr", "s1te", "s2te", "s3te"]
for k in FILES:
    n = pl.scan_csv(str(P[k]), separator="\t", infer_schema_length=10000).select(pl.len()).collect().item()
    eda[k + "_rows"] = n
    print(k, "rows=", n)
sch = pl.scan_csv(str(P["s1tr"]), separator="\t", infer_schema_length=10000).collect_schema()
eda["columns"] = sch.names()
print("columns=", eda["columns"])
PX = {"s1tr": "S1-", "s2tr": "S2-", "s3tr": "S3-", "s1te": "S1-", "s2te": "S2-", "s3te": "S3-"}
eda["prefix_bad"] = {}
for k, px in PX.items():
    bad = pl.scan_csv(str(P[k]), separator="\t", infer_schema_length=10000).filter(
        ~pl.col("entity_id").str.starts_with(px)).select(pl.len()).collect().item()
    eda["prefix_bad"][k] = bad
print("prefix_mismatches=", eda["prefix_bad"])
for k in FILES:
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
json.dump(eda, open(OUT / "eda_summary.json", "w"), indent=1)
print("basics done")
""")

c02b = nbf.v4.new_code_cell(r"""# ---- EDA structural: S2/S3 share, reuse=1, distractors, cross-country, lengths, FR samples ----
# S2 vs S3 share without exploding (string counting over full GT)
agg = gt.select([pl.col("m").str.count_matches("S2-").sum().alias("s2pairs"),
                 pl.col("m").str.count_matches("S3-").sum().alias("s3pairs")]).to_dicts()[0]
eda["s2_pairs"], eda["s3_pairs"] = int(agg["s2pairs"]), int(agg["s3pairs"])
tot = eda["s2_pairs"] + eda["s3_pairs"]
eda["s2_share_pct"] = round(100 * eda["s2_pairs"] / tot, 1)
eda["s3_share_pct"] = round(100 * eda["s3_pairs"] / tot, 1)
print("pairs S2=", eda["s2_pairs"], "S3=", eda["s3_pairs"], "share=", eda["s2_share_pct"], "/", eda["s3_share_pct"])
# Explode once: reuse check + distractor math
pairs = gt.select(["source1_entity_id", "m"]).filter(pl.col("m") != "").select(
    [pl.col("source1_entity_id").alias("s1"),
     pl.col("m").str.split(",").alias("lst")]).explode("lst").rename({"lst": "mid"})
vc = pairs.group_by("mid").len()
eda["reuse_max"] = int(vc["len"].max())
eda["matched_unique"] = int(vc.height)
s2s3_total = eda["s2tr_rows"] + eda["s3tr_rows"]
eda["distractor_pct"] = round(100 * (s2s3_total - eda["matched_unique"]) / s2s3_total, 2)
print("reuse_max=", eda["reuse_max"], "matched_unique=", eda["matched_unique"],
      "distractor%=", eda["distractor_pct"])
# Cross-country check: join pair countries (S1 map 2.2M + S2/S3 maps, id+country only)
s1map = pl.scan_csv(str(P["s1tr"]), separator="\t", infer_schema_length=10000).select(["entity_id", "country"]).collect()
s2map = pl.scan_csv(str(P["s2tr"]), separator="\t", infer_schema_length=10000).select(["entity_id", "country"]).collect()
s3map = pl.scan_csv(str(P["s3tr"]), separator="\t", infer_schema_length=10000).select(["entity_id", "country"]).collect()
p2 = pairs.filter(pl.col("mid").str.starts_with("S2-")).join(s1map, left_on="s1", right_on="entity_id", how="left").rename({"country": "c1"})
p2 = p2.join(s2map, left_on="mid", right_on="entity_id", how="left").rename({"country": "c2"})
p3 = pairs.filter(pl.col("mid").str.starts_with("S3-")).join(s1map, left_on="s1", right_on="entity_id", how="left").rename({"country": "c1"})
p3 = p3.join(s3map, left_on="mid", right_on="entity_id", how="left").rename({"country": "c2"})
eda["cross_country_mismatches"] = int(p2.filter(pl.col("c1") != pl.col("c2")).height + p3.filter(pl.col("c1") != pl.col("c2")).height)
eda["cross_country_checked"] = int(p2.height + p3.height)
print("cross-country mismatches=", eda["cross_country_mismatches"], "checked=", eda["cross_country_checked"])
del p2, p3
# Lengths per source (20k samples: char + token medians)
for k in ["s1tr", "s2tr", "s3tr"]:
    s = pl.scan_csv(str(P[k]), separator="\t", infer_schema_length=10000).select(
        ["business_name", "business_address"]).head(20000).collect().with_columns(
        [pl.col("business_name").fill_null("").str.len_chars().alias("nl"),
         pl.col("business_name").fill_null("").str.split(" ").list.len().alias("nt"),
         pl.col("business_address").fill_null("").str.len_chars().alias("al")])
    eda[k + "_len"] = {"name_char_med": float(s["nl"].median()), "name_tok_med": float(s["nt"].median()),
                       "addr_char_med": float(s["al"].median())}
    print(k, eda[k + "_len"])
# Completeness per country (30k samples: null-addr rate + PIN-presence rate)
for k in ["s1tr", "s2tr", "s3tr"]:
    s = pl.scan_csv(str(P[k]), separator="\t", infer_schema_length=10000).select(
        ["business_address", "country"]).head(30000).collect().with_columns(
        [pl.col("business_address").is_null().alias("null_addr"),
         pl.col("business_address").fill_null("").str.contains(r"\d{5,}").alias("has_pin")])
    eda[k + "_completeness"] = {r["country"]: {"null_addr_pct": round(100 * float(r["null_addr"]), 1),
        "pin_present_pct": round(100 * float(r["has_pin"]), 1)}
        for r in s.group_by("country").agg([pl.col("null_addr").mean(), pl.col("has_pin").mean()]).to_dicts()}
    print(k, eda[k + "_completeness"])
# FR samples: 5 rows per test source
for k in ["s1te", "s2te", "s3te"]:
    print("--- FR sample", k, "---")
    for r in pl.scan_csv(str(P[k]), separator="\t", infer_schema_length=10000).filter(
            pl.col("country") == "France").head(5).collect().to_dicts():
        print("   ", safe(r["business_name"]), "|", safe(r["business_address"]))
# S2/S3 country splits (train + test)
for k in ["s2tr", "s3tr", "s2te", "s3te"]:
    d = pl.scan_csv(str(P[k]), separator="\t", infer_schema_length=10000).group_by("country").len().collect().sort("country")
    eda[k + "_country"] = {r["country"]: r["len"] for r in d.to_dicts()}
    print(k, eda[k + "_country"])
# Frequent tokens + legal-suffix counts (100k raw S1 names)
LEGAL_RAW = {"pvt", "private", "ltd", "limited", "corp", "corporation", "inc", "llc", "llp", "co",
             "enterprises", "sarl", "sas", "sa", "eurl", "sci", "groupe"}
tok = pl.scan_csv(str(P["s1tr"]), separator="\t", infer_schema_length=10000).select("business_name").head(
    100000).collect().with_columns(pl.col("business_name").fill_null("").str.to_lowercase().alias("n"))
toks = tok.select(pl.col("n").str.split(" ").alias("t")).explode("t").filter(
    (pl.col("t") != "") & (~pl.col("t").is_in(sorted(LEGAL_RAW))))
eda["top_tokens"] = [[r["t"], int(r["len"])] for r in toks.group_by("t").len().sort("len", descending=True).head(20).to_dicts()]
suf = tok.select(pl.col("n").str.split(" ").list.last().alias("last")).filter(pl.col("last").is_in(sorted(LEGAL_RAW)))
eda["top_legal"] = [[r["last"], int(r["len"])] for r in suf.group_by("last").len().sort("len", descending=True).head(10).to_dicts()]
print("top_tokens=", eda["top_tokens"][:8])
print("top_legal=", eda["top_legal"])
json.dump(eda, open(OUT / "eda_summary.json", "w"), indent=1)
print("structural EDA done")
""")

c02c = nbf.v4.new_code_cell(r"""# ---- True-pair similarity (20k matched pairs) + 15-pair noise catalog ----
from rapidfuzz import fuzz
pr = pairs.sample(20000, seed=42)
s1ids, mids = pr["s1"].to_list(), pr["mid"].to_list()
s1rec = {r["entity_id"]: r for r in pl.scan_csv(str(P["s1tr"]), separator="\t",
    infer_schema_length=10000).filter(pl.col("entity_id").is_in(s1ids)).collect().to_dicts()}
s2need = [m for m in mids if m.startswith("S2-")]
s3need = [m for m in mids if m.startswith("S3-")]
mx = {}
for kk, pp, need in [("s2", P["s2tr"], s2need), ("s3", P["s3tr"], s3need)]:
    if need:
        for r in pl.scan_csv(str(pp), separator="\t", infer_schema_length=10000).filter(
                pl.col("entity_id").is_in(need)).collect().to_dicts():
            mx[r["entity_id"]] = r
ts, fr, aj = [], [], []
shown = 0
print("--- noise catalog (S1 vs matched S2/S3, 15 pairs) ---")
for a, b in zip(s1ids, mids):
    r1, r2 = s1rec.get(a), mx.get(b)
    if not r1 or not r2:
        continue
    n1, n2 = (r1["business_name"] or "").lower(), (r2["business_name"] or "").lower()
    d1, d2 = (r1["business_address"] or "").lower(), (r2["business_address"] or "").lower()
    ts.append(fuzz.token_set_ratio(n1, n2))
    fr.append(fuzz.ratio(n1, n2))
    t1, t2 = set(d1.replace(",", " ").split()), set(d2.replace(",", " ").split())
    aj.append(len(t1 & t2) / max(1, len(t1 | t2)))
    if shown < 15:
        print(f"[{b}]")
        print("  S1:", safe(r1["business_name"]), "|", safe(r1["business_address"]))
        print("  MT:", safe(r2["business_name"]), "|", safe(r2["business_address"]))
        shown += 1
import statistics as _st
eda["sim_20k"] = {"token_set_mean": round(_st.mean(ts), 1),
    "token_set_gt90_pct": round(100 * sum(1 for x in ts if x >= 90) / len(ts), 1),
    "token_set_lt60_pct": round(100 * sum(1 for x in ts if x < 60) / len(ts), 1),
    "fuzz_mean": round(_st.mean(fr), 1), "addr_jaccard_mean": round(_st.mean(aj), 3), "n": len(ts)}
print("sim_20k=", eda["sim_20k"])
json.dump(eda, open(OUT / "eda_summary.json", "w"), indent=1)
print("similarity EDA done")
""")

c03 = nbf.v4.new_code_cell(r"""# ---- Normalization (full preprocessing.md §3; MIT-safe, no unidecode/GPL) ----
try:
    from anyascii import anyascii
    HAS_AA = True
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anyascii"])
    from anyascii import anyascii
    HAS_AA = True
print("anyascii available:", HAS_AA, "| e.g.", anyascii("Pr\u00edvate \u0926\u093f\u0932\u094d\u0932\u0940"))
JUNK_RE = re.compile(r"(\*+|>>|--|<<|##|\[services\]|\(id:\s*\d+\)|<null>)", re.I)
HON_RE = re.compile(r"^\s*(smt|shri|mr|mrs|miss|dr)\.?\s+", re.I)
SPACE_RE = re.compile(r"\s+")
DOMAIN_RE = re.compile(r"^([\w\-]+)\.(com|in|fr|co\.in|net|org)(\.\w+)?$", re.I)
HOUSE_RE = re.compile(r"(?:\b(?:h\.?\s*no\.?|house|plot|door)\s*\.?\s*#?\s*(\d+[a-z]?)|#\s*(\d+))", re.I)
PO_RE = re.compile(r"\b(p\.?o\.?\s*box|unit\s*\d+|hn\b|door\s*no)\b[^,]*,?", re.I)
PIN5_RE = re.compile(r"(?<!\d)(\d{5})(?!\d)")
PIN6_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
LAND_RE = re.compile(r"\b(near|opp|opposite|behind|atm|temple|mosque|church|bank|hospital|station|mall)\b", re.I)
LEGAL = {"pvt", "private", "ltd", "limited", "corp", "corporation", "inc", "llc", "llp",
         "co", "enterprises", "sarl", "sas", "sa", "eurl", "sci", "groupe", "s.a.s"}
ABBR = {"rd": "road", "st": "street", "cir": "circle", "ave": "avenue", "av": "avenue",
        "blvd": "boulevard", "bd": "boulevard", "r": "rue"}
US_CODES = {"AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
            "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
            "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
            "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"}
US_NAMES = {"alabama": "AL", "alaska": "AK", "arizona": "AZ", "california": "CA", "texas": "TX",
            "florida": "FL", "new york": "NY", "ohio": "OH", "oklahoma": "OK", "maryland": "MD",
            "iowa": "IA", "north carolina": "NC"}
IN_MAP = {"MH": "maharashtra", "RJ": "rajasthan", "TN": "tamil nadu", "KA": "karnataka",
          "DL": "delhi", "WB": "west bengal", "UP": "uttar pradesh", "GJ": "gujarat"}
IN_NAMES = {"maharashtra", "rajasthan", "tamil nadu", "karnataka", "delhi", "west bengal",
            "uttar pradesh", "gujarat", "kolkata", "howrah"}
NATIVE_STATE = {"\u0ba4\u0bae\u0bbf\u0bb4\u0bcd\u0ba8\u0bbe\u0b9f\u0bc1": "tamil nadu",
                "\u092e\u0939\u093e\u0930\u093e\u0937\u094d\u091f\u094d\u0930": "maharashtra",
                "\u0926\u093f\u0932\u094d\u0932\u0940": "delhi",
                "\u092a\u0936\u094d\u091a\u093f\u092e \u092c\u0902\u0917\u093e\u0932": "west bengal",
                "\u0915\u0930\u094d\u0928\u093e\u091f\u0915": "karnataka"}

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def soundex(s):
    s = re.sub(r"[^a-z]", "", s.lower())
    if not s:
        return ""
    mapping = {"b": "1", "f": "1", "p": "1", "v": "1", "c": "2", "g": "2", "j": "2",
               "k": "2", "q": "2", "s": "2", "x": "2", "z": "2", "d": "3", "t": "3",
               "l": "4", "m": "5", "n": "5", "r": "6"}
    out = s[0].upper()
    prev = mapping.get(s[0], "0")
    for ch in s[1:]:
        if ch in "hw":
            continue
        d = mapping.get(ch, "0")
        if d != "0" and d != prev:
            out += d
            if len(out) == 4:
                break
        prev = d
    return (out + "000")[:4]

def base_clean(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = strip_accents(s).lower()
    s = JUNK_RE.sub(" ", s)
    return SPACE_RE.sub(" ", s).strip(" ,.-")

def normalize_name(raw):
    s = base_clean("" if raw is None else str(raw))
    hon = int(bool(HON_RE.search(s)))
    s = HON_RE.sub("", s)
    m = DOMAIN_RE.match(s.replace(" ", ""))
    stem = m.group(1) if m else ""
    toks = [t.strip(".") for t in s.split(" ") if t]
    legal = [t for t in toks if t in LEGAL]
    core_toks = [t for t in toks if t not in LEGAL]
    if not core_toks:
        core_toks = toks  # name is only a legal word (e.g. "Private"); keep it as core
    core = " ".join(core_toks)
    return {"name_norm": s, "core_name": core,
            "legal_form": " ".join(t for t in toks if t in LEGAL), "domain_stem": stem,
            "name_lat": anyascii(core) if HAS_AA else core,
            "phonetic_code": soundex(core.split(" ")[0]) if core else "",
            "has_honorific": hon}

def extract_city_state(s):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    city, state = "", ""
    if parts:
        last = parts[-1]
        toks = last.split()
        cand1 = toks[-1].upper().strip(".") if toks else ""
        cand2 = " ".join(toks[-2:]) if len(toks) >= 2 else ""
        if cand1 in US_CODES:
            state = cand1
        elif cand2 in US_NAMES:
            state = US_NAMES[cand2]
        elif cand2 in IN_NAMES or cand2 in IN_MAP.values():
            state = cand2
        elif toks and toks[-1].upper() in IN_MAP:
            state = IN_MAP[toks[-1].upper()]
        else:
            for tok in toks:
                if tok in NATIVE_STATE:
                    state = NATIVE_STATE[tok]
                    break
        if state and len(parts) >= 2:
            city = re.sub(r"\d+", "", parts[-2]).strip()
    return city, state

def normalize_address(raw):
    s = base_clean("" if raw is None else str(raw))
    mh = HOUSE_RE.search(s)
    house = (mh.group(1) or mh.group(2) or "").lower() if mh else ""
    m6, m5 = PIN6_RE.search(s), PIN5_RE.search(s)
    pin = m6.group(1) if m6 else (m5.group(1) if m5 else "")
    land = int(bool(LAND_RE.search(s)))
    s2 = PO_RE.sub(" ", s)
    toks = [ABBR.get(t.strip("."), t.strip(".")) for t in s2.replace(",", " ").split()]
    norm = SPACE_RE.sub(" ", " ".join(toks)).strip()
    city, state = extract_city_state(s)
    return {"addr_norm": norm, "street_tokens": " ".join(sorted(set(norm.split()))),
            "house_no": house, "zip_pin_cp": pin, "city": city, "state_code": state,
            "empty_addr": int(norm == ""), "missing_pin": int(pin == ""),
            "missing_city": int(city == ""), "landmark_flag": land}

assert normalize_name("*** Pr\u00edvate *** [Services] (ID: 1072)")["core_name"] == "private"
assert normalize_name("GURU INFOTECH PVT. LTD.")["core_name"] == "guru infotech"
assert normalize_name("SARL Dupont")["legal_form"] == "sarl"
assert normalize_name("projectsmanipalindia.com")["domain_stem"] == "projectsmanipalindia"
assert normalize_name("Shri Ram Traders")["has_honorific"] == 1
assert normalize_address("H.no 780, R. Voltaire, 75011 Paris")["house_no"] == "780"
assert normalize_address("H.no 780, R. Voltaire, 75011 Paris")["zip_pin_cp"] == "75011"
assert normalize_address("175 Boulevard Bd Saint-Germain, Paris")["addr_norm"].find("boulevard") >= 0
assert normalize_address("#12, Lake Town, Kolkata")["house_no"] == "12"
assert normalize_address("Shop 5, Near SBI ATM, Pune")["landmark_flag"] == 1
assert normalize_address("")["empty_addr"] == 1
if HAS_AA:
    assert anyascii("\u0926\u093f\u0932\u094d\u0932\u0940").strip() != ""
print("all normalize unit tests PASS (anyascii=%s)" % HAS_AA)
""")

c04 = nbf.v4.new_code_cell(r"""# ---- Run: per-country parquet (preprocessing.md §4 contract) ----
NEW_COLS = ["name_norm", "core_name", "legal_form", "domain_stem", "name_lat",
            "phonetic_code", "has_honorific", "addr_norm", "street_tokens", "house_no",
            "zip_pin_cp", "city", "state_code", "empty_addr", "missing_pin",
            "missing_city", "landmark_flag"]

def normalize_frame(df):
    names = [normalize_name(x) for x in df["business_name"].to_list()]
    addrs = [normalize_address(x) for x in df["business_address"].to_list()]
    cols = [[n["name_norm"] for n in names], [n["core_name"] for n in names],
            [n["legal_form"] for n in names], [n["domain_stem"] for n in names],
            [n["name_lat"] for n in names], [n["phonetic_code"] for n in names],
            [n["has_honorific"] for n in names], [a["addr_norm"] for a in addrs],
            [a["street_tokens"] for a in addrs], [a["house_no"] for a in addrs],
            [a["zip_pin_cp"] for a in addrs], [a["city"] for a in addrs],
            [a["state_code"] for a in addrs], [a["empty_addr"] for a in addrs],
            [a["missing_pin"] for a in addrs], [a["missing_city"] for a in addrs],
            [a["landmark_flag"] for a in addrs]]
    return df.with_columns([pl.Series(c, v) for c, v in zip(NEW_COLS, cols)])

def countries_of(src):
    return [r["country"] for r in pl.scan_csv(str(src), separator="\t",
            infer_schema_length=10000).select("country").unique().collect().to_dicts()]

def process_sample_per_country(src, stem, quota_each=40000):
    out = {}
    for c in countries_of(src):
        df = normalize_frame(pl.scan_csv(str(src), separator="\t", infer_schema_length=10000).filter(
            pl.col("country") == c).head(quota_each).collect())
        dst = OUT / f"{stem}_{c}.parquet"
        df.write_parquet(str(dst))
        out[c] = len(df)
        print(f"WROTE {dst.name} rows={len(df)} (sample)")
    return out

def process_full_per_country(src, stem, chunk=CHUNK):
    total = pl.scan_csv(str(src), separator="\t", infer_schema_length=10000).select(pl.len()).collect().item()
    tmpdir = OUT / f"_tmp_{stem}"
    tmpdir.mkdir(exist_ok=True)
    seen = set()
    for off in range(0, total, chunk):
        df = pl.scan_csv(str(src), separator="\t", infer_schema_length=10000).slice(off, chunk).collect()
        df = normalize_frame(df)
        for c in df["country"].unique().to_list():
            part = df.filter(pl.col("country") == c)
            p = tmpdir / f"{c}_{off}.parquet"
            part.write_parquet(str(p))
            seen.add(c)
        print(f"{Path(src).name} {off + len(df)}/{total}")
    out = {}
    for c in sorted(seen):
        full = pl.concat([pl.read_parquet(str(f))
                          for f in sorted(tmpdir.glob(f"{c}_*.parquet"))], how="diagonal")
        dst = OUT / f"{stem}_{c}.parquet"
        full.write_parquet(str(dst))
        out[c] = len(full)
        print(f"WROTE {dst.name} rows={len(full)}")
    for f in tmpdir.glob("*.parquet"):
        f.unlink()
    tmpdir.rmdir()
    return out

counts = {}
SPEC = [("s1tr", "train_s1"), ("s1te", "test_s1"), ("s2tr", "train_s2"), ("s3tr", "train_s3")]
for k, stem in SPEC:
    if FULL_RUN:
        counts[k] = process_full_per_country(P[k], stem)
    else:
        counts[k] = process_sample_per_country(P[k], stem)
json.dump(counts, open(OUT / "processed_counts.json", "w"), indent=1)
print("DONE:", counts)
""")

c05 = nbf.v4.new_code_cell(r"""# ---- Verify: before/after, FR inspection, schema freeze ----
stem = "train_s1"
pat = str(OUT / f"{stem}_US.parquet")
df = pl.read_parquet(pat).select(["business_name", "core_name", "legal_form", "name_lat",
    "business_address", "addr_norm", "house_no", "zip_pin_cp", "city", "state_code"]).head(5)
for r in df.to_dicts():
    print(safe(r["business_name"]), "->", safe(r["core_name"]), "| legal=", safe(r["legal_form"]),
          "| lat=", safe(r["name_lat"]))
    print("   ", safe(r["business_address"]), "->", safe(r["addr_norm"]),
          "| house=", r["house_no"], "pin=", r["zip_pin_cp"], "city=", safe(r["city"]), "st=", safe(r["state_code"]))
frf = str(OUT / "test_s1_France.parquet")
fr = pl.read_parquet(frf).select(["business_name", "core_name", "business_address", "addr_norm"]).head(3)
print("--- FR inspection (test S1 France) ---")
for r in fr.to_dicts():
    print(safe(r["business_name"]), "->", safe(r["core_name"]))
    print("   ", safe(r["business_address"]), "->", safe(r["addr_norm"]))
print("schema=", pl.read_parquet(pat).columns)
print("counts=", json.load(open(OUT / "processed_counts.json")))
print("verify OK")
""")

nb.cells = [c00, c01, c02, c02b, c02c, c03, c04, c05]
nbf.write(nb, "01_eda_preprocessing.ipynb")
print("wrote 01_eda_preprocessing.ipynb")
