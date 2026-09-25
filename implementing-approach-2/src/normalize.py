"""Vendored normalizer (single importable copy of notebook-01 logic + normalize_frame).

Source of truth for all downstream stages. Notebook 01 remains the EDA owner;
this module is what blocking/inference import. MIT-safe: stdlib + anyascii (ISC).
"""
import re
import unicodedata

try:
    from anyascii import anyascii
    HAS_AA = True
except ImportError:  # Kaggle-cold fallback (pip installed by notebook preamble)
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anyascii"])
    from anyascii import anyascii
    HAS_AA = True

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

NEW_COLS = ["name_norm", "core_name", "legal_form", "domain_stem", "name_lat",
            "phonetic_code", "has_honorific", "addr_norm", "street_tokens", "house_no",
            "zip_pin_cp", "city", "state_code", "empty_addr", "missing_pin",
            "missing_city", "landmark_flag"]


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
        core_toks = toks  # name is only a legal word (e.g. "Private"); keep it
    core = " ".join(core_toks)
    return {"name_norm": s, "core_name": core, "legal_form": " ".join(legal),
            "domain_stem": stem, "name_lat": anyascii(core),
            "phonetic_code": soundex(core.split(" ")[0]) if core else "",
            "has_honorific": hon}


def extract_city_state(s):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    city, state = "", ""
    if parts:
        toks = parts[-1].split()
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


def normalize_frame(df):
    """Add all NEW_COLS to a raw frame (entity_id/business_name/business_address/country)."""
    import polars as pl
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
