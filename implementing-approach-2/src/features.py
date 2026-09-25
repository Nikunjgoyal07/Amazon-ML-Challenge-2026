"""Pair feature engineering: lexical + address + structural + missingness + provenance."""
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

NEED = ["core_name", "name_norm", "legal_form", "domain_stem", "phonetic_code",
        "addr_norm", "house_no", "zip_pin_cp", "city", "state_code",
        "empty_addr", "missing_pin"]


def build_lookup(df):
    cols = [c for c in NEED if c in df.columns]
    return {e: r for e, r in
            zip(df["entity_id"].to_list(),
                zip(*[df[c].to_list() for c in cols]))}, cols


def _jac(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _tri(x, y):
    if x and y:
        return 1 if x == y else -1
    return 0


def featurize(pairs, s1map, cdmap, scols, ccols):
    si = {c: i for i, c in enumerate(scols)}
    ci = {c: i for i, c in enumerate(ccols)}
    s1l = pairs["s1"].to_list()
    midl = pairs["mid"].to_list()
    n = len(s1l)
    F = {k: [0.0] * n for k in
         ["name_fuzz", "name_set", "name_jw", "name_jac", "name_len_diff",
          "core_exact", "legal_agree", "domain_exact", "phonetic_match",
          "addr_fuzz", "addr_set", "addr_jac", "house_tri", "pin_tri",
          "city_match", "state_match", "missing_addr", "tfidf_score", "n_prov"]}
    prov = [c for c in pairs.columns if c.startswith("from_")]
    for i, (a, b) in enumerate(zip(s1l, midl)):
        A, B = s1map[a], cdmap[b]
        cn1, cn2 = A[si["core_name"]] or "", B[ci["core_name"]] or ""
        an1, an2 = A[si["addr_norm"]] or "", B[ci["addr_norm"]] or ""
        F["name_fuzz"][i] = fuzz.ratio(cn1, cn2) / 100.0
        F["name_set"][i] = fuzz.token_set_ratio(cn1, cn2) / 100.0
        F["name_jw"][i] = float(JaroWinkler.normalized_similarity(cn1, cn2))
        F["name_jac"][i] = _jac(cn1, cn2)
        F["name_len_diff"][i] = abs(len(cn1) - len(cn2)) / max(1, max(len(cn1), len(cn2)))
        F["core_exact"][i] = float(bool(cn1) and cn1 == cn2)
        la, lb = A[si["legal_form"]] or "", B[ci["legal_form"]] or ""
        F["legal_agree"][i] = float(bool(la or lb) and la == lb)
        da, db = A[si["domain_stem"]] or "", B[ci["domain_stem"]] or ""
        F["domain_exact"][i] = float(bool(da) and da == db)
        pa, pb = A[si["phonetic_code"]] or "", B[ci["phonetic_code"]] or ""
        F["phonetic_match"][i] = float(bool(pa) and pa == pb)
        F["addr_fuzz"][i] = fuzz.ratio(an1, an2) / 100.0
        F["addr_set"][i] = fuzz.token_set_ratio(an1, an2) / 100.0
        F["addr_jac"][i] = _jac(an1, an2)
        F["house_tri"][i] = _tri(A[si["house_no"]], B[ci["house_no"]])
        F["pin_tri"][i] = _tri(A[si["zip_pin_cp"]], B[ci["zip_pin_cp"]])
        F["city_match"][i] = float(bool(A[si["city"]]) and A[si["city"]] == B[ci["city"]])
        F["state_match"][i] = float(bool(A[si["state_code"]]) and
                                    A[si["state_code"]] == B[ci["state_code"]])
        F["missing_addr"][i] = float((not an1) or (not an2))
        F["n_prov"][i] = 0
    for c in prov:
        col = pairs[c].to_list()
        for i, v in enumerate(col):
            F["n_prov"][i] += (v > 0)
        F[c] = [float(v > 0) for v in col]
    if "tfidf_score" in pairs.columns:
        F["tfidf_score"] = [float(v) for v in pairs["tfidf_score"].to_list()]
    order = ["name_fuzz", "name_set", "name_jw", "name_jac", "name_len_diff",
             "core_exact", "legal_agree", "domain_exact", "phonetic_match",
             "addr_fuzz", "addr_set", "addr_jac", "house_tri", "pin_tri",
             "city_match", "state_match", "missing_addr", "tfidf_score", "n_prov"] + prov
    return pl.DataFrame({k: F[k] for k in order}), order
