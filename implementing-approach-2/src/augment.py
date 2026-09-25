"""Ditto-style augmentation adapted to feature-based matching (no LM needed).

Ops (Ditto Table 2, MixDA-less since our features are explicit, not LM states):
- span_del: delete a random token span (<=3) from core or address
- span_shuffle: shuffle a random span (<=4)
- attr_del: drop one side's address entirely (trains missing-address robustness;
  Ditto chose attr_del for exactly this missing-value reason)
entry_swap is a NO-OP here (all our pair features are symmetric) — documented, skipped.

Applied to positive pairs (also eases the 1:1000 imbalance); labels preserved.
"""
import random

from features import _jac  # noqa: F401  (shared primitive)
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


def _toks(s):
    return (s or "").split()


def span_del(s, rng, max_len=3):
    t = _toks(s)
    if len(t) < 2:
        return s
    i = rng.randrange(len(t))
    j = min(len(t), i + rng.randint(1, max_len))
    return " ".join(t[:i] + t[j:])


def span_shuffle(s, rng, max_len=4):
    t = _toks(s)
    if len(t) < 2:
        return s
    i = rng.randrange(len(t))
    j = min(len(t), i + rng.randint(2, max_len))
    sub = t[i:j]
    rng.shuffle(sub)
    return " ".join(t[:i] + sub + t[j:])


def augment_pair(core1, addr1, core2, addr2, op, rng):
    """Returns perturbed (core1, addr1, core2, addr2); op in del/swap/attr_del/all."""
    if op == "all":
        op = rng.choice(["del", "swap", "attr_del"])
    c1, a1, c2, a2 = core1, addr1, core2, addr2
    side = rng.choice([1, 2])
    if op == "del":
        if side == 1:
            c1, a1 = span_del(c1, rng), span_del(a1, rng)
        else:
            c2, a2 = span_del(c2, rng), span_del(a2, rng)
    elif op == "swap":
        if side == 1:
            c1, a1 = span_shuffle(c1, rng), span_shuffle(a1, rng)
        else:
            c2, a2 = span_shuffle(c2, rng), span_shuffle(a2, rng)
    elif op == "attr_del":
        if side == 1:
            a1 = ""
        else:
            a2 = ""
    return c1, a1, c2, a2


def _tri(x, y):
    if x and y:
        return 1 if x == y else -1
    return 0


def augment_positives(pairs_df, s1map, cdmap, scols, ccols, ops=("del", "swap", "attr_del"),
                      reps=2, seed=42):
    """pairs_df: positive (s1, mid, +prov/tf-idf cols). Returns DataFrame of augmented
    feature rows with the same schema as features.featurize output (+ label=1)."""
    import polars as pl
    rng = random.Random(seed)
    si = {c: i for i, c in enumerate(scols)}
    ci = {c: i for i, c in enumerate(ccols)}
    rows = {k: [] for k in
            ["name_fuzz", "name_set", "name_jw", "name_jac", "name_len_diff",
             "core_exact", "legal_agree", "domain_exact", "phonetic_match",
             "addr_fuzz", "addr_set", "addr_jac", "house_tri", "pin_tri",
             "city_match", "state_match", "missing_addr", "tfidf_score", "n_prov",
             "label", "s1", "mid"]}
    prov = [c for c in pairs_df.columns if c.startswith("from_")]
    for c in prov:
        rows[c] = []
    base = pairs_df.to_dicts()
    for r in base:
        A, B = s1map[r["s1"]], cdmap[r["mid"]]
        cn1, an1 = A[si["core_name"]] or "", A[si["addr_norm"]] or ""
        cn2, an2 = B[ci["core_name"]] or "", B[ci["addr_norm"]] or ""
        for _ in range(reps):
            op = rng.choice(list(ops))
            q1, g1, q2, g2 = augment_pair(cn1, an1, cn2, an2, op, rng)
            rows["name_fuzz"].append(fuzz.ratio(q1, q2) / 100.0)
            rows["name_set"].append(fuzz.token_set_ratio(q1, q2) / 100.0)
            rows["name_jw"].append(float(JaroWinkler.normalized_similarity(q1, q2)))
            rows["name_jac"].append(_jac(q1, q2))
            rows["name_len_diff"].append(abs(len(q1) - len(q2)) / max(1, max(len(q1), len(q2))))
            rows["core_exact"].append(float(bool(q1) and q1 == q2))
            rows["legal_agree"].append(0.0)
            rows["domain_exact"].append(0.0)
            rows["phonetic_match"].append(0.0)
            rows["addr_fuzz"].append(fuzz.ratio(g1, g2) / 100.0)
            rows["addr_set"].append(fuzz.token_set_ratio(g1, g2) / 100.0)
            rows["addr_jac"].append(_jac(g1, g2))
            rows["house_tri"].append(0)
            rows["pin_tri"].append(0)
            rows["city_match"].append(0.0)
            rows["state_match"].append(0.0)
            rows["missing_addr"].append(float((not g1) or (not g2)))
            rows["tfidf_score"].append(float(r.get("tfidf_score", 0.0)))
            rows["n_prov"].append(sum(1 for c in prov if r.get(c, 0) > 0))
            for c in prov:
                rows[c].append(float(r.get(c, 0) > 0))
            rows["label"].append(1)
            rows["s1"].append(r["s1"])
            rows["mid"].append(r["mid"] + "#aug")
    return pl.DataFrame(rows)
