"""Blocking + pair + entity (macro-F0.5) metrics + submission-format checks."""


def f05(p, r, beta=0.5):
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def blocking_recall(cands, gtmap, s1_ids):
    """Fraction of true pairs retrieved. cands: DataFrame(s1, mid)."""
    tot, hit = 0, 0
    got = {}
    for s1, mid in zip(cands["s1"].to_list(), cands["mid"].to_list()):
        got.setdefault(s1, set()).add(mid)
    for s1 in s1_ids:
        for mid in gtmap.get(s1, ()): 
            tot += 1
            hit += mid in got.get(s1, ())
    return hit / tot if tot else 1.0


def candidate_stats(cands):
    import statistics as st
    per = {}
    for s1, _ in zip(cands["s1"].to_list(), cands["mid"].to_list()):
        per[s1] = per.get(s1, 0) + 1
    v = sorted(per.values()) or [0]
    return {"avg": round(sum(v) / len(v), 1), "median": float(st.median(v)),
            "p95": float(v[int(0.95 * (len(v) - 1))]), "max": v[-1], "n_s1": len(v)}


def pair_prf(y, pred):
    tp = sum(1 for a, b in zip(y, pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, pred) if a == 1 and b == 0)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, f05(p, r)


def entity_macro_f05(match, gtmap, s1_ids):
    """match: {s1: [mid]}. Singleton correct-empty = 1.0."""
    tot, n = 0.0, 0
    sing_tot, sing_hit = 0, 0
    for s1 in s1_ids:
        pred, true = set(match.get(s1, ())), set(gtmap.get(s1, ()))
        if not pred and not true:
            f = 1.0
        elif not pred or not true:
            f = 0.0
        else:
            tp = len(pred & true)
            f = f05(tp / len(pred), tp / len(true))
        tot += f
        n += 1
        if not true:
            sing_tot += 1
            sing_hit += (not pred)
    return tot / n if n else 0.0, (sing_hit / sing_tot if sing_tot else 1.0), sing_tot


def entity_macro_f05_from_probs(match, probs, rows):
    """Adapter for threshold search: rebuild gt from labeled rows."""
    gtmap = {}
    for r in rows:
        if r["label"] == 1:
            gtmap.setdefault(r["s1"], set()).add(r["mid"])
    s1_ids = sorted({r["s1"] for r in rows})
    return entity_macro_f05(match, gtmap, s1_ids)


def check_submission_format(match, s1_ids):
    """Validator-style checks. Returns list of issue strings (empty = clean)."""
    issues = []
    if set(match.keys()) != set(s1_ids):
        issues.append(f"row coverage: {len(match)} match keys vs {len(s1_ids)} S1 ids")
    for s1, mids in match.items():
        if len(mids) != len(set(mids)):
            issues.append(f"dupes inside list for {s1}")
        for m in mids:
            if m.startswith("S1-"):
                issues.append(f"self-match {m} in {s1}")
            elif not (m.startswith("S2-") or m.startswith("S3-")):
                issues.append(f"bad prefix {m} in {s1}")
    return issues
