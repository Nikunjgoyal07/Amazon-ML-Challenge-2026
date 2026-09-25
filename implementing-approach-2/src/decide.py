"""Vetoes + sparsity-aware thresholds + one-owner/singleton decisions + TSV output."""
from metrics import entity_macro_f05, pair_prf


def apply_vetoes(df):
    """Zero proba on hard conflicts (both populated + disagree). Returns proba list."""
    out = df["proba"].to_list()
    if "house_tri" in df.columns and "pin_tri" in df.columns:
        h = df["house_tri"].to_list()
        p = df["pin_tri"].to_list()
        for i in range(len(out)):
            if h[i] == -1 or p[i] == -1:
                out[i] = 0.0
    return out


def tune_thresholds(rows, gtmap, s1_ids, thresholds, t_empties):
    """Per-bin (addr missing?) threshold + global t_empty maximizing macro-F0.5.

    rows: dicts with s1, mid, label, proba, missing_addr. Returns (t_avail, t_missing, t_empty, best_f05).
    """
    best = (-1, None, None, None)
    for ta in thresholds:
        for tm in thresholds:
            for te in t_empties:
                match = decide_all(rows, ta, tm, te)
                f05, _, _ = entity_macro_f05(match, gtmap, s1_ids)
                if f05 > best[0]:
                    best = (f05, ta, tm, te)
    return best[1], best[2], best[3], best[0]


def decide_all(rows, t_avail, t_missing, t_empty, bin_t=None, bin_fn=None):
    """Per-S1 decisions. If bin_t/bin_fn given, threshold = bin_t[bin_fn(r)]
    (N-bin mode); else 2-bin missing_addr mode."""
    per_s1 = {}
    for r in rows:
        per_s1.setdefault(r["s1"], []).append(r)
    match = {}
    for s1, rs in per_s1.items():
        rs = sorted(rs, key=lambda r: -r["proba"])
        if bin_t is not None:
            t = lambda r: bin_t[bin_fn(r)]  # noqa: E731
        else:
            t = lambda r: t_missing if r["missing_addr"] else t_avail  # noqa: E731
        kept = [r["mid"] for r in rs if r["proba"] >= t(r)]
        if not kept or rs[0]["proba"] < t_empty:
            match[s1] = []
        else:
            match[s1] = kept
    return match


def tune_binned(rows, bin_fn, gtmap, s1_ids, thresholds, t_empties):
    """Per-bin threshold by pair-F0.5 within bin, then global t_empty by macro-F0.5.
    Returns ({bin: t}, t_empty, macro_f05)."""
    bins = sorted({bin_fn(r) for r in rows})
    bin_t = {}
    for b in bins:
        sub = [r for r in rows if bin_fn(r) == b]
        y = [r["label"] for r in sub]
        best_t, best_f = thresholds[0], -1.0
        for t in thresholds:
            _, _, f = pair_prf(y, [int(r["proba"] >= t) for r in sub])
            if f > best_f:
                best_f, best_t = f, t
        bin_t[b] = best_t
    best_te, best_f = t_empties[0], -1.0
    for te in t_empties:
        m = decide_all(rows, 0, 0, te, bin_t=bin_t, bin_fn=bin_fn)
        f, _, _ = entity_macro_f05(m, gtmap, s1_ids)
        if f > best_f:
            best_f, best_te = f, te
    return bin_t, best_te, best_f


def resolve_one_owner(rows, t_pair):
    """Each mid -> argmax S1 with proba >= t_pair. Returns {s1: [mid]}."""
    best = {}
    for r in rows:
        if r["proba"] >= t_pair and (r["mid"] not in best or
                                       r["proba"] > best[r["mid"]][1]):
            best[r["mid"]] = (r["s1"], r["proba"])
    match = {}
    for mid, (s1, _) in best.items():
        match.setdefault(s1, []).append(mid)
    return match


def write_matching_tsv(match, s1_ids, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in s1_ids:
            f.write(f"{s1}\t{','.join(match.get(s1, []))}\n")
