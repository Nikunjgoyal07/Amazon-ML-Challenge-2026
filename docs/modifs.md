# modifs.md — Evidence-Based Modifications for `challenge-baseline`

**Rule of this document:** every recommendation below carries a measured number from
simulations run against the uploaded train TSVs. No assumptions, no "should help" —
each item states the setup, the delta, the caveats, and the gate. Where measurement
was impossible (France has no labels), that is stated instead of faked.

**Nothing in the branch was changed.** All evidence comes from scratch code in
`sim_typo/` (`expA_decisions.py`, `expB2/B3/B4_*`, `expC_parser.py`, `expC2_f05.py`,
`expD_centering.py`, `step8_ablation.py`) + result JSONs beside them. Both clones
verify clean via `git status --porcelain` (empty).

**Shared harness (so deltas are comparable):** 3,200 eval S1 (1,500 US + 1,500 India
with matches + 200 singletons), 11,084 true pairs, 15K distractor pool (true matches
excluded), LightGBM (63 leaves, lr 0.1, early stopping), 3-fold grouped-by-S1 OOF,
per-run tuned threshold + one-owner, exact macro-F0.5. Buckets are oracle-built
(trues force-added) → **absolute levels are inflated; only deltas are evidence.**
Second caveat: buckets were ranked by name similarity, so retrieval features leak
name signal — "remove names" rows measure redundancy, not value (see §5).

---

## 1. Blocking (where the ceiling lives)

### 1.1 ADD a char-3-gram TF-IDF second view — highest-value blocking change

-Setup (`expB2`): per S1, rank 15K distractors + all its trues by sklearn TF-IDF
char-3-gram on `core_name`, top-30; same universe for the token_set proxy and unions.

| Method (30/S1 unless noted) | Edge recall | Full-list recall | Ceiling F0.5 |
|---|---|---|---|
| Dense proxy (token_set) | 0.8227 | 0.5725 | 0.8513 |
| TF-IDF char-3-gram | **0.8572** (+0.035) | **0.6622** (+0.090) | **0.8649** (+0.014) |
| Union (~44/S1) | **0.8708** (+0.048) | **0.6844** (+0.111) | **0.8714** (+0.020) |
| Union capped ~35 (dense30 + ≤5 TF-IDF-exclusive) | 0.8427 (+0.020) | 0.6169 (+0.044) | 0.8590 (+0.008) |

TF-IDF alone beats the token proxy (different signal: typo-robust grams + IDF
weighting), and the capped union buys +0.044 full-list recall for +5 candidates.
TF-IDF indexing cost was ~2 s for 26K docs on CPU (vs 221 s for the pairwise pass) —
at full scale it is a chunked CPU job, no GPU needed.
**Recommendation:** implement TF-IDF top-k on `core_name` (+ address variant) as an
additive pass; cap total ≈35/S1; gate on test-bed ceiling lift.

### 1.2 Address keys: (house,city) yes — (house,pin) no, house-only never raw

Measured key variants over the same universe (`expB3`, loose fields; `expB4`
union-marginal over the dense+TF-IDF union):

| Key | Share of ALL true pairs retrievable | Union-marginal recall | Extra cands/S1 |
|---|---|---|---|
| (house, pin) | 0.027 | +0.001 | 0.01 |
| (house, city) | 0.131 | **+0.007** (ceiling 0.8714→0.8743) | **0.03** |
| (pin, country) | 0.041 | — | 0.15 |
| (house, country) | 0.358 | — | 7.0 (p95: 31 — mega-blocks, reject) |

Read it correctly: keys catch little *over the union* (most key-findable pairs are
already found) — their value concentrates in **address-only true pairs** (names
replaced by domains/random words), which are rare in a random sample but real in
production (notably France crowding). The repo's "18–28% of misses" claim did NOT
reproduce here (measured +0.007 ceiling) — likely because that estimate assumed
full-index embedding misses, a different miss composition.
**Recommendation:** implement (house,city) keys only (+0.03 cands, near-free);
never house-only without a block-size cap (drop blocks >500, require a second
conjunct like street token). Re-audit on the France-heavy test bed, not the sample.

### 1.3 Union design (what to actually build)

dense-30 + TF-IDF-exclusive ≤5 + (house,city)-key hits, dedup, cap ~35/S1, and save
**per-pair provenance** (which passes fired) as matcher features. Expected from
measurements: edge recall ≈ 0.85 (+0.03 over dense alone), full-list ≈ 0.62 (+0.05),
at ~35 vs 30 candidates. Rank-fusion (RRF) is optional polish; rank-then-truncate
per source already captures most of it.

### 1.4 Learned blocking (SC-Block-style fine-tune): phase 2, gated

Not run (needs GPU fine-tune time + hard-negative mining). Literature (WDC-Block:
halved candidate sets, 1.5–4× faster pipelines, ~5 min training) + our 7.6M labeled
links make it the highest-upside *learned* step — but only after §§1.1–1.2 bank
their cheap gains. Gate: leave-one-country-out ceiling lift (the France proxy),
never sample-only recall.

---

## 2. Preprocessing + parser (the coverage story)

### 2.1 FIX house-number extraction; ADD as new columns (measured: 7% → 59–64% coverage)

Current `HOUSE_RE` misses bare leading numbers (`45 Rue du Port` — all of France),
`Shop/Unit/Office/Survey/S.No/Gat` prefixes, and fractions (`120/7/1`).
Improved extractor (`expC_parser.py`): extended prefixes + bare-leading-number
fallback; PIN = trailing 5/6-digit run (space-tolerant `400 001`), skipping runs
equal to the house number (fixes the documented `17560 Ellis Road → pin 17560`
false positive — that number is a house, not a PIN).

| Coverage (non-missing rate) | house v1 → v2 | pin v1 → v2 |
|---|---|---|
| Sim entities (18K) | 0.074 → **0.592** | 0.071 → 0.029 |
| 100K train S1 sample | 0.022 → **0.645** | 0.067 → 0.021 |
| True-pair house agreement | 0.016 → **0.346** (+33pp newly agreeing) | — |

Two subtleties, both measured: (a) pin coverage *drops* because v1's PINs were
partly misattributed house numbers (v1 pin-agreement 4.1% → v2 0.9%, while the lost
pairs moved into house-agreement) — so **ADD v2 columns, don't replace v1**; let
LightGBM learn trailing-PIN ≫ anywhere-PIN trust. (b) 8.9% of true pairs newly
*conflict* on house (multi-unit buildings) — conflict-vs-missing must stay separate
features (they already are), since conflict is evidence too.
Matcher delta with v2 features added: **+0.0002 macro** (features subsume it at
macro level; value is in blocking keys §1.2 + close-call adjudication, not the mean).

### 2.2 Do NOT "fix" accent-stripping by deleting it — branch around it

01's stripper deletes Unicode `Mn` marks = French accents AND Indic vowel
signs/viramas (`స్మార్ట్`→`స్మార్ట`; then transliterates as `samaarata` not `smart` —
measured in the hybrid sim). Deleting the stripper breaks French (`École`≠`ecole`
for fuzzy features). Correct fix (see `hybridplanner.md` in the clone):
script-detect on RAW text → native-script words to IndicXlit pre-strip → rejoin the
normal pipeline. anyascii stays post-strip (measured position-invariant: `nmste`
raw and stripped).

---

## 3. Matcher (what to keep, cut, tune)

| Experiment (`expA`) | Macro | Δ | Verdict |
|---|---|---|---|
| full (22 feats) | 0.9927 | — | baseline |
| −address sims | 0.9426 | **−0.050** | untouchable (France-crowding defense) |
| −name sims | 0.9916 | −0.001 | redundant w/ name-built buckets, not valueless (§5 caveat) |
| −structured | 0.9929 | ≈0 | keep (rare-but-decisive; fix coverage §2.1) |
| −legal/phonetic/domain | 0.9919 | −0.001 | cut `phon_same` only if forced |
| −flags | 0.9927 | 0 | keep (singleton gating, zero cost) |
| drop `lat_ratio` alone | 0.9931 | +0.0004 (noise) | drop it OR replace with hybrid-xlit similarity |
| drop `phon_same` alone | 0.9929 | +0.0002 (noise) | cuttable |
| drop `domain_ratio` alone | 0.9927 | 0 | untestable here (no website-names sampled); keep |
| LightGBM grid (31/63/127 leaves × lr 0.1/0.05) | 0.9925–0.9931 | ±0.0005 | confirms repo claim; take **leaves=31** (simplest, marginally best) |
| Train on 1/3 S1 | 0.9918 | −0.0009 | `TRAIN_S1_PER_COUNTRY` can drop to ~100K for memory |
| Candidates 30→20 | tuned −0.003 | matches repo's 0.967→0.964 | keep 30; 20 only if matcher runtime binds; 30→10 costs −0.012 |
| Expected-F0.5 prefix vs global-t | 0.9913 vs 0.9927 | −0.0014 | keep dropped (confirms repo's +0.0002) |
| Singleton `has_match` gate | 0.9924 | −0.0003 | skip (gate learned to stay out of the way) |

Feature importance on this sample is address-dominated (`addr_tset` ~92% gain —
sample artifact of random distractors; the notebook's 56% `cand_margin` reflects
real full-index competition the sample lacks — do NOT reweight the model from this).

---

## 4. Embeddings (confirmed, don't touch)

Mean-centering re-verified at sample scale on CPU (`expD`, India slice, 1,583 S1 ×
11.5K pool, real `multilingual-e5-small`): mean raw cosine **0.807** (anisotropy
confirmed) → edge recall 0.9914→**0.9940** (+0.003), full-list 0.9716→**0.9810**
(+0.009) with centering. Keep exact GPU search; the prior IVF-PQ rejection
(85.5% vs 95.3%) stands unchallenged — HNSW/IVF-Flat are standby options only if a
session lacks 2×T4.

---

## 5. Bottlenecks & costs (measured wall-times, 8-CPU + CPU-torch box)

| Operation | Cost | Note |
|---|---|---|
| 01-style normalize, 29K entities | ~10 s | scales linearly; full 24M ≈ 2–3 h CPU single-thread — parallelize by country |
| Pair features, 107K pairs (rapidfuzz `cpdist`, all cores) | ~15 s | full 52M pairs ≈ 2 h CPU; chunk (`PAIRS_PER_CHUNK`) as today |
| LightGBM train+OOF (22 feats, 107K pairs, 3-fold) | ~15 s | full 12M pairs ≈ tens of minutes CPU — the documented 3–4 h holds |
| Dense-proxy ranking 3.2K×26K (`cdist`) | ~4 min | real e5 GPU path is minutes/country — no change needed |
| TF-IDF char-3gram index+query (26K docs) | ~2 s | the cheap pass — full scale stays CPU-tractable |
| e5-small CPU encode, 13K texts | ~4.5 min | GPU path stays; CPU fallback viable for slices only |
| IndicXlit greedy CPU | ~0.12 s/word | cache unique words (vocab-bounded); GPU-batch at scale |
| Full-size risks (not sample risks) | — | Kaggle 20 GB output cap (don't save embeddings), 30 GB RAM (chunk), 5 subs/day (validate locally first) |

## 6. France protocol (no labels → process, not numbers)

Keep notebook 04's leaderboard tuning (one country per submission so deltas
attribute). Add the early-warning the sample suggests: compare France's
*predicted-match rate / matches-per-matched-S1* against US/India after every run
(the repo's v1 showed France 95.8%/3.67 vs US 94.1%/3.52 — lookalike over-matching).
If the warning fires, tighten `COUNTRY_THRESHOLDS["France"]` before touching features.

## 7. Workarounds & small fixes checklist

- TSV empties read as NULL → `fill_null("")` or 123K singletons vanish silently.
- Build script-regexes programmatically from code-point tuples (a hand-written
  `\x24F` class silently matched Indic as Latin and cost an entire sim round).
- Bare-leading numbers as houses reclassifies trailing-number PIN false positives
  away — keep both PIN definitions as separate features.
- Cap key-blocks (>500) and never ship house-only keys raw (+7 cands, p95 31).
- Telugu ZWJ sequences pass through the normalizer fine — no special-casing needed.
- `XLIT_DIR=None` must reproduce today's outputs byte-for-byte (diff `processed/`
  in sample mode) — the hybrid stays one env var from off.

## 8. Priority build order (gates in parentheses)

1. Parser v2 as new columns + (house,city) key pass (§§2.1, 1.2 — hours; gate: key
   hit-rate + test-bed ceiling).
2. TF-IDF char-3-gram pass, cap ~35/S1 with provenance flags (§1.1, §1.3 — days;
   gate: +full-list recall).
3. Drop `phon_same`/`lat_ratio` (or swap lat for hybrid-xlit sim), leaves=31,
   optional train-size cut (§3 — minutes; gate: OOF no-drop).
4. Hybrid blocking pilot per `hybridplanner.md` (§2.2 here; gate: OOF lift).
5. SC-Block fine-tune (§1.4 — weeks; gate: leave-one-country-out ceiling).
6. Re-run France watch (§6) + leaderboard tuning via 04.

**Rollback rule:** every item is additive (new columns/passes/features); any gate
failure disables that item only — the 0.935 baseline is never at risk.
