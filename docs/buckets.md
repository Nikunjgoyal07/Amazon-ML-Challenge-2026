# buckets.md — Blocking / Candidate Generation for This Pipeline

**Scope:** everything that turns 24M records into ~30 candidates per S1. The blocker
sets the **ceiling** (best score a perfect judge could reach); the matcher can only
lose points from there. Our current ceiling: US **0.994**, India **~0.97** (top-30).

**Facts about OUR data that decide which technique wins** (all measured in
`docs/ARCHITECTURE.md`, not assumed):
- 1.73M test S1 queries × ~10M S2/S3 candidates per split; everything per country
  (0 cross-country matches in 7.64M train links).
- Noise: typos, abbreviations, legal-form swaps, reordered/dropped address parts,
  domain-replaced names, native scripts, address-only matches (`Evoavi`).
- France is test-only (zero-shot) and crowded (`Bordeaux Club` × hundreds).
- Hardware: 2× T4 GPUs for embedding/search, CPU for the rest; Kaggle 20 GB output cap.
- Metric is precision-heavy macro-F0.5 → blocking must maximize recall, matching
  handles precision. A blocker is judged on **recall per candidate spent**.

**Vocabulary used below** (one definition each, then used consistently):
- **Pair Completeness (PC) / bucket recall** = of all true matches, share inside the
  buckets. The only blocking metric that moves the ceiling.
- **Pair Quality (PQ)** = of all bucketed pairs, share that are true. Matters for
  matcher cost, not for the ceiling.
- **Reduction Ratio (RR)** = 1 − (bucketed pairs / all possible pairs). How much work
  you avoided. Ours: ~1 − 52M/17B ≈ 0.997.
- **Ceiling macro-F0.5** = score of a perfect judge on your buckets. The number that
  decides whether a blocker is good enough.
- **k** = candidates kept per query. Ours: 30.

```
ALL PAIRS (1.7M x 10M = 17 trillion, impossible)
  │
  ▼  BLOCKING (this document — cheap, high recall)
BUCKETS (1.7M x 30 = 52M pairs, feasible)
  │
  ▼  MATCHING (notebook 03 — expensive, high precision)
MATCHES (subset of buckets)
```

Rule of thumb for every technique below: **report PC and ceiling, not just recall@k
on a sample.** A technique with 99% recall@5 on 10K records can still miss 15% at
full-index density (verified in this repo: sample scores overstated reality
0.974 vs 0.946 until full-size buckets were used).

---

## 1. Exact dense kNN (CURRENT PRODUCTION — the reference everything beats or loses to)

**What.** Embed every record with `multilingual-e5-small` (384-dim, MIT), then for
each S1 compute similarity against EVERY S2/S3 vector of its country with one GPU
matrix multiply, keep top-30. Vectors are mean-centered first (subtract country mean,
re-normalize; fixes e5's anisotropy where all cosines sit at 0.85–1.0).

**Working.**
```
texts ──e5(fp16, 2xT4)──▶ 384-dim unit vectors
S2/S3 vectors: v' = normalize(v - mean_country)      # spread the arrows
per S1 block Q:  S = Q @ B.T        (fp16 matmul on GPU)
                 top 30+32 ──fp32 rescore──▶ top 30   # coarse-to-fine
```

**Why it's good HERE.** 95.3% recall at full density; minutes per country on 2×T4;
zero tuning knobs (no clusters, no probes, no quantization levels); deterministic
and reproducible. It handles scripts, paraphrases and reorderings that kill lexical
methods (`projectsmanipalindia.com` vs its expansion still embeds nearby).

**Example.** S1 `Sharma Trading Co, 12 MG Road` vs S2 `sharma trdaing co, MG road 12`:
typo splits subwords, but 90% of the vector survives → cosine 0.97 → rank 2. Kept.

**Analogy.** A librarian who has read every book and, given a description, walks the
whole library in one sweep and returns the 30 closest — tireless and exact, but you
pay her to walk the full building every time.

**Cost.** O(Q × N × d) FLOPs, all GPU-fused: ~1.7M × 10M × 384 ≈ 6.5e15 per split —
sounds huge, runs in minutes on 2×T4 because it's dense matmul, the GPU's favorite
operation. Memory: country pools fit in T4 RAM in fp16 (why compression was rejected).

**Verdict: KEEP as the backbone.** Every technique below is evaluated as an
*addition* to this (union), never a replacement — unless it beats 95.3% at equal k.

---

## 2. Standard (key) blocking — the oldest trick, still the cheapest recall

**What.** Records sharing an exact blocking key are compared. Key = some function of
the record, e.g. `(house_number, postcode)`, `(city, first-3-letters-of-name)`,
`(phonetic core name, city)`.

**Working.**
```
record ──key_fn──▶ key ──group-by──▶ block ──compare all pairs inside──▶ candidates
S1 "45 Rue du Port, 44000" ──▶ ("45","44000")
S2 "45 rue du port durand 44000" ──▶ ("45","44000")   # same block -> compared
S3 "46 rue du port 44000" ──▶ ("46","44000")          # different block -> never compared
```

**Why it's good HERE.** O(N log N) via hashing — nearly free on CPU, no GPU, no
model. It catches exactly the cases embeddings fumble: **address-only matches**
(`Evoavi` — name unrelated, address identical) and exact-number agreements. The
repo's own analysis: 18–28% of India's bucket misses would be caught by a
"same house number + street word" key for +1–2 candidates/S1.

**Example.** S1 `Evoavi, 45 Rue du Port 44000` vs S2 `Boulangerie Moderne, 45 Rue du
Port 44000`. Embedding similarity: low (names share nothing) → dense rank ~200,
missed. Key `(45, 44000)`: identical → compared → matcher's address features
confirm. This is a true match the current system CANNOT find.

**Analogy.** Sorting mail by postcode before reading names — crude, but a letter
with an illegible name still reaches the right street.

**Failure mode.** One corrupted key character = total miss (`44000` vs `44001`
never meet). Remedy: multi-pass with several keys (union), never a single key.

**Cost.** One hash + group-by per record; blocks sized 2–few-hundred. Watch for
mega-blocks (`(city=Paris)` = 100K records → 10B pairs): cap block size, discard or
sub-key oversized blocks (standard practice: Singhal-style purging).

**Verdict: PILOT FIRST (highest value/hour).** Implement 2–3 passes —
`(house_no, zip_pin_cp)`, `(city, core-name-3gram-key)`, `(street-tokens-set)` —
as an *additive* +1–2 candidates/S1 source. Measure ceiling lift on the test bed
before touching anything else.

---

## 3. Sorted neighborhood — sliding window over a sorted key

**What.** Sort all records by a blocking key (e.g. `city + core_name`), slide a window
of size w over the order, compare everything inside each window.

**Working.**
```
sorted by key:  ... Bordeaux Cafe | Bordeaux Caffe | Bordeaux Club | Bordeaux Club X ...
window w=4:     [cafe, caffe, club] compared; [caffe, club, clubX] compared; ...
```

**Why it can help HERE.** Catches near-duplicates whose keys differ slightly
(`Bordeaux Club` vs `Bordeaux Clubs` sort adjacently) without enumerating all pairs
of a mega-block. Good for France's crowded cities where hundreds of names differ by
one token.

**Example.** `Gupta Sons Delhi` / `Gupta Son Delhi` / `Gupta Sons Delhi Karol Bagh`:
different exact keys, adjacent after sorting → compared → matcher decides.

**Analogy.** Books shelved alphabetically: near-identical titles sit side by side,
so you only check each book against its 4 shelf-neighbors instead of the library.

**Failure mode.** Totally different keys never meet (`Evoavi` vs anything) — same
weakness as key blocking, needs a complementary pass. Key choice is everything;
use 2 sort keys (name-order, address-order) in separate passes.

**Cost.** O(N log N) sort + O(N·w) comparisons. w=10–20 typical.

**Verdict: SKIP for now.** Overlaps heavily with key blocking (§2) + TF-IDF (§4) at
higher implementation cost. Revisit only if crowded-city misses persist after §2+§4.

---

## 4. TF-IDF / BM25 top-k — SPARKLY (the strongest classical blocker — missing here)

**What.** Index S2/S3 texts in an inverted index (word → posting list with term
statistics). For each S1, score candidates by BM25/TF-IDF cosine over character
3-grams (or words) and keep top-k. **Sparkly** (Paulsen et al., PVLDB 2023) showed
plain top-k TF-IDF with a 3-gram tokenizer beats 8 state-of-the-art blockers
(including deep ones) on 15 datasets: 92.5–100% recall at k=10, 96.4–100% at k=20,
98.7–100% at k=50, with output size capped at exactly k·|queries| (perfectly
predictable cost — unlike key blocking's mega-blocks).

**Working.**
```
S2/S3 corpus ──char-3-gram tokenize──▶ inverted index {gram: [(doc, tf)] + idf}
per S1: tokenize ──▶ score docs sharing grams: sum over grams of tf*idf ──▶ top-k
"sharma trading" grams: {$sh, sha, har, arm, rma, ma$, ...}
"sharma trdaing" shares ~80% of grams (typo changes only ~3) ──▶ high score, kept
```

**Why it's good HERE.** Three properties no dense method has: (a) **typo-robust by
construction** — one wrong letter changes only the grams covering it; (b)
**rare-word driven** — IDF automatically upweights `diaspora` over `club`, the same
intuition as our matcher's best new features; (c) **predictable budget** — exactly k
per query, no mega-blocks. Gaps vs deep blockers are biggest on *textual* data
(Sparkly paper) — and our records are pure text. It also needs no GPU and runs on CPU.

**Example.** S1 `Wcstgrove Associates` (typo in data) vs S2 `Westgrove Associates`:
e5 may rank it ~40th (subword split divergence); 3-gram TF-IDF shares ~85% of grams
(`wes`,`est`,`stg`… survive) → rank ~3 → caught. Union of both = caught either way.

**Analogy.** Finding a misspelled street name by matching overlapping letter-triples
instead of whole words — `Wcstgrove` and `Westgrove` share almost all triples, so
the typo is nearly invisible.

**Failure mode.** Pure paraphrases with zero token overlap (`projectsmanipalindia.com`
vs `Manipal Projects India Pvt Ltd` partially overlap — fine; truly disjoint
wordings are missed). That's precisely what dense retrieval catches → union.
Needs sparse top-k machinery at our scale (`sparse_dot_topn`, Lucene-style index,
or chunked GPU matmul on sparse matrices); BM25-3 timed out on the largest WDC
benchmark, so chunk and cap.

**Cost.** Index build O(total grams); query O(grams per query × postings). CPU-only.
At 10M docs × ~100 grams, a chunked implementation runs in tens of minutes on CPU.

**Verdict: PILOT SECOND (with §2).** `name_core` char-3-gram top-k + address TF-IDF
top-k as two union passes. This + §2 + dense is the classic winning union
(Foursquare 7th used 5 retrievers; DeepBlocker proved unions win). Tune k per pass
on the test bed; keep total ≤ ~35/S1.

---

## 5. Character q-gram canopy + clustering (PatentsView style)

**What.** Represent each name by its character 4-gram TF-IDF vector; form overlapping
canopies (loose clusters) by threshold; cluster hierarchically inside each canopy.

**Working.**
```
"bordeaux club" ──4-grams──▶ {bord, orde, rdea, deau, eaux, aux#, ...}
canopy key = rare-gram subset ──▶ candidate canopy of ~hundreds
inside canopy: agglomerative clustering, tuned threshold ──▶ match groups
```

**Why it could help HERE.** Purpose-built for **organization-name disambiguation at
millions scale** (PatentsView production system). Handles typos + reorderings +
affix variation in one structure, and thresholds are learned, not hand-set.

**Example.** `Shri Balaji Traders`, `Balaji Traders`, `Balaji Trader` → same canopy
via shared rare grams (`bala`,`laji`), clustered together despite honorific/plural
noise that exact keys miss.

**Analogy.** Tents (canopies) pitched over a festival crowd by shirt color — friends
in matching shirts land in the same tent, then you sort out individuals tent by tent
instead of searching the whole field.

**Failure mode.** Threshold tuning is fiddly; overlapping canopies can explode
candidate counts on generic names (`Club` canopies = half of France). Strictly
weaker-controlled than top-k methods (no hard cap per query).

**Cost.** Canopy formation + in-canopy clustering; needs the same sparse machinery
as §4 with less predictable output size.

**Verdict: SKIP.** §4 (top-k TF-IDF) dominates it on the properties we care about
(hard per-query cap, no thresholds, same typo robustness). Cite as considered.

---

## 6. Phonetic blocking (Soundex / Metaphone)

**What.** Bucket records by how names SOUND (`Smith`/`Smyth` → same code), using
Soundex (first letter + consonant classes) or Metaphone (English-phonetics rules).

**Working.**
```
"Smith" ──soundex──▶ S530 ; "Smyth" ──soundex──▶ S530 → same bucket, compared
01 already stores phonetic_code (soundex of first word) — the column exists,
the blocking pass using it does not.
```

**Why it could help HERE.** spelling variants that preserve sound
(`Krishna`/`Krishnah`, transliteration variants across scripts) collapse to one
bucket for free. The column already exists in `processed/` — cheapest pilot after §2.

**Example.** S1 `Ecole Moderne` vs S2 `Ecoll Moderne` (typo): soundex `E245` both →
compared. (Though §4 catches this too.)

**Analogy.** A bouncer who admits anyone whose name *sounds* like the guest list —
catches creative spellings, occasionally admits a stranger.

**Failure mode.** English-phonetics rules misfire on French (`Beauchamp` vs
`Beecham`) and badly on transliterated Indic names; low precision → use only as a
low-weight union pass, never alone. Keep as feature (already is), pass optional.

**Cost.** Trivial (string function + group-by).

**Verdict: CHEAP ADD-ON.** One phonetic+city pass in the §2 union batch. Expect
small gains; keep iff ceiling moves.

---

## 7. Address-key blocking (domain pass for THIS dataset)

**What.** Keys from parsed address parts: `(house_no, zip_pin_cp)`,
`(street rare-token, city)`, `(phone-ish digit strings)`. 01 already parses
`house_no/zip_pin_cp/city/state_code` — again, columns exist, pass doesn't.

**Working.**
```
S1 "H.no 780, MG Road, 500034" ──▶ ("780","500034") + street key {mg, road}
S2 "780 MG Rd 500034" ──▶ same keys → compared even though names are unrelated
missing PIN ≠ conflicting PIN: keys with empty parts go to a separate "unknown" lane
```

**Why it's good HERE.** The ONLY pass that finds **address-only matches** (names
replaced by domains/random words — a known synthetic noise type here) and the
France-crowding antidote (same name × different street). The repo's estimate:
18–28% of India's misses recoverable for +1–2 candidates/S1.

**Example.** S1 `Evoavi, 45 Rue du Port 44000` vs S2 `Boulangerie Moderne, 45 Rue du
Port 44000` — dense rank ~200 (missed), key `(45,44000)` → compared → matcher
confirms via house+street+city agreement.

**Analogy.** Finding twins separated at birth via fingerprints (address numbers)
when their names (faces) were changed.

**Failure mode.** Corrupted house numbers (observed: true pairs with different
numbers) and shared buildings (non-matches sharing an address). Mitigate: keys are
*candidates*, the matcher (with conflict-vs-missing features) convicts.

**Cost.** Group-bys on parsed columns; cap mega-blocks (entire PIN areas).

**Verdict: PILOT WITH §2** (it's the same work batch — the parsers already exist).

---

## 8. HNSW graph index (best recall-per-RAM among approximate methods)

**What.** Multi-layer navigable graph over the e5 vectors: greedy walk from top layer
down, `efSearch` controls effort. Query returns approximate top-k in ~log(N) hops.

**Working.**
```
build: insert vectors one by one, linking each to M nearest in layered graphs
query: enter top layer ──greedy step to nearest──▶ descend ──▶ widen (efSearch)
       ──▶ return best k
FAISS one-liner: faiss.index_factory(384, "HNSW32")
```

**Why it could help HERE.** Published ANN comparisons (incl. a 2025 ER-focused
study) agree: **graph > IVF > LSH for in-memory recall** (~0.95–0.99 recall@10 at
~⅓ corpus scanned). If exact GPU search ever becomes infeasible (bigger data, no
GPU session), HNSW is the first fallback that preserves ~all recall — unlike IVF-PQ
which cost us 10 points. CPU-only, very low query latency.

**Example.** Same 1.7M queries × 10M vectors: HNSW returns 29.5 of the true top-30
on average (vs 30.0 exact) in milliseconds per query on CPU.

**Analogy.** Highway network with local roads: zoom into the right neighborhood via
motorways (top layers), then door-to-door locally (bottom layer) — no need to walk
every street like exact search does.

**Failure mode.** High RAM (full vectors + edge lists ≈ 2–3× exact-search memory);
slow builds (hours at 10M); awkward updates; recall never 100% — the 0.5–2% loss
lands disproportionately on hard near-duplicate pairs (exactly our margin cases).

**Cost.** Build O(N log N) slow; query O(log N) fast; RAM high.

**Verdict: STANDBY, don't build.** Exact GPU search fits and is perfect. Revisit only
if a Kaggle session lacks 2×T4 or the corpus grows past GPU RAM.

---

## 9. Partition-based ANN done right: IVF-flat / ScaNN (when exact doesn't fit)

**What.** Partition space into coarse clusters (IVF), optionally compress with
anisotropic quantization (ScaNN: score-aware losses + asymmetric distance + reorder).
A 2025 ER study: partition methods (esp. ScaNN) win high-throughput/moderate-recall;
a 2026 Milvus study: ScaNN + IVF-SQ8 top throughput, HNSW top recall at higher RAM.

**Working.**
```
train coarse quantizer (k-means over sample) ──▶ assign vectors to cells
query: find nearest nprobe cells ──▶ exact/quantized scan inside ──▶ top-k
ScaNN adds: learned quantization minimizing SCORE error (not reconstruction) +
            rescore top candidates in full precision (same coarse-to-fine as ours)
```

**Why it matters HERE.** Mostly as documented context for WHY exact was chosen: our
own measurement (IVF-PQ 85.5% vs exact 95.3%) sits exactly on the published
recall/work curves. IVF-Flat (no compression, just partitioning) would lose far less
than IVF-PQ and is the sane middle ground if exact ever breaks.

**Example.** IVF16384,Flat with nprobe=64: searches 64/16384 cells (~0.4% of data)
per query, recovers ~99% of exact top-30 at ~50× less compute.

**Analogy.** Library with 16,384 rooms: check the 64 most plausible rooms instead of
every shelf — you miss the book only if it was shelved in the wrong wing.

**Cost.** Needs cluster training sample; nprobe tuning per country; cell-size skew
on generic names.

**Verdict: SKIP unless exact breaks** (same trigger as §8). If triggered, try
IVF-Flat first (no quantization loss), ScaNN second.

---

## 10. SC-Block: supervised contrastive blocking (the learned upgrade)

**What.** Fine-tune the encoder with supervised contrastive loss on the 7.6M labeled
train links so same-business records cluster tightly, then kNN. SC-Block (2023)
reports candidate sets ~HALF the size at equal recall (99.5%), pipelines 1.5–2×
faster (4× on the huge WDC benchmark: 30h→8h), training cost ~5 minutes.
SC-Block++ (2025) adds AdaFlood regularization for better generalization.

**Working.**
```
batch: (S1 anchor, its true S2/S3 positives, in-batch negatives incl. lookalikes)
loss: pull positives together, push negatives apart (temperature τ)
      + hard negatives: same-name/different-address pairs (our France crowding!)
encode all ──▶ kNN (smaller k suffices: clusters are tighter) ──▶ buckets
```

**Why it's good HERE.** We have what most papers lack: **7.6M labeled links** — the
exact fuel supervised contrastive learning wants. It directly attacks our two
hard slices: typo'd names (learned robustness > subword luck) and crowded lookalikes
(hard negatives teach the encoder that `Bordeaux Club ≠ Bordeaux Club X`). France
benefits without French labels IF the encoder learns generic discrimination (must
verify on the leave-one-country-out proxy — same protocol as the matcher's transfer
check). WDC-Block evidence: the advantage GROWS with vocabulary size/density, and
our index is far denser than academic benchmarks.

**Example.** Pre-fine-tune: `Bordeaux Club` (S1) has 300 lookalikes within cosine
0.95–0.99; true match ranks 12th. Post-fine-tune with same-street negatives: true
match (same street tokens) pulls to rank 1, lookalikes pushed below 0.9.

**Analogy.** A wine judge trained on labeled tastings (ours: 7.6M) vs a talented
amateur (off-the-shelf e5): both taste well, but only the trained judge reliably
separates twin vintages.

**Failure mode.** Overfitting to train noise patterns (the noise is synthetic and
learnable — good — but France's French-specific patterns are unseen; the transfer
check is the gate). Training infra on Kaggle (needs GPU hours + hard-negative
mining pipeline). Temperature/τ tuning matters (uniformity-vs-tolerance trade-off
flagged in SC-Block++ paper).

**Cost.** Fine-tune mE5-small (118M) on ~1M pairs: hours on 2×T4, one-time. Then
search is identical to today (same exact GPU code, smaller k possible: 30→20?).

**Verdict: PILOT AFTER the union (§2+§4) lands.** Highest-upside learned blocker,
but only after cheap wins are banked. Gate strictly on leave-one-country-out
ceiling + full-index recall (never sample-only).

---

## 11. UniBlocker / self-supervised dense (no-labels alternative)

**What.** Self-supervised contrastive pre-training on domain-independent tabular
data, then kNN with NO domain fine-tuning. UniBlocker (2024): +10% mAP over prior
self-supervised dense blockers, comparable to sparse SOTA, and **ensembling
UniBlocker + Sparkly gained up to +5% pair-completeness** (the union thesis again).
Older siblings: DeepBlocker (autoencoder/cross-tuple — best on dirty data),
Sudowoodo, Barlow-Twins/SimCLR variants.

**Why it matters HERE.** Zero-shot France: a universal blocker needs no French
labels by construction. But: we HAVE 7.6M labels, so supervised SC-Block (§10)
strictly dominates the motivation — self-supervised is the answer when labels are
absent, and ours aren't.

**Example.** Same as §10's Bordeaux example, but the encoder was pre-trained on
web tables generally rather than our links — separates most twins, misses the
trickiest (same-street-different-shop) that only our labels teach.

**Analogy.** A sommelier trained on all wines worldwide vs our judge trained on
Bordeaux specifically — broader, but worse at twin Bordeaux vintages.

**Verdict: SKIP (dominated by §10 given our labels).** The one exportable idea:
UniBlocker+Sparkly's +5% ensemble gain re-confirms the union design (§14).

---

## 12. Metric learning with ArcFace (Foursquare 3rd place)

**What.** Treat each S1 ID as a CLASS; train the encoder to classify records into
2.2M classes with angular margin loss. Same-business records collapse into tight
angular clusters; then bi-encoder kNN. Won Foursquare 3rd + Shopee 1st (as one leg).

**Why it could help HERE.** Our S1 IDs are ready-made classes (2.2M of them).
Angular margin explicitly optimizes the crowded-lookalike separation France needs.
Can reuse the same encoder for blocking (kNN) AND matching (cosine feature) —
one training run, two consumers (also true of §10).

**Example.** 300 `Bordeaux Club*` records across 40 true entities: ArcFace forces
40 tight angular clusters with margins; a query's true matches sit inside its
cluster cone, lookalikes outside it.

**Analogy.** Assigning every business its own reserved parking zone with painted
buffer lines — lookalikes physically can't park in your spot.

**Failure mode.** 2.2M-way classification head is heavy (sampled-softmax/partial-FC
needed); long-tail entities (1–2 records) train poorly; needs the same GPU budget
as §10 with trickier engineering. Less literature for blocking specifically than
SC-Block.

**Cost.** Highest engineering cost of all options here.

**Verdict: SKIP unless §10 succeeds first.** If SC-Block works, ArcFace is the
named follow-up experiment; if SC-Block fails, ArcFace likely fails harder.

---

## 13. Two-stage retrieve-and-rerank (Foursquare 2nd place pattern)

**What.** Coarse retriever (cheap: TF-IDF or small encoder, huge k) → light
transformer re-ranker → final top-k. Foursquare's 2nd place used a Transformer
blocking stage between coarse candidates and the matcher.

**Why it could help HERE.** Our exact search already IS coarse-to-fine (fp16 sweep
→ fp32 rescore of top 62). A learned re-ranker (tiny cross-encoder or LateInteraction
like ColBERT) between buckets and LightGBM could rescue pairs ranked 25–60 into
the top-30 — but that's matcher territory (notebook 03's uncertain band), not
blocking. As pure blocking, it adds latency for recall we already have (95.3%).

**Verdict: SKIP as blocking.** The re-ranking instinct belongs in the matcher
(Ditto-style cross-encoder on LightGBM's uncertain band — already in PLAN.md).

---

## 14. The union: how to combine passes (the actual recommendation)

No single pass covers all noise types — DeepBlocker, Foursquare-7th (5 retrievers),
and UniBlocker+Sparkly (+5%) all prove unions win. Design for OUR noise:

```
per country:
  pass A (dense, exists):      e5 exact top-30                    → semantic/paraphrase/script
  pass B (lexical, NEW §4):    char-3-gram TF-IDF top-k on core   → typos/domains
  pass C (address, NEW §2+§7): address TF-IDF + (house,PIN) keys  → Evoavi + lookalikes
  pass D (cheap, NEW §6):      phonetic+city                      → spelling variants
  pass E (learned, LATER §10): fine-tuned encoder top-k           → twins/crowding
                        │ union + dedup, cap ~35/S1
                        ▼
              rank-fusion (RRF) or rank-then-truncate per source
                        ▼
              buckets (with per-pair provenance: WHICH passes fired
                       → becomes a matcher feature, per PLAN.md)
```

**Fusion rule.** Reciprocal Rank Fusion per candidate: `Σ 1/(60 + rank_pass)` over
passes that retrieved it; keep top-35. RRF needs no training and rewards
multi-pass agreement (a pair found 3 ways is almost certainly worth judging).
Provenance flags (`from_dense/from_tfidf/from_addrkey`) become matcher features —
the matcher then learns pass-trust per slice (e.g. address-pass pairs need stronger
name agreement in crowded France).

**Budget.** A:30 (exists) + B:5 + C:5 + D:2, dedup → ~32–35/S1. Matcher cost +~15%
for the pairs that matter. Every pass addition gated on test-bed ceiling lift.

---

## 15. Measurement protocol (how to judge any blocker here — non-negotiable)

1. **Full-index, never sample-only.** Build at least 50–100K realistic queries per
   country against the FULL country index before trusting a cap (sample indexes
   overstate recall — verified 0.974 vs 0.946 in this repo's history).
2. **Report per slice:** edge recall@k AND full-list recall (an S1 with 5/5 found
   differs from one with 4/5), ceiling macro-F0.5, per country + hard slices
   (native-script, empty-address, domain-name, crowded-name, France proxy).
3. **Leave-one-country-out** (train US → score India and reverse) as the France
   stand-in for any LEARNED blocker (§10/§12). Small drop (≤0.02) or it doesn't ship.
4. **Cost accounting:** candidates/S1 added, CPU/GPU minutes, RAM — a +0.002 ceiling
   for +10 candidates/S1 is a bad trade under F0.5 (matcher noise eats it).
5. **Ablate the union:** report each pass's marginal recall (pass alone, union minus
   pass). Kill passes with ~zero marginal gain — every extra candidate is matcher
   noise risk under a precision-heavy metric.

**Priority order for this repo:** §2+§7 address keys (hours, biggest measured gap) →
§4 TF-IDF (days, typo/domain gap) → §6 phonetic (minutes, include in the same batch) →
§10 SC-Block (weeks, learned upside) → §8/§9 only if exact search breaks → skip the rest.
