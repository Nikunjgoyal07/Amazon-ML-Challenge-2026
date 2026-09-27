# Architecture: how the solution works

**Task.** For every business in Source 1 (S1), find the records in Source 2 and Source 3 (S2/S3) that
are the same business. The score is F0.5 per S1, averaged over all S1:

- Precision counts twice as much as recall.
- An S1 with no true match scores 1 only if we predict nothing.

**Approach in one sentence.** Clean the records, use a multilingual embedding model to shortlist the
30 most similar S2/S3 records for every S1 (plus a few found by address, name and reverse searches),
let a LightGBM model judge each (S1, candidate) pair, and keep the likely pairs under the rule that
each S2/S3 record belongs to at most one S1.

```
 raw TSVs (train + test)
        │
        ▼
 ① 01_eda_preprocessing          clean names/addresses, split per source/split/country → processed/*.parquet
        │
        ▼
 ② 02_full_e5_buckets            multilingual-e5-small embeddings (both GPUs)
        │                         exact top-30 search per country + extra candidates (reverse search,
        │                         address key, name 3-grams) → buckets/<split>_<country>.parquet
        ▼
 ③ 03_full_lightgbm_submission   85 comparison features per (S1, candidate) pair
        │                         LightGBM trained on train buckets → probability per test pair
        │                         keep p ≥ t, one owner per S2/S3 → matching_results.tsv, candidate_pairs.tsv
        ▼
 ④ 04_rethreshold (optional)     re-write the matches with other per-country cutoffs, no recomputation
```

`02_e5_embeddings` and `03_lightgbm_matcher` are the same two stages on a 100K–500K sample. They are
used for experiments and are not needed for a submission.

---

## What the data looks like (measured)

| | S1 | S2 | S3 |
|---|---|---|---|
| Train | 2,206,821 (US 1,323,633 · India 883,188) | 5,034,616 | 5,285,603 |
| Test | 1,732,544 (India 809,986 · US 663,106 · **France 259,452**) | 4,887,273 | 5,082,316 |

- **Match counts:** an S1 has 3.46 matches on average (max 11); 5.6% have none (singletons).
- **One owner:** every S2/S3 record belongs to **at most one** S1; 26% belong to none (distractors).
- **Same country:** matches never cross countries, so everything is done per country.
- **France is test-only:** the model never sees French training data.
- **Noise:**
  - typos, abbreviations, legal-form changes, reordered address parts, dropped city/PIN
  - names replaced by a website or a random word
  - names written in Indian scripts (Hindi, Telugu, Malayalam, ...)
- **France is crowded:** its 259K S1 sit in only ~15 cities, and names are mostly "city + generic
  word" ("Bordeaux Club"). Lookalike businesses are common.

---

## ① Preprocessing: `01_eda_preprocessing.ipynb`

It runs EDA on the full data, then normalizes every record.

**Name steps:**
- Unicode clean-up and lower case, accent removal, junk wrappers removed (`***`, `[Services]`,
  `(ID: 12)`), honorifics removed and flagged.
- Legal form split off (EN/IN/FR: pvt ltd, llc, sarl, sas, ...).
- Website names reduced to their stem, a Latin transliteration (`name_lat`), and a phonetic code.

**Address steps:**
- House number and postcode extracted, PO-box/unit noise removed.
- Abbreviations expanded (`Rd` → road, `Bd` → boulevard, `R.` → rue).
- City and state detected where the format allows, plus flags for empty address, missing PIN and
  landmarks.

**Output:** one parquet file per split, source and country (`train_s2_India.parquet`, …). With
`FULL_RUN=true` it processes every row. Test S2/S3 are included because the submission needs them.

**Known weakness:** the accent removal also deletes the vowel signs of Indian scripts, so 01's
cleaned text is garbled for those records. Stages ② and ③ work around it with their own
transliteration (below).

---

## ② Candidate generation ("blocking"): `02_full_e5_buckets.ipynb`

**Goal:** for every S1, a short list (its *bucket*) that contains its true matches. Nothing outside the
bucket can ever be matched, so this stage sets the ceiling of the whole system.

**Text given to the model:**
- Latin-script records: `name | address` from 01's cleaned fields.
- Records with Indian script: the **original** text transliterated to Latin letters with `anyascii`
  (ISC license), e.g. "సన్ ఎనర్జీ లిమిటెడ్" → "sn enrji limited".
- Without this, 57% of India's Indian-script true matches were missing from the buckets. With it,
  India's ceiling rose from 0.939 to 0.963 on the test bed.
- Two fixes on top of anyascii: Malayalam "റ്റ" becomes "tt" (anyascii gives "rr": "limirrd" for
  "limited"), and the nasal dot becomes "n" before most consonants (anyascii always gives "m": "anmd"
  for "Anand"). Name similarity of Indian-script true matches to their S1 rose 71.9 → 75.1
  (Malayalam 66.1 → 72.5). The same function is used in 03.
- **Word map for Indian-script names:** most such names are English words written in an Indian
  script, so their transliterations are misspelled English ("smart solyusns praivet limited"). A map of
  ~370 words learned from the train true pairs sends them back to English ("smart solutions private
  limited"). Name similarity of Indian-script true pairs to their S1, measured on S1 not used for
  learning: Hindi 76 → 91, Tamil 63 → 88, Malayalam 73 → 92, the other scripts similar. The test-bed
  matcher went 0.9667 → 0.9675 (cross-country +0.003 to +0.005). 02 learns it (~1.5 min) and saves
  `translit_wordmap.json` with the buckets; 03 uses the same file.
- Indian-script records remain the weak spot of the search: on the test bed they were a third of
  India's true matches but two thirds of its misses. 02's train report shows this split at full size.

**Embeddings:** `intfloat/multilingual-e5-small` (MIT, 118M parameters, 384 dimensions), with the
`query: ` prefix:
- fp16 on both T4 GPUs, one process per GPU
- each unique text encoded once, texts sorted by length to reduce padding

**Search:** per country, exact nearest-neighbour search on the GPUs.
- Vectors are **mean-centered** (subtract the country's average S2/S3 vector, re-normalize). e5
  vectors share a large common direction; removing it raised recall (93.0% → 95.3% in a test).
- Each block of S1 queries is scored against all S2/S3 vectors in one fp16 matrix product. The best
  k + 32 are re-scored in fp32, and the top **30** are kept.
- **Why not FAISS IVF-PQ:** on India data it found 85.5% of true matches vs 95.3% for exact search,
  and its compression saved no memory here. Exact search on two T4s takes minutes per country.

**Extra candidate sources.** e5 compares name and address together, so it misses some matches.
Five other searches add candidates to each bucket:

| Source | A pair is added when | Catches |
|---|---|---|
| Reverse search | the S1 is among the 2 best S1 of the S2/S3 record (same e5 score, searched from the candidate's side) | matches pushed below rank 30 by lookalikes in crowded cities |
| Address key | same house number and first street word (keys shared by more than 20 candidates are skipped) | same address, different or garbled name |
| Name 3-grams | among the S1's 5 most similar names by character 3-grams (TF-IDF cosine, on the GPUs) | same name, different or empty address; typos |
| Number key (India) | the addresses share a number (house, plot, door, PIN) with another word of 3+ letters, e.g. `206 + pune` (keys shared by more than 10 candidates are skipped) | chopped-down addresses with no street word ("Door No 206, Pune, MH"), which the address key cannot use |
| Empty-address names | the name search over only the candidates with no address: each one's 3 closest S1 names, and each S1's 2 closest names among them | a candidate with a name only, crowded out of the other searches by lookalikes that have an address |

On the test bed, the address key and name 3-grams raised India's recall from 91.4% to 93.3% (ceiling
0.967 → 0.974) for 3.7 extra pairs per S1, and the US's from 98.7% to 99.1%. The reverse search was
not measured there (it needs the embeddings). Per extra pair, they found 4–10× more missed matches
than a bigger e5 top-k. 02's train report measures every source at full size. Each bucket row
records which searches found it, and 03 uses that as features. `EXTRA_SOURCES=""` turns them off.

The number key and the empty-address search were measured on the full train data (submission 2's buckets,
which had the first three sources). India's buckets missed 167.7K of 3.06M true pairs (recall 0.9452, 37.3
pairs per S1):
- 23% of those misses had a candidate with **no address** (38.0K; 70% of the US's 31.2K misses). Their names
  are close to the S1's (median similarity 97), but e5 compares "name | address" with a bare name and the name
  search is crowded out by lookalikes. The empty-address search (S1 top-2 + candidate top-2) found 38.6% of them
  for at most 1.8 new pairs per S1 (US: 21.0% for 1.4); the candidate side alone (top-3) 24.9% for 0.3.
- Of the misses with an address, 70% had an Indian-script name (the word map rescues about half of those),
  and the candidate's address was usually cut down to "number, city, state": 83% had a number but only 28%
  a street word, so the address key could not key them. The number key (cap 10) found 22.6K misses for 3.9
  new pairs per S1 (recall 0.9452 -> 0.9526); cap 5: 15.0K for 0.9 pairs.
- For comparison, a bigger e5 top-k (30 -> 50) costs 20 pairs per S1.

**Output:** `buckets/<split>_<country>.parquet`, one row per (S1, candidate) with rank, two
similarity values and which searches found it. The embeddings themselves are not saved: all 24M records would take ~18.6 GB,
close to Kaggle's 20 GB output limit.

**How good is it (train, top 30, full size):** the ceiling (best possible score if the judge were
perfect) is 0.994 for the US and 0.946 for India before the transliteration fix. It is ~0.97 for
India with the fix, measured on the test bed.

---

## ③ Matching model: `03_full_lightgbm_submission.ipynb`

**Training data:**
- 200,000 random train S1 per country, with their 30 e5 candidates and the extra candidates each
  (~13M pairs), labeled from `train_ground_truth.tsv`.
- The buckets are the full-size ones, so every S1 competes with all its real lookalikes, exactly as
  on test. (Training on a small sample gave misleadingly high scores: 0.974 vs 0.946 in reality.)

**Features (85 per pair).** Country is **not** a feature, so the same model applies to France.

| Group | Examples | Why |
|---|---|---|
| Embedding | rank, similarity, gap to the S1's best candidate | overall resemblance |
| Competition between S1s | how many S1 buckets hold this candidate; is this S1 its best S1; margin to its next-best S1 | one owner: a record usually belongs to the S1 it resembles most (the strongest feature group) |
| Name | edit-distance, token-set/sort, Jaro-Winkler, transliterated, compact (spaces/dots removed, catches websites), phonetic, legal form | spelling noise, reordering, websites |
| Address | cleaned and raw address ratios, shared street tokens, shared numbers, postcode, house number, city, state | the address separates lookalikes with the same name |
| **Street** | street name (the address part with the house number, minus numbers and street words); house number | "45 Rue du Port Durand" ≠ "45 Rue du Pressor" |
| **Word rarity** | name/address overlap weighted by how rare each word is among the country's S1 (IDF); the rarest unmatched word on each side | sharing "club" means little, sharing "diaspora" a lot |
| **Typo-tolerant word rarity** | the word-rarity features again, counting words that differ by a typo (1 letter for 4–5 letter words, 2 for longer) as shared | "Westgrove" vs "Wcstgrove" should not count as a missing rare word; kept next to the exact ones (test bed 0.9667 → 0.9675; tolerant alone was worse, 0.9659) |
| **Candidate vs candidate** | how far this candidate is behind the best candidate of the same S1 on address, street, name and word rarity, and its rank; how many near-identical names/addresses the bucket has; how many candidates have an equally close name and a clearly better address | "Bordeaux Club, 45 Rue Judaïque" loses to the candidate at the S1's own street; test bed 0.9667 → 0.9668, cross-country check +0.003 / +0.0004 (the France stand-in) |
| **Name frequency** | how many S1 per 100K in the country share the name | "Bordeaux Club" appears hundreds of times |
| Flags | candidate from S2/S3, native script, website name, empty address, honorific, landmark | tells the model which comparisons to trust |
| **Found by** | rank in the e5 search; rank of this S1 among the candidate's best S1 (reverse search); same house number + street word; rank among the S1's most similar names; same number + address word; for candidates with no address, the rank in the name search over them from each side | how much to trust a candidate from each search |

- Rarity and frequency are computed per country from unlabeled S1 records, so they adapt to France.
- Indian-script names and addresses are transliterated with `anyascii` for all comparisons.

**Model:**
- LightGBM binary classifier (MIT), 63 leaves, learning rate 0.1, early stopping.
- Uses the GPU if the installed LightGBM supports it, otherwise the CPU. Larger trees and slower
  learning rates made no measurable difference.

**Validation:**
- 3-fold cross-validation grouped by S1: every pair is scored by a model that never saw its S1.
- Macro F0.5 is computed exactly as the competition does. True matches missing from the buckets
  count as misses, so the score is realistic.
- A "train on one country, score the other" check stands in for France.

---

## ④ Decision layer

1. Every test pair gets a probability p.
2. **One owner:** each S2/S3 record is kept only for the S1 that gives it the highest p.
3. **Cutoff:** pairs with p ≥ t are kept. t is tuned on cross-validation, ~0.65–0.70; the score is
   flat around it. `COUNTRY_THRESHOLDS` can set a stricter t for one country (e.g. France).
4. An S1 whose pairs are all below t gets an empty list, the prediction for "no match".

**Alternatives tried and dropped:**
- per-S1 expected-F0.5 selection: +0.0002
- label-free France cutoff by matching prediction volume: helped once, hurt once
- self-training on pseudo-labels: −0.005 to −0.01

---

## Outputs and validation

- `matching_results.tsv`: every test S1 exactly once, tab-separated, comma-separated IDs, empty
  when no match. **This is the file the leaderboard scores.**
- `candidate_pairs.tsv`: the exact pairs the model scored (30 e5 candidates per S1 plus the extra
  ones), so the matches are always a subset of them.
- 03 checks both files against the submission rules and runs `validate_submission.py` if it is
  attached. Run the validator locally too (see `SUBMISSION_STEPS.md`).

---

## Scores so far

| What | India | US | France | Overall |
|---|---|---|---|---|
| First submission (public leaderboard) | — | — | — | **0.935** |
| Same model, full-size cross-validation | 0.921 | 0.970 | ~0.88 (derived from the leaderboard) | |
| Test bed, before the fixes | 0.914 | 0.976 | | 0.943 |
| Test bed, with transliteration + 57 features + 30 candidates | 0.953 | 0.983 | | 0.967 |

The **test bed** is a local copy of whole regions of the real data: Arizona + Utah, Kerala +
Telangana, and Pays de la Loire for France. Every business there faces its real competition, and
its "before" scores match the full Kaggle run closely (0.943 vs 0.946), so improvements measured
there carry over.

---

## Rules compliance

- **No external data or APIs:** everything comes from the provided files. Word rarity and name
  frequency are computed from the test records' own text, without labels.
- **Models:** `multilingual-e5-small` (MIT, 118M parameters) and LightGBM (MIT). Both are within the
  MIT/Apache and ≤ 8B-parameter rule.
- **Libraries:** sentence-transformers (Apache 2.0), rapidfuzz (MIT), anyascii (ISC), polars,
  pandas, pyarrow, PyTorch.
- **No train/test leakage:** test IDs and records do not overlap with train, and the model learns
  only from the train ground truth. The first submission's files were checked for this.

## Known limitations

- **France** is the weakest country (~0.88). The model never sees French data, and on unseen
  countries it loses 0.04–0.06. The best remaining tool is testing France cutoffs on the leaderboard
  with notebook 04.
- **India's search** still misses some true matches, e.g. empty addresses and names replaced by a
  random word. A second, keyword-based candidate source ("same house number + street word") would
  catch 18–28% of the misses for 1–2 extra candidates per S1. It is not implemented.
