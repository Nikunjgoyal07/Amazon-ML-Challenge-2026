# Solution write-up: business entity resolution (Amazon ML Challenge 2026)

This document explains the whole solution: what the problem is, how the pipeline works stage by stage,
and how it got there, iteration by iteration, with the measurements behind each decision. It also
records what did not work, and why.

Companion documents (shorter, more operational):
- [pipeline/](pipeline/): how each notebook's code works, cell by cell, with every setting
  ([01](pipeline/01_PREPROCESSING.md), [02](pipeline/02_CANDIDATES.md), [03](pipeline/03_MATCHER.md),
  [04](pipeline/04_RETHRESHOLD.md))
- [RESULTS_EXPLAINED.md](RESULTS_EXPLAINED.md): how to read the notebook outputs
- [SUBMISSION_STEPS.md](SUBMISSION_STEPS.md): how to run it
- the README of each submission folder (`submission-1/` … `submission-5/`): what each submission ran
- [DATA_AND_OUTPUTS.md](DATA_AND_OUTPUTS.md): files and columns
- [experiments/](experiments/): work after the final submission (fine-tuned and own embedding models, ensemble)

---

## Contents

1. [Summary](#1-summary)
2. [The problem and the metric](#2-the-problem-and-the-metric)
3. [What the data told us](#3-what-the-data-told-us)
4. [The pipeline at a glance](#4-the-pipeline-at-a-glance)
5. [Stage 1: preprocessing](#5-stage-1-preprocessing-01_eda_preprocessingipynb)
6. [Stage 2: candidate generation](#6-stage-2-candidate-generation-02_full_e5_bucketsipynb)
7. [Stage 3: the matching model](#7-stage-3-the-matching-model-03_full_lightgbm_submissionipynb)
8. [Stage 4: the decision layer](#8-stage-4-the-decision-layer)
9. [How we measured](#9-how-we-measured)
10. [Iteration history](#10-iteration-history)
11. [Final results and error analysis](#11-final-results-and-error-analysis)
12. [What did not work, or was not worth it](#12-what-did-not-work-or-was-not-worth-it)
13. [Engineering at full scale](#13-engineering-at-full-scale)
14. [Rules compliance](#14-rules-compliance)
15. [Lessons learned and next steps](#15-lessons-learned-and-next-steps)
16. [Reproducing the submission](#16-reproducing-the-submission)

---

## 1. Summary

**Approach.** A two-stage *retrieve, then judge* pipeline:

1. **Retrieve.** For every Source 1 (S1) business, a multilingual sentence-embedding model
   (`intfloat/multilingual-e5-small`) finds its 30 most similar Source 2/3 (S2/S3) records by exact
   nearest-neighbour search on two GPUs. Five cheap, targeted searches add the matches the embedding
   misses: a reverse search, an address key, name character 3-grams, a number key and an
   empty-address name search. This shortlist is the S1's **bucket**.
2. **Judge.** A LightGBM classifier scores every (S1, candidate) pair from 85 comparison features:
   name and address similarity, word rarity, how the candidate compares with the S1's other
   candidates, how it compares with the other S1s that want it, and which searches found it.
3. **Decide.** Each S2/S3 record is given only to the S1 that scores it highest (the data's
   one-owner rule), and pairs above a tuned probability cutoff are kept. France, which never appears
   in training, gets its own stricter cutoff.

**Results.**

| | Macro F0.5 |
|---|---|
| First submission (public leaderboard) | 0.935 |
| Submission 2 (public leaderboard) | 0.960 |
| **Final pipeline (public leaderboard)** | **0.967** |
| Final pipeline, 3-fold cross-validation on India + US | 0.9796 (India 0.9766, US 0.9826) |
| Ceiling: a perfect judge on our buckets (India + US) | 0.9947 |

**Where the remaining points are.** France scores about 0.90 by our estimate (it has no labels;
section 11.2 derives the number), against about 0.98 for the two countries the model was trained on.
France is 15% of the test S1 and accounts for most of the gap between 0.967 and 0.98. On India and
the US, the judge, not the search, is now the bottleneck: the buckets allow 0.9947, the model
reaches 0.9796, and doubling the model's capacity did not move it (section 10.12).

**Key ideas that moved the score:**
- measuring at full scale (samples inflate scores badly: 0.974 on a sample vs 0.946 in reality);
- transliterating Indian-script records from their *original* text, plus a word map learned from
  the training pairs that turns misspelled transliterations back into English;
- small targeted candidate searches instead of a bigger embedding top-k;
- "competition" features: whether this S1 is the candidate's best S1, and by what margin, which is
  by far the strongest signal the model has.

---

## 2. The problem and the metric

**Task.** Each test S1 business may appear, noisily, in two other sources (S2, S3). For every S1 we
output the list of S2/S3 records that are the same business, or an empty list.

**Metric: macro F0.5 per S1.**
- For each S1, precision and recall of our list against the true list are combined as F0.5, which
  weights precision twice as heavily as recall. A wrong match hurts more than a missed one.
- An S1 with no true match (a *singleton*) scores 1 if we predict nothing and 0 if we predict
  anything.
- The final score is the plain average over all S1. Every business counts the same, whether it has
  one match or ten.

Worked example (from the problem statement): 2 true matches, we predict 3 of which 2 are right.
Precision 0.67, recall 1.0, F0.5 = 0.714.

**What the metric implies for design:**
- Precision matters most, so the cutoff should lean strict, and any rule that removes wrong matches
  cheaply (the one-owner rule) is worth having.
- Singletons are all-or-nothing, so the model must be confident enough to predict *nothing*.
- Only the S1-level average matters, so a small, noisy slice (France) can move the score a lot.

**Constraints.** No external data or APIs; final models under MIT/Apache licenses and at most 8B
parameters; compute on Kaggle (two T4 GPUs, about 30 GB RAM); a few submissions per day over
three days.

---

## 3. What the data told us

Measured on the full files during EDA (`01_eda_preprocessing.ipynb`):

| | S1 | S2 | S3 |
|---|---|---|---|
| Train | 2,206,821 (US 1,323,633; India 883,188) | 5,034,616 | 5,285,603 |
| Test | 1,732,544 (India 809,986; US 663,106; **France 259,452**) | 4,887,273 | 5,082,316 |

**Structure of the matches**
- An S1 has 3.46 true matches on average (maximum 11). 5.6% of train S1 have none.
- True matches split 48.4% S2 / 51.6% S3.
- **Every S2/S3 record belongs to at most one S1.** 26% belong to none; these *distractors* are
  the main source of false positives.
- **No match crosses countries** (0 of 7.64M checked), so everything can be done per country.
- **France appears only in test.** The model can never see a labeled French example.

**Noise we observed (it looks synthetic and systematic):**
- junk wrappers and prefixes: `***`, `>>`, `[Services]`, `(ID: 1072)`, `<NULL>`
- typos (`Wcstgrove`), injected accents (`Prívate`), shuffled and duplicated words
- legal forms added, dropped or expanded (Pvt/Private, Ltd/Limited, SARL, SAS)
- names replaced by a website (`eelegal.com`) or by an unrelated word, so only the address matches
- names and states written in Indian scripts (Devanagari, Tamil, Telugu, Kannada, Malayalam,
  Gujarati, Bengali, Odia)
- addresses reordered, abbreviated (`Rd`, `R.`, `Bd`), cut down to "number, city, state", with
  injected `H.no`, `PO Box`, `Unit`, and states swapped between code and name (`MH`/Maharashtra)
- missing addresses: about 3% of S2/S3 records (169K and 176K in train S2 and S3)

**France is crowded.** Its 259K test S1 sit in about 15 cities, and many names are "city + generic
word" ("Bordeaux Club", "Nantes Federation SASU"). Lookalike businesses are everywhere, which is
exactly what a precision-heavy metric punishes.

---

## 4. The pipeline at a glance

```
 raw TSVs (train + test, 6 files, ~24M records)
        │
        ▼
 ① 01_eda_preprocessing.ipynb      EDA; clean names and addresses; extract legal form, house number,
        │                           postcode, city, state, flags → processed/<split>_<source>_<country>.parquet
        ▼
 ② 02_full_e5_buckets.ipynb        e5-small embeddings on both GPUs; exact top-30 search per country;
        │                           + reverse / address key / name 3-grams / number key / empty-address search
        │                           → buckets/<split>_<country>.parquet (one row per S1-candidate pair)
        ▼
 ③ 03_full_lightgbm_submission.ipynb   85 features per pair; LightGBM trained on train buckets
        │                           (3-fold CV grouped by S1, tuned cutoff); probability for every test pair;
        │                           one owner per S2/S3; keep p ≥ t → matching_results.tsv, candidate_pairs.tsv
        ▼
 ④ 04_rethreshold.ipynb (optional) re-write the matches with other per-country cutoffs from the saved
                                    probabilities, in minutes, without re-running ② or ③
```

The four notebooks are in `submission-4/`. Two sample-scale notebooks next to them run stages ② and ③ on a
sample (`02_e5_embeddings.ipynb`, `03_lightgbm_matcher.ipynb`). They were used to test ideas quickly and are not
part of the submission.

---

## 5. Stage 1: preprocessing (`01_eda_preprocessing.ipynb`)

### 5.1 What it produces

One parquet file per split, source and country (`train_s2_India.parquet`, …). With `FULL_RUN=true`
every row is processed in chunks of 200K rows. Test S2/S3 are included because the submission needs
them. Each record keeps its original name and address and gains these columns:

| Column | How it is made |
|---|---|
| `name_norm` | Unicode NFKC, accents stripped, lower case, junk wrappers removed, honorific removed |
| `core_name`, `legal_form` | the name split into its legal-form words (pvt, ltd, llc, sarl, sas, sasu, eurl, sci, ei, cie, …) and the rest |
| `domain_stem` | `projectsmanipalindia.com` → `projectsmanipalindia` |
| `name_lat`, `phonetic_code` | Latin transliteration of the core name; Soundex of its first word |
| `addr_norm`, `street_tokens` | cleaned address; PO box / unit noise removed; `Rd`→road, `Bd`→boulevard, `R.`→rue |
| `house_no`, `zip_pin_cp` | house number (marker such as `H.no`/`plot`/`#`, else the first bare 1–4 digit number); 5- or 6-digit postcode |
| `city`, `state_code` | from the last address parts: US state codes and names, Indian states (including native-script names), French regions |
| flags | `has_honorific`, `empty_addr`, `missing_pin`, `missing_city`, `landmark_flag` |

Every normalizer has unit assertions in the notebook (e.g. `"H.no 780, R. Voltaire, 75011 Paris"`
must give house number 780 and postcode 75011).

### 5.2 Sampling for experiments

Experiments ran on samples, and how a sample is drawn changes the scores (section 9.1). The sample
mode (`SAMPLE_MODE=linked`, default) draws random train S1 first, then gives each S2/S3 file all true
matches of those S1 plus random other records, so the sample contains both real pairs and real
distractors. In the final iteration the random draws became **stratified**: every draw guarantees a
small floor to each stratum (native-script name, website-as-name, honorific prefix, and for train S1
the match count: none / 1–2 / 3–5 / 6+), so rare cases still appear in small samples.

### 5.3 The known weakness, and the workaround

The accent-stripping step (NFKD, then drop combining marks) also deletes the combining marks of
Indian scripts (the virama that joins consonants, and several vowel signs), so 01's cleaned name and
address for Indian-script records are garbled: `प्राइवेट` ("private") becomes `पराइवेट`. Rather than rebuild 01,
stages ② and ③ transliterate those records from their **original** text themselves (section 6.1).
This matters later: it is why changing 01's transliteration barely affects the pipeline (section 10.10).

### 5.4 France-specific fixes (final iteration)

Because France has no training data, we audited, field by field, whether 01 extracts anything at all
from French records. Measured on 5,000 real test rows per country:

| Field | Before | After | Change |
|---|---|---|---|
| `state_code` and `city` | France **0%** (US 86%, India 19%) | France **100%** | added the 13 French regions and 5 overseas regions; `city` is only extracted once a state is found |
| `legal_form` | SASU (10,721 of 259,452 France names), EI (4,182) and Cie (877) not recognized | recognized | none of the three ends a single US or India name out of 2.2M, so no false positives |
| `landmark_flag` | English words only | + banque, hôpital, gare, église, mosquée, centre commercial | small (≤ 461 addresses per word) |
| `house_no` | France 0.0%, US 0.2%, India 5.4% | 99.4% / 90% / 47% | fallback to the first bare number |

The house-number fix turned out to be redundant for the model: stage ③ already computes its own
house number from the raw address (`house_number_same`), which was working all along. The city and
state fix is not redundant: `city_same` and `state_same` read 01's columns directly, so before the
fix they were missing (NaN) for every French pair. Their effect cannot be measured offline (France has
no labels).

---

## 6. Stage 2: candidate generation (`02_full_e5_buckets.ipynb`)

**Goal:** for every S1, a short list (its bucket) that contains its true matches. Nothing outside the
bucket can ever be matched, so this stage sets the ceiling of the whole system.

### 6.1 The text the embedding model sees

- **Latin-script records:** `name | address` from 01's cleaned fields.
- **Indian-script records:** the *original* name and address transliterated to Latin with `anyascii`
  (ISC license), with two corrections measured on true pairs:
  - Malayalam `റ്റ` is read `tt` (anyascii gives `rr`: "limirrd" instead of "limittd");
  - the nasal dot (anusvara) before a non-labial consonant is read `n` (anyascii gives `m`:
    "anmd" instead of "annd" for आनंद).
  Name similarity of Indian-script true matches to their S1 rose 71.9 → 75.1 (Malayalam 66.1 → 72.5).
- **Word map.** Most Indian-script names are English words written in an Indian script, so their
  transliterations are misspelled English ("smart solyusns praivet limited"). A map of about 370 words
  is learned from the train true pairs: each transliterated word is aligned to the most similar word of
  its true S1's name, and kept when the same English word wins often enough. It sends "solyusns" to
  "solutions" and "praivet" to "private". Name similarity of Indian-script true pairs, measured on S1
  not used for learning: Hindi 76 → 91, Tamil 63 → 88, Malayalam 73 → 92. The matcher on the test bed
  went 0.9667 → 0.9675. The map is saved with the buckets and reused by stage ③.

Without the transliteration, 57% of India's Indian-script true matches were missing from the buckets;
with it, India's ceiling on the test bed rose from 0.939 to 0.963.

### 6.2 Embeddings and search

- `intfloat/multilingual-e5-small` (MIT, 118M parameters, 384 dimensions), `query: ` prefix, fp16,
  one encoding process per T4 GPU. Each unique text is encoded once; texts are sorted by length to
  reduce padding.
- **Mean-centering:** e5 vectors share a large common direction. Subtracting the country's average
  S2/S3 vector and re-normalizing raised recall from 93.0% to 95.3% in a test.
- **Exact search:** each block of S1 queries is scored against all of the country's S2/S3 vectors in
  one fp16 matrix product; the best k+32 are re-scored in fp32 and the top 30 kept. The work is split
  across both GPUs.
- **Why not an approximate index:** FAISS IVF-PQ found 85.5% of India's true matches against 95.3% for
  exact search, and its compression saved no memory here. Exact search on two T4s takes minutes per
  country.

### 6.3 The five extra searches

e5 compares name and address together as one text, so it misses matches where one half is broken.
Each extra search targets a failure pattern we measured:

| Search | A pair is added when | Catches |
|---|---|---|
| Reverse | the S1 is among the 2 best S1 of the candidate, searching from the candidate's side | true matches pushed below rank 30 by lookalikes in crowded cities |
| Address key | same house number and first street word (keys shared by more than 20 candidates skipped) | same address, garbled or replaced name |
| Name 3-grams | among the S1's 5 closest names by character 3-gram TF-IDF (on GPU) | same name, different or empty address; typos |
| Number key (India only) | a shared number (house, plot, door, PIN) plus another shared word of 3+ letters, e.g. `206 + pune` (keys shared by more than 10 candidates skipped) | addresses cut down to "Door No 206, Pune, MH", which have no street word for the address key |
| Empty-address names | name search over only the candidates with no address: each one's 3 closest S1 names, and each S1's 2 closest names among them | candidates with only a name, crowded out of the other searches by lookalikes that have an address |

Each bucket row records which searches found it (`e5_rank`, `rev_rank`, `addr_key`, `name_rank`,
`num_key`, `empty_rank`, `empty_rev_rank`), and stage ③ uses these as features.

The number key and the empty-address search came from a miss analysis on the full train buckets of
submission 2 (section 10.8).

### 6.4 How good the buckets are (full train data)

Share of true matches that reach the bucket, and the ceiling (score of a perfect judge on the bucket):

| Candidates | Pairs per S1 (India / US) | True matches in bucket (India / US) | India, empty-address candidates | Ceiling (India / US) |
|---|---|---|---|---|
| e5 top-20 | 20 / 20 | 0.9397 / 0.9805 | 0.4418 | 0.9798 / 0.9942 |
| e5 top-30 | 30 / 30 | 0.9471 / 0.9833 | 0.4892 | 0.9824 / 0.9950 |
| e5 top-50 | 50 / 50 | 0.9549 / 0.9862 | 0.5448 | 0.9851 / 0.9959 |
| top-30 + reverse | 32.5 / 31.8 | 0.9584 / 0.9873 | 0.5605 | 0.9864 / 0.9962 |
| top-30 + address key | 31.1 / 31.2 | 0.9503 / 0.9882 | 0.4892 | 0.9835 / 0.9965 |
| top-30 + name 3-grams | 33.5 / 32.2 | 0.9542 / 0.9837 | 0.6228 | 0.9849 / 0.9951 |
| top-30 + number key | 33.9 / – | 0.9544 / – | 0.4892 | 0.9850 / – |
| top-30 + empty-address search | 31.8 / 31.3 | 0.9564 / 0.9846 | 0.7275 | 0.9854 / 0.9954 |
| top-30 + empty-address search, other direction | 30.3 / 30.2 | 0.9561 / 0.9849 | 0.7206 | 0.9853 / 0.9955 |
| **top-30 + all five (used)** | **42.8 / 36.6** | **0.9745 / 0.9919** | **0.8027** | **0.9917 / 0.9976** |

The five searches together add about 13 pairs per S1 in India and raise its ceiling from 0.9824 to
0.9917. Going from e5 top-30 to top-50 costs 20 pairs per S1 and raises it only to 0.9851. Targeted
searches find 4–10 times more missed matches per extra pair than a bigger top-k.

---

## 7. Stage 3: the matching model (`03_full_lightgbm_submission.ipynb`)

### 7.1 Training data

- 200,000 random train S1 per country (seeded), each with its full bucket: the 30 e5 candidates plus
  the extra ones. About 15.9M labeled pairs, of which the true pairs come from
  `train_ground_truth.tsv`.
- The buckets are the **full-size** ones, so every training S1 competes with all of its real
  lookalikes, exactly as on test. The features that compare a candidate across S1s (below) are
  computed over *every* S1 of the country before the 200K subset is taken.

### 7.2 Features (85 per pair)

Country is deliberately **not** a feature, so the same model applies to France.

| Group | Features | Why |
|---|---|---|
| Embedding (7) | rank in the bucket, e5 score, cosine, the S1's best cosine, gap to the best and next candidate, the S1's top-5 cosine | overall resemblance, and how clear-cut this S1's bucket is |
| Competition between S1s (3) | how many S1 buckets contain this candidate; this S1's rank among them; margin to the candidate's next-best S1 (`cand_margin`) | the one-owner rule: a record usually belongs to the S1 it resembles most. The strongest group by far |
| Name (15) | Levenshtein ratio, token-set / token-sort / partial ratios, Jaro-Winkler on the core name, ratios on the transliteration, on the raw name, and on a compact form (spaces and dots removed, catches websites), phonetic code, legal form, lengths | spelling noise, word reordering, websites, legal-form changes |
| Address (13) | ratios on the cleaned and raw address, shared street tokens, shared numbers, postcode, house number, city, state, length | the address separates businesses that share a name |
| Street (3) | the street name (the address part holding the house number, minus numbers and street words) and house number, compared | "45 Rue du Port Durand" is not "45 Rue du Pressor" |
| Word rarity (8) | name and address overlap weighted by how rare each word is among the country's S1 (IDF); the rarest unmatched word on each side | sharing "club" means little, sharing "diaspora" a lot |
| Typo-tolerant rarity (8) | the same, counting words one or two letters apart as shared | "Westgrove" vs "Wcstgrove" should not count as a missing rare word. Kept next to the exact versions: exact + tolerant scored 0.9675 on the test bed, exact alone 0.9667, tolerant alone 0.9659 |
| Candidate vs candidate (13) | how far this candidate is behind the best candidate of the same S1 on address, street, name and rarity, and its rank on each; how many near-identical names and addresses the bucket holds; how many candidates have an equally close name and a clearly better address | "Bordeaux Club, 45 Rue Judaïque" should lose to the candidate on the S1's own street |
| Name frequency (2) | how many S1 per 100K in the country share this name | "Bordeaux Club" appears hundreds of times; a match on it is weak evidence |
| Flags (6) | candidate from S3, native script, website name, empty address, honorific, landmark | tells the model which comparisons to trust |
| Found by (7) | e5 rank, reverse rank, address key, name-search rank, number key, empty-address ranks from each side | how much to trust a candidate from each search |

Rarity and name frequency are computed per country from the S1 records themselves, without labels, so
on test they adapt to France. Indian-script names and addresses are replaced by their transliteration
plus word map before any comparison.

The feature set grew in steps, and each step can still be selected with `FEATURE_SET`: `base` (44),
`submission2` (61: + rarity, street, frequency, first four found-by), `all` (85: + typo-tolerant
rarity, candidate vs candidate, last three found-by).

### 7.3 Model

- LightGBM binary classifier (MIT license): 63 leaves, learning rate 0.1, minimum 100 rows per leaf,
  80% feature and row subsampling, L2 1.0.
- Uses the GPU build when the installed LightGBM supports it, otherwise the CPU.
- Early stopping on each cross-validation fold; the final model is trained on all pairs for
  1.1 × the mean best iteration.

**Most important features (final model, share of total gain):** `cand_margin` 48.6%, `rev_rank`
25.5%, `numbers_jaccard` 5.3%, `numbers_cover` 1.5%, `name_ratio` 1.4%, `house_number_same` 1.4%,
`cand_rank` 1.4%, `name_token_sort` 1.3%. The model leans first on "is this S1 the candidate's best
fit, and by how much", then on shared numbers, then on names.

---

## 8. Stage 4: the decision layer

1. Every test pair gets a probability p.
2. **One owner:** each S2/S3 record is kept only for the S1 that gives it the highest p.
3. **Cutoff:** pairs with p ≥ t are kept. t is tuned on the cross-validation predictions (about 0.69).
   The score is flat around the optimum: every cutoff from 0.60 to 0.75 is within 0.0004 of the best.
4. An S1 with no pair above the cutoff gets an empty list, which is the correct answer for singletons.
5. `COUNTRY_THRESHOLDS` sets a different cutoff for one country. The final submission uses
   **t = 0.9 for France**, because France matched more S1, with more matches each, than the trained
   countries do at the same cutoff (section 11.2).

`04_rethreshold.ipynb` rewrites `matching_results.tsv` from the saved test probabilities with any
per-country cutoffs, in minutes. This is the tool for tuning France on the leaderboard, one change per
submission so the score difference can be attributed.

`candidate_pairs.tsv` lists exactly the pairs the model scored, in rank order, so every match is also a
candidate, as the rules require.

---

## 9. How we measured

Getting the measurement right was the single most important lesson of the project.

### 9.1 Samples lie

On a random sample, each S1 competes with far fewer lookalikes than in the real data, and many of its
true matches may not be sampled. Both make the task look easier. Our first matcher scored **0.974** in
cross-validation on a 500K sample and **0.946** at full size. Every sample result after that was read
as a *relative* comparison between two variants, never as an estimate of the leaderboard.

### 9.2 The test bed

To iterate fast without that bias, we built a local copy of whole regions of the real data: Arizona +
Utah (US), Kerala + Telangana (India) and Pays de la Loire (France). Every business there faces its real
local competition. Its "before" score matched the full Kaggle run closely (0.943 vs 0.946), so
improvements measured there carried over.

### 9.3 Full-size cross-validation

The final numbers come from 3-fold cross-validation **grouped by S1** on the full-size buckets: every
pair is scored by a model that never saw its S1. Macro F0.5 is computed exactly as the competition
does (a vectorized scorer checked against a reference implementation), and true matches missing from
the buckets count as misses, so the score includes the search step's losses.

### 9.4 A stand-in for France

France has no labels, so its score can never be measured offline. Two proxies:
- **Cross-country transfer:** train on one country, score the other, as if it were unseen.
- **Prediction volume:** compare France's share of S1 with a predicted match, and matches per matched
  S1, with the same numbers for the trained countries. Much higher values mean false matches.

---

## 10. Iteration history

Each step lists what changed, why, and what it did to the numbers.

### 10.1 Plan and EDA

We considered six families of approaches: rules, blocking + gradient boosting, dense retrieval, a
cross-encoder re-ranker, graph clustering, and an LLM judge. The plan was blocking with dense retrieval
plus LightGBM as the core, with a cross-encoder and graph post-processing as later layers. The LLM
judge was ruled out early: far too slow for tens of millions of pairs on T4s. EDA established the
facts in section 3, most importantly the one-owner rule, no cross-country matches, and France being
test-only.

### 10.2 Baseline: embeddings and a cosine rule

multilingual-e5-small buckets and a rule "keep candidates with cosine ≥ t, one owner per record" scored
0.913 on a 100K sample (India 0.871, US 0.955). It is fooled by lookalikes and by Indian-script names,
which set the direction for everything after.

### 10.3 LightGBM matcher (submission 1: 0.935)

A LightGBM judge with 44 features over the e5 top-20 scored 0.974 in cross-validation on a 500K
sample, against 0.935 for the cosine rule on the same sample. The leaderboard gave **0.935**.
Full-size cross-validation of the same model explained the gap: 0.946 overall (India 0.921, US 0.970),
and France about 0.88 (derived from the leaderboard). This is where the sample-inflation lesson
(section 9.1) and the test bed came from.

### 10.4 Transliteration from the original text

The biggest single retrieval problem: 57% of India's Indian-script true matches never reached the
bucket, because 01's cleaned text for those records had lost its vowel signs. Transliterating the
original text with anyascii raised India's ceiling on the test bed from 0.939 to 0.963. The two
anyascii corrections (Malayalam `റ്റ`, nasal dot) added a few similarity points on top.

### 10.5 More candidates and more features (test bed 0.943 → 0.967)

- e5 top-30 instead of top-20: tested at 20 and 30, and 30 won.
- 13 new features: word rarity, street name and house number, name frequency.
Together with the transliteration, the test bed went from 0.943 to 0.967 (India 0.914 → 0.953,
US 0.976 → 0.983).

### 10.6 Extra searches (submission 2: 0.960)

- The first three extra searches (reverse, address key, name 3-grams). On the test bed the address
  key and name 3-grams raised India's recall from 91.4% to 93.3% for 3.7 extra pairs per S1, and the
  US's from 98.7% to 99.1%.
- Found-by features for the new searches (61 features in all).

Submission 2 scored **0.960** on the leaderboard. Its full run trained on 14.5M pairs and died with an
out-of-memory error during test prediction; a predict-only notebook that loads records one group of S1
at a time finished it (section 13).

### 10.7 Word map, typo-tolerant rarity, candidate-vs-candidate features

- The word map for Indian-script names (section 6.1): name similarity of Indian-script true pairs
  rose by 15–25 points depending on the script; test bed 0.9667 → 0.9675.
- Typo-tolerant word rarity: test bed 0.9667 → 0.9675 when kept next to the exact versions.
- Candidate-vs-candidate features, aimed at France's lookalikes: test bed 0.9667 → 0.9668, and +0.003
  in the cross-country check (the France stand-in).

### 10.8 Miss analysis at full size: number key and empty-address search

On submission 2's full train buckets, India missed 167.7K of 3.06M true pairs (recall 0.9452 at 37.3
pairs per S1). Breaking the misses down:
- **23% had a candidate with no address** (70% of the US's misses). Their names are close to the S1's
  (median similarity 97), but e5 compares a bare name with "name | address", and the name search is
  crowded out by lookalikes that have an address. An empty-address search from both sides found 38.6%
  of these for at most 1.8 new pairs per S1 (US: 21% for 1.4).
- **Of the misses with an address,** 70% had an Indian-script name (the word map rescues about half),
  and the candidate's address was usually cut down to "number, city, state": 83% had a number but only
  28% a street word, so the address key could not use them. A number key (number + another word,
  cap 10) found 22.6K of the misses for 3.9 new pairs per S1 (recall 0.9452 → 0.9526).

Both searches were added. With all five, India's ceiling on the full train data is 0.9917 (table in
section 6.4), and India's empty-address slice went from 0.49 to 0.80.

### 10.9 Engineering for full scale

Several fixes were needed before the full pipeline could run reliably on Kaggle; they are described in
section 13: an out-of-memory crash in the sample notebook at 1M records, the pair-similarity step
running on one GPU only, test prediction loading a whole country at once, and the word map being
looked up in the wrong folder by stage ③ (which then silently re-learned it).

We also made the experimental notebook write the same all-searches bucket file as the full pipeline,
so the experimental matcher can score exactly what the submission would.

### 10.10 IndicXlit transliteration: tested, not adopted

A teammate proposed replacing anyascii with IndicXlit (AI4Bharat's neural transliteration model, MIT)
for Indian-script text, and saw good-looking bucket numbers on a 1M sample. We tested it carefully:
- **Same results within noise.** Bucket recall and ceilings on a 1M sample matched ours within ±0.001
  (India top-50 recall 0.9652 vs 0.9646; India ceiling 0.9793 vs 0.9788; Indian-script recall@1 0.908
  vs 0.909). The reason is structural: stages ② and ③ transliterate Indian-script records from the
  original text themselves, so 01's transliteration never reaches the embedding text or the main
  comparison features. It only reaches legal form, phonetic code, city and state for those records.
- **Cheap to run.** Only 1,549 unique Indian-script words exist in the whole dataset (the data is
  templated). With a per-word cache, decoding all of them took 170 seconds on a laptop GPU, one time.
- **Mixed quality.** Some words came out better than anyascii (`solutions` vs `solyusns`, `pharmacy`
  vs `pharmesi`), others worse (`egeneerig` for engineering, `cusultaceae ajcj` for consultancy
  agencies, `anmd` for Anand).
- Two bugs found while wiring it in: a missing module-level import in the spliced engine, and 812
  encoding-corrupted tokens (mojibake apostrophes and dashes, also in non-Indian rows) being sent
  through the neural decoder as if they were Hindi.

Decision: not used in the main pipeline. The research track is in `submission-3/`; the two test notebooks
(`01_keshav.ipynb`, `02_keshav_exp.ipynb`) were later removed from the repo and are in the git history
([experiments/INDICXLIT.md](experiments/INDICXLIT.md)).

### 10.11 France audit and the final run (leaderboard 0.967)

The fixes in section 5.4 (regions, SASU/EI/Cie, French landmarks, house-number fallback) were made on
the last day, after checking every field 01 extracts on real French records. The final full run:

- cross-validation **0.9796** (India 0.9766, US 0.9826) at t ≈ 0.69; ceiling 0.9947;
- cross-country transfer: trained on US only, India scored 0.9135 (vs 0.9766 when trained on both);
  trained on India only, US scored 0.9280 (vs 0.9826);
- France predicted a match for 95.2% of its S1 with 3.57 matches each, against 94.1% and 3.51–3.55
  for the trained countries at the same cutoff;
- with the France cutoff at 0.9, the leaderboard gave **0.967**.

The effect of the France fixes alone was not isolated: France has no labels, and no pair of
submissions compared the two versions of 01 directly.

### 10.12 A bigger model: no gain

The last experiment doubled the tree capacity (127 leaves instead of 63), with the same training S1
and folds:

| | 63 leaves | 127 leaves |
|---|---|---|
| Best CV macro F0.5 | 0.9796 | 0.9797 |
| India / US | 0.9766 / 0.9826 | 0.9767 / 0.9827 |
| CV at t = 0.90 | 0.9762 | 0.9770 |

A gain of 0.0001 is noise. The model is not capacity-limited: the gap between 0.9797 and the 0.9947
ceiling is about what the features can distinguish, not the model's size. This agrees with an earlier
test-bed finding that larger trees and slower learning rates made no measurable difference.

---

## 11. Final results and error analysis

### 11.1 Where the points go (India + US, cross-validation)

```
1.0000  perfect
   │    −0.0053  search: true matches that never reached a bucket (India −0.0082, US −0.0024)
0.9947  ceiling: a perfect judge on our buckets
   │    −0.0150  judge: LightGBM's wrong keeps and wrong drops
0.9797  our cross-validation score
```

Early in the project (on a sample) the search lost more than the judge (0.018 vs 0.014). The extra
searches reversed that: the judge is now where the loss is.

### 11.2 France, derived from the leaderboard

The leaderboard averages over all test S1. With the test S1 counts per country and the cross-validated
India and US scores standing in for their test scores:

```
share of test S1:  India 809,986 / 1,732,544 = 0.4675   US 0.3827   France 0.1498
0.967 = 0.4675 × 0.9766 (India) + 0.3827 × 0.9826 (US) + 0.1498 × France
France ≈ 0.90   (0.89–0.90, since the leaderboard score is rounded)
```

That is about 8 points below the trained countries, and a little worse than the cross-country transfer
proxy (0.91–0.93) suggested. France is not only unseen, it is also harder: crowded cities and generic
names produce many near-identical lookalikes.

### 11.3 Common errors

**False positives (wrong matches):**
- Lookalikes in crowded cities: same city, same generic name word, sometimes the same house number or
  legal form, different street. In a small early test, France accepted such lookalikes about three
  times as often as India and the US. France's higher prediction volume (95.2% of S1 matched vs 94.1%)
  points the same way.
- Singletons that have a plausible lookalike: any match there costs the whole point.

**False negatives (missed matches):**
- Matches never retrieved. India loses 2.6% of its true pairs at the search step (recall 0.9745),
  concentrated in candidates with no address (80% retrieved) and names replaced by an unrelated word.
- Matches retrieved but rejected by the strict cutoff, mostly when the only evidence is a short or
  generic name, or when a lookalike scores higher and takes the record under the one-owner rule.

---

## 12. What did not work, or was not worth it

| Idea | Result | Why we stopped |
|---|---|---|
| FAISS IVF-PQ index | 85.5% recall vs 95.3% exact | exact search on two GPUs is fast enough and loses nothing |
| Bigger e5 top-k (30 → 50) | ceiling +0.003 for +20 pairs per S1 | the five targeted searches gave +0.009 for +13 pairs |
| Larger trees (127 leaves), slower learning rate | +0.0001 | the model is feature-limited, not capacity-limited |
| Per-S1 expected-F0.5 selection instead of one cutoff | +0.0002 | the flat cutoff curve leaves little to gain |
| Label-free France cutoff chosen by prediction volume | helped once, hurt once | not reliable; tuned on the leaderboard with notebook 04 instead |
| Self-training on France pseudo-labels | −0.005 to −0.01 | likely reinforces the model's own confident mistakes on lookalikes |
| IndicXlit neural transliteration | within ±0.001 | stages ② and ③ already transliterate from the original text; mixed word quality |
| House-number fallback in 01 | no measurable effect expected | stage ③ already extracts house numbers from the raw address |
| XGBoost or neural-network ensemble with LightGBM | not tried | same features and same missing France labels; small expected gain for a new, untested pipeline on the last day |
| Cross-encoder re-ranker (planned) | not built | the time went to recall and features, which paid more |

---

## 13. Engineering at full scale

**Compute:** Kaggle notebooks, two T4 GPUs (16 GB each), about 30 GB RAM. A full stage ③ run takes about
5 hours: roughly 2 hours for features, cross-validation and the final model, and 3 hours for test
prediction (France 22 min, India 1 h 42 min, US 55 min).

**Memory problems and their fixes:**
- **Sample notebook at 1M records ran out of memory.** Per-source embeddings were kept after the
  candidate matrix was built, a list of DataFrames leaked as a global, and five string columns were
  plain Python objects. Freed and converted to categoricals. One side effect had to be fixed: a
  pandas `.map()` over a categorical column returns float64 when some categories are unused, which broke
  `np.bincount`; the result is now cast to int.
- **Only one GPU was working** in the pair-similarity step. It is now split across both GPUs with one
  thread per GPU.
- **Test prediction ran out of memory** in submission 2's run, because it loaded the records of a whole
  country (India: 35M pairs) at once. It now loads them for one of `PRED_PARTS = 8` groups of S1 at a
  time. We checked, with the same model in the same kernel, that the probabilities are bit-identical
  to the old path: a group never splits an S1's bucket, and the statistics that compare a candidate
  across S1s are computed on the whole bucket first.
- **Training memory.** The feature matrix is float32: about 15.9M pairs × 85 features = 5.4 GB. Each
  cross-validation fold briefly adds copies of its train and validation rows, so doubling the training
  S1 to 400K per country would exceed Kaggle's RAM. Doubling the tree leaves only adds tens of MB (the
  per-leaf histogram cache).

**Why the embeddings are not saved:** all 24M records would take about 18.6 GB, close to Kaggle's output
limit. The buckets (a few hundred MB) are the stage's output.

---

## 14. Rules compliance

- **No external data or APIs.** Everything comes from the provided files. Word rarity and name
  frequency are computed from the records' own text, without labels. The French regions, legal forms
  and landmark words are hand-written domain knowledge, like the US and Indian state lists; nothing is
  learned from test labels.
- **Models:** `multilingual-e5-small` (MIT, 118M parameters) and LightGBM (MIT). IndicXlit (MIT) is
  only in the experimental notebooks.
- **Libraries:** sentence-transformers (Apache 2.0), rapidfuzz (MIT), anyascii (ISC), polars, pandas,
  pyarrow, PyTorch, scikit-learn.
- **No leakage:** the model learns only from the train ground truth; cross-validation is grouped by S1;
  test records are never labeled or hand-inspected for training.
- **Submission format:** 03 checks both files against the rules (every test S1 exactly once, S2/S3 IDs
  only, each record used once, matches ⊆ candidates) and runs `validate_submission.py` when available.

---

## 15. Lessons learned and next steps

**Lessons**
1. **Measure at full scale, or on whole regions.** A sample made a 0.946 model look like 0.974. Every
   later decision was checked on the test bed or on the full data.
2. **Fix the ceiling first, then the judge.** Early on, most points were lost at retrieval. Once the
   extra searches raised the ceiling to 0.995, the judge became the limit, and more model capacity did
   not help.
3. **Look at the misses, then build a search for each failure pattern.** Two small searches built from
   a miss analysis beat 20 more embedding neighbours per S1.
4. **Structural signals beat text similarity.** "Is this S1 the candidate's best S1, and by how much"
   carries half the model's gain, because the data has a one-owner structure.
5. **Audit every field on every country.** City and state were silently empty for all of France.
   Checking coverage per country, per field, is cheap and should have been done on day one.
6. **Verify a fix actually reaches the model.** Two of our fixes (house number, IndicXlit) changed
   01's output but not what stage ③ compares, because stage ③ recomputes those fields itself.

**What we would do next**
- **Features that separate lookalikes** (the judge's loss): street-level agreement weighted by how many
  businesses share the street, and name-word rarity *within the city* rather than the country.
- **France:** careful pseudo-labeling, keeping only pairs that are confident *and* unique in their city;
  or per-country calibration of the probabilities.
- **A cross-encoder** on the uncertain band only (0.2 < p < 0.9), where the cheap features disagree.
- **Per-country cutoffs** for India and the US, tuned on cross-validation.

**Experiments started after the final submission** (in `experiments/`, not part of the submitted pipeline):
- a fine-tuned embedding model: [EMBEDDING_FINETUNING.md](experiments/EMBEDDING_FINETUNING.md)
- our own embedding model, pretrained from scratch: [EMBEDDING_PRETRAINING.md](experiments/EMBEDDING_PRETRAINING.md).
  First full run (outputs in `submission-5/`): India recall@30 on S1 it never trained on went from 0.9467 (e5) to
  0.9947. It has not yet been run through 02 full and 03.
- an ensemble of LightGBM, XGBoost, CatBoost and a neural network: [ENSEMBLE.md](experiments/ENSEMBLE.md)

---

## 16. Reproducing the submission

Run on Kaggle with the competition data attached; the notebooks are in `submission-4/`. Details in
[SUBMISSION_STEPS.md](SUBMISSION_STEPS.md).

1. `01_eda_preprocessing.ipynb` with `FULL_RUN=true` → `processed/` (all rows, train and test).
2. `02_full_e5_buckets.ipynb` with `SPLITS=train,test` → `embeddings_full/multilingual-e5-small/buckets/`
   and `translit_wordmap.json`.
3. `03_full_lightgbm_submission.ipynb` with `COUNTRY_THRESHOLDS = {"France": 0.9}` →
   `submission/output/matching_results.tsv`, `candidate_pairs.tsv`, the model, and
   `test_probabilities/` per country.
4. Optional: `04_rethreshold.ipynb` to try other per-country cutoffs from the saved probabilities.
5. Validate: `python validate_submission.py --matching matching_results.tsv --candidate candidate_pairs.tsv --test-dir <test data>`.

Main settings (environment variables or the first code cell): `CANDIDATES_PER_S1=30`,
`EXTRA_SOURCES=reverse,address,name,number,empty`, `NUMBER_COUNTRIES=India`,
`TRAIN_S1_PER_COUNTRY=200000`, `FEATURE_SET=all`, `PRED_PARTS=8`.
