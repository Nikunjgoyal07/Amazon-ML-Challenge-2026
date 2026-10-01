# 03 · Matcher and submission (`submission-4/03_full_lightgbm_submission.ipynb`)

**What it does:** judges every (S1, candidate) pair in the buckets with a LightGBM model, decides the matches, and
writes the two submission files.

```
buckets (02) + processed/ (01) + ground truth  ──►  03  ──►  submission/output/matching_results.tsv
                                                              submission/output/candidate_pairs.tsv
                                                              submission/model/…, submission/test_probabilities/…
```

**Run:** Kaggle; the CPU is enough, since LightGBM uses the GPU only if its build supports it. A full run took about
5 hours:
- **Training side** (features, cross-validation, final model): about 2 hours.
- **Test prediction:** about 3 hours (France 22 min, India 1 h 42 min, US 55 min).

`PREDICT_TEST=false` stops after the scores, for experiments.

Why it works this way and what each idea gained: [SOLUTION_WRITEUP.md §7-8](../SOLUTION_WRITEUP.md). How to read the
printed tables: [RESULTS_EXPLAINED.md](../RESULTS_EXPLAINED.md).

---

## Cell by cell

| Cell | What it does | Key names |
|---|---|---|
| 1 | Settings, input discovery, LightGBM device test, parameters | `PARAMS`, `lgbm_device` |
| 3 | The competition metric, with the problem statement's example as a test (0.714) | `f05`, `macro_f05` |
| 5 | **Records and features**: reading records, per-country statistics, the 85 features | `load_records`, `country_stats`, `read_buckets`, `make_features` |
| 7 | Ground truth: the true S1 of every S2/S3 record | `GT_LIST`, `OWNER` |
| 9 | Training pairs: 200,000 random train S1 per country, their features and labels | `X`, `y`, `L` |
| 11 | 3-fold cross-validation grouped by S1 | `fit`, `OOF` |
| 13 | Cutoff and scores (vectorized metric, one-owner rule, cutoff search) | `score`, `one_owner`, `tune`, `METRIC` |
| 15 | Score at other cutoffs (should be flat around the best) | |
| 17 | Train on one country, score the other (stand-in for France) | `TRANSFER` |
| 19 | Final model on all training pairs, feature importance | `FINAL`, `IMPORTANCE` |
| 21 | Test prediction in parts, one owner, cutoff | `part_bounds`, `join_by_s1` |
| 23 | Writing the two TSV files and `matcher_config.json` | `write_tsv`, `CONFIG` |
| 25 | Checking both files against the submission rules, then the official validator | `check_file` |

---

## Reading records and statistics (cell 5)

- **`read_buckets(split, country)`:** the bucket file, cut to the top `CANDIDATES_PER_S1` e5 candidates plus 02's
  extra candidates.
- **`load_records(...)`:** 01's columns for the IDs needed. Any Indian-script name or address is replaced by its
  transliteration (`translit`, plus the word map for names), the same as in 02. This happens before any comparison.
  It also sets the flags used as features: native script, website name, empty address, honorific, landmark.
- **`country_stats(split, country)`:** computed over **all** S1 of the country, from text only, never labels.
  - **Word rarity:** IDF over the country's S1 names and addresses. "club" is cheap in France; "diaspora" is rare
    everywhere.
  - **Name frequency:** how many S1 per 100,000 share a name's set of words.
  - **Competition between S1s:** for every candidate, all S1 buckets that contain it, so a pair can be compared with
    the other S1s that want the same record.

Because these use every S1 of the country, the training pairs see exactly the competition they will see on test.

## Training pairs (cell 9)

- **Which S1:** for each train country, in sorted order, `rng.choice(s1_all, 200000, replace=False)` with
  `rng = np.random.default_rng(42)`. `s1_all` is the S1 in bucket-file order. 02b and 02c recreate this exact draw to
  keep these S1 out of their training.
- **Pairs and labels:** every pair of their buckets, labeled 1 when the candidate's true S1 (from the ground truth)
  is this S1.
- **Size:** about 15.9M pairs × 85 features in float32 = 5.4 GB. That's why 400,000 S1 per country doesn't fit in
  Kaggle's memory.

---

## The 85 features (`make_features`)

Country is deliberately **not** a feature, so the same model can judge France. When either side of a comparison is
empty, the feature is left empty (NaN), and LightGBM handles that as "unknown".

**Search and embedding (7)**

| Feature | Meaning |
|---|---|
| `rank` | position of the candidate in the S1's bucket |
| `score`, `cosine` | e5 similarity: mean-centered, and plain |
| `s1_best_cosine` | the S1's best cosine among all its candidates |
| `gap_to_best_score`, `gap_to_next_score` | how far behind the S1's best candidate; lead over the next one |
| `s1_top5_cosine` | mean cosine of the S1's top 5 (how crowded its bucket is) |

**Competition between S1s (3).** These are the strongest group: each record belongs to at most one S1.

| Feature | Meaning |
|---|---|
| `cand_n_s1` | how many S1 buckets contain this candidate |
| `cand_rank` | this S1's rank among them (0 = the candidate's best S1) |
| `cand_margin` | if best: lead over the second-best S1; otherwise how far behind the best (negative) |

**Name (15).** Similarity scores are rapidfuzz scores from 0 to 100.

| Feature | Compares |
|---|---|
| `name_ratio`, `name_token_set`, `name_token_sort`, `name_partial` | cleaned names |
| `core_ratio`, `core_jaro_winkler` | names without legal form |
| `latin_ratio`, `latin_token_set` | transliterated core names |
| `raw_name_token_set` | original names, lowercased |
| `compact_ratio`, `compact_partial` | names with spaces and dots removed, or the website stem (`eelegal.com` vs "E Legal") |
| `phonetic_same` | same Soundex code of the first word |
| `legal_form_same` | same legal form (pvt ltd, llc, sarl, …) |
| `s1_name_len`, `cand_name_len` | name lengths |

**Address (13)**

| Feature | Meaning |
|---|---|
| `addr_ratio`, `addr_token_set` | cleaned addresses |
| `raw_addr_token_set`, `raw_addr_partial` | original addresses |
| `street_jaccard`, `street_cover` | shared address words: overlap, and share of the S1's words found |
| `numbers_jaccard`, `numbers_cover` | the same for the numbers in the address (`0058` = `58`) |
| `postcode_same`, `house_no_same`, `city_same`, `state_same` | same field (01's parsed columns) |
| `cand_addr_len` | candidate address length |

**Street (3)**

| Feature | Meaning |
|---|---|
| `street_name_token_set`, `street_name_ratio` | the street names (from `street_of`, as in 02): "45 Rue du Port" vs "45 Rue du Pressoir" |
| `house_number_same` | same house number, parsed here from the original address |

**Word rarity (8).** Name and address words are weighted by their rarity in the country (IDF):

| Feature (`name_…` and `addr_…`) | Meaning |
|---|---|
| `…_idf_jaccard` | rarity-weighted overlap |
| `…_idf_cover` | share of the S1's rarity weight found in the candidate |
| `…_idf_miss_s1`, `…_idf_miss_cand` | rarity of the rarest word one side has and the other lacks |

**Typo-tolerant word rarity (8).** The same 8 features with a `_fuzzy` suffix. Here, words 1 edit apart (4-5
letters) or 2 edits apart (6+ letters) count as shared, so "westgrove" vs "wcstgrove" isn't a missing rare word.

**Name frequency (2):** `s1_name_freq`, `cand_name_freq`, how common each name is in the country (log scale). A
match on "Bordeaux Club" means little.

**Candidate vs candidate (13).** The candidate is compared with the other candidates of the same S1:

| Feature | Meaning |
|---|---|
| `{addr_token_set, addr_idf_cover, street_name_token_set, name_token_set, name_idf_cover}_gap_best` and `…_rank` | how far behind the bucket's best, and the rank in the bucket, for 5 similarities (10 features) |
| `n_name_twins`, `n_addr_twins` | candidates in the bucket with a near-identical name / address (token-set ≥ 90) |
| `better_addr_same_name` | candidates with a name at least as close and an address at least 10 points closer |

**Candidate flags (6):** `cand_is_s3`, `cand_native_script`, `cand_web_name`, `cand_empty_address`,
`cand_honorific`, `cand_landmark`.

**Found by (7).** Which of 02's searches found the pair: `e5_rank`, `rev_rank`, `addr_key`, `name_rank`, `num_key`,
`empty_rank`, `empty_rev_rank` (empty = not found by that search).

**What the model actually uses** (share of gain in the final model):

| Feature | Share of gain |
|---|---|
| `cand_margin` | 48.6% |
| `rev_rank` | 25.5% |
| `numbers_jaccard` | 5.3% |
| `numbers_cover` | 1.5% |
| `name_ratio` | 1.4% |
| `house_number_same` | 1.4% |
| `cand_rank` | 1.4% |
| `name_token_sort` | 1.3% |

Every other feature has under about 1%.

`FEATURE_SET` selects smaller sets for comparisons: `base` (44 features: search, competition, name, address and
flags), `submission2` (61), `all` (85, default).

Features are computed in chunks of about 2M pairs (`PAIRS_PER_CHUNK`, `chunk_bounds`) that never split an S1's
bucket. That matters because the candidate-vs-candidate features need the whole bucket.

---

## Training and validation (cells 11-17)

- **LightGBM settings:** binary objective, 63 leaves, learning rate 0.1, at least 100 rows per leaf, 80% of features
  and rows per tree, L2 regularization of 1.0.
- **Cross-validation:** the training S1 are split into 3 groups (folds), with all pairs of an S1 in the same group.
  Each group is predicted by a model trained on the other two, which stops 50 rounds after its log-loss stops
  improving. Every pair thus gets an out-of-fold probability `p` from a model that never saw its S1.
- **Metric:** `score(df, keep)` computes macro F0.5 exactly as the competition does, vectorized with `np.bincount`.
  It is asserted equal to the simple per-S1 reference (`macro_f05`). True matches missing from the buckets count as
  misses, so the score is realistic.
- **Decision rule:**
  - `one_owner` keeps each S2/S3 record only for the S1 that gives it the highest `p`.
  - `tune` tries cutoffs 0.02 to 0.98 and keeps the best one, `T_BEST` (about 0.69-0.71 in the final runs).
  - The table also shows the cosine-only rule, the model without one-owner, and the **ceiling** (a perfect judge on
    the buckets).
- **Country transfer:** train on India and score the US, then the reverse. It is the only stand-in for France,
  which has no labels.
- **Final model:** all training pairs, for 1.1 × the folds' average best round count.

## Test prediction (cell 21)

- **Parts:** per country, the S1 are split into `PRED_PARTS` = 8 groups, and the records of one group are loaded at
  a time. Loading all of India at once ran out of memory. A group never splits an S1's bucket, and the statistics
  were computed on the whole bucket first, so the probabilities are identical to computing everything at once
  (checked bit for bit).
- **Decision:** one owner per S2/S3 record, then `p ≥ t`, where `t` is `COUNTRY_THRESHOLDS[country]` or `T_BEST`.
  An S1 with nothing above the cutoff gets an empty list, which is the "no match" prediction.
- **Saved probabilities:** every test pair's `p` goes to `test_probabilities/<country>.parquet`, so 04 can try other
  cutoffs in minutes.

## Writing and checking the files (cells 23-25)

- **`matching_results.tsv`:** every S1 of `test_source1.tsv` exactly once, in file order. Matches are
  comma-separated, most likely first, and the list is empty when there is no match.
- **`candidate_pairs.tsv`:** the exact pairs that were scored, so the matches are always a subset of them.
- **`check_file`** verifies:
  - the header
  - one row per test S1 (none missing, extra or duplicated)
  - no duplicate ID within a list
  - only `S2-`/`S3-` IDs, all of which exist in the test S2/S3 files
  - matches ⊆ candidates

  If `validate_submission.py` is attached, it then runs with `--check-ids`.

## Outputs (`submission/`)

| File | Content |
|---|---|
| `output/matching_results.tsv`, `output/candidate_pairs.tsv` | the submission |
| `model/lgbm_matcher.txt` | the final LightGBM model |
| `model/matcher_config.json` | features, parameters, cutoff, per-country cutoffs, all scores and timings (04 reads the cutoff from here) |
| `model/metric_f05.csv`, `model/feature_importance.csv` | the score table and feature importance |
| `model/oof_pairs.parquet` | every training pair with its out-of-fold `p`, fold and label |
| `test_probabilities/<country>.parquet` | every test pair with its `p` |

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `BUCKET_DIR`, `PROCESSED_DIR`, `DATA_ROOT`, `SUBMISSION_DIR` | searched / `submission` | inputs and output |
| `CANDIDATES_PER_S1` | 30 | e5 candidates judged per S1 (≤ 02's `TOP_K`) |
| `USE_EXTRA_CANDIDATES` | `true` | also judge 02's extra candidates |
| `TRAIN_S1_PER_COUNTRY` | 200,000 | training S1 per country (memory limit; keep it for 02b/02c) |
| `FEATURE_SET` | `all` | `all` (85), `submission2` (61), `base` (44) |
| `WORD_MAP` | `true` | word map for Indian-script names (same file as 02) |
| `PRED_PARTS` | 8 | test S1 groups per country; raise it if memory runs out |
| `COUNTRY_THRESHOLDS` | `{}` | stricter cutoff per country, e.g. `{"France": 0.9}` (in the first code cell) |
| `PREDICT_TEST` | `true` | `false` = experiment: scores only |
| `LGBM_DEVICE` | `auto` | `cpu`, `gpu`, `cuda` |

## Adding a feature

1. **Compute it** in `make_features` as an array aligned with the bucket rows, and add its name to `FEATURES`.
2. **Keep it country-free:** it must make sense for France. Use only text and unlabeled statistics.
3. **Compare** cross-validation with and without it, at full size or on whole regions; samples overstate gains. The
   model is limited by its features, not its size: 127 leaves gave +0.0001.
