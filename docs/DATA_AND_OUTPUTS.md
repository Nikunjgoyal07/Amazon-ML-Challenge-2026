# What has what: data, folders and output files

Each notebook reads the output of the one before it. This page lists every folder they create, what
each file contains, and roughly how big it is at full size.

```
competition data ─► 01 ─► processed/ ─► 02_full ─► embeddings_full/…/buckets/ ─► 03_full ─► submission/ ─► 04 ─► rethreshold/
                                   └──► 02 (sample) ─► embeddings/… ─► 03 (sample) ─► lgbm_matcher/
```

On Kaggle, every folder is written under `/kaggle/working/`. To use it in the next notebook, add the
previous notebook's output as an input. Each notebook finds its inputs by searching `/kaggle/input`.

---

## Competition data (input)

| File | Content |
|---|---|
| `train/train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv` | `entity_id`, `business_name`, `business_address`, `country` |
| `train/train_ground_truth.tsv` | `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 IDs, empty = no match) |
| `test/test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv` | same columns as train; no labels. France appears only here |

Locally: `Data/student_resource/dataset/`. The validator is `Data/student_resource/utils/validate_submission.py`.

---

## `processed/`, written by `01_eda_preprocessing.ipynb`

One parquet file per split, source and country:
- `train_s1_India`, `train_s1_US`, `train_s2_*`, `train_s3_*`
- `test_s1_France`, `test_s1_India`, `test_s1_US`, `test_s2_*`, `test_s3_*`

`FULL_RUN=true` is needed for all rows; without it each file is capped at 40K rows (sample mode).

| Column | Meaning |
|---|---|
| `entity_id`, `business_name`, `business_address`, `country` | the original record |
| `name_norm` | cleaned name: lower case, no accents, no junk wrappers or honorifics |
| `core_name`, `legal_form` | name without its legal form, and the legal form (`pvt ltd`, `llc`, `sarl`, ...) |
| `domain_stem` | for names that are a website: the name without `.com` etc. |
| `name_lat`, `phonetic_code` | Latin transliteration of the core name; soundex code of its first word |
| `has_honorific` | 1 if an honorific (Shri, Smt, Mr, Dr) was removed |
| `addr_norm`, `street_tokens` | cleaned address with abbreviations expanded; its words, sorted |
| `house_no`, `zip_pin_cp`, `city`, `state_code` | parsed address parts (often empty: they are only filled when the format allows) |
| `empty_addr`, `missing_pin`, `missing_city`, `landmark_flag` | 0/1 flags |

Also written: `eda_summary.json` (EDA facts) and `processed_counts.json` (rows per file).

**Caveat:** for text in Indian scripts, `name_norm`, `name_lat` and `addr_norm` are garbled, because
vowel signs are removed with the accents. Stages 02 (full) and 03 (full) transliterate the original
text themselves.

---

## `embeddings_full/multilingual-e5-small/`, written by `02_full_e5_buckets.ipynb`

| File | Content | Full-size rows / size |
|---|---|---|
| `buckets/train_India.parquet`, `buckets/train_US.parquet` | top 30 e5 candidates of every train S1, plus the extra candidates | ~70M rows in total (estimate) |
| `buckets/test_France.parquet`, `buckets/test_India.parquet`, `buckets/test_US.parquet` | the same for every test S1 | ~57M rows in total (estimate) |
| `bucket_recall_train.csv` | per country and candidate set (e5 top-20/30/40/50, e5 top-30 plus each extra source, everything saved): pairs per S1, share of true matches inside (overall, split by whether the candidate is written in an Indian script or in Latin script, and for candidates with an empty address), ceiling score | a few rows |
| `manifest.json` | settings (model, text mode, top-k, extra sources), records, new pairs per source and timings per bucket file | |
| `translit_wordmap.json` | the word map for Indian-script names ({"praivet": "private", …}, ~370 words), learned from the train true pairs; 03 (full) reads it | a few KB |

Bucket columns, one row per (S1, candidate):

| Column | Meaning |
|---|---|
| `s1_entity_id`, `candidate_entity_id` | the pair |
| `rank` | position in the bucket by `score`, 0 = the most similar candidate of this S1 |
| `candidate_source` | `S2` or `S3` |
| `score` | similarity after mean-centering (what the search ranks by) |
| `cosine` | original e5 cosine similarity (0–1) |
| `country` | the country of both records |
| `e5_rank` | rank in the e5 search (0–29); empty = added by another search only |
| `rev_rank` | rank of this S1 among the candidate's 5 best S1 (reverse search); empty = not among them |
| `addr_key` | 1 = same house number and first street word |
| `name_rank` | rank among the S1's 5 most similar names (character 3-grams); empty = not among them |
| `num_key` | 1 = the addresses share a number and another word (the number key; India only by default) |
| `empty_rank` | candidates with no address: rank among the S1's 2 most similar names of such candidates; empty = not among them |
| `empty_rev_rank` | candidates with no address: rank of this S1 among the candidate's 3 most similar S1 names; empty = not among them |

The embeddings themselves are **not** saved: ~18.6 GB for all 24M records, close to Kaggle's 20 GB
output limit. Nothing downstream needs them.

---

## `submission/`, written by `03_full_lightgbm_submission.ipynb`

| File | Content |
|---|---|
| `output/matching_results.tsv` | **the leaderboard file**: `source1_entity_id` <TAB> `matched_entity_ids`, one row per test S1, IDs most likely first, empty = no match (~95 MB) |
| `output/candidate_pairs.tsv` | `source1_entity_id` <TAB> `candidate_entity_ids`: the 30 candidates the model scored per S1 (~700 MB at 30 per S1) |
| `model/lgbm_matcher.txt` | the final LightGBM model (text format) |
| `model/matcher_config.json` | feature list, parameters, rounds, cutoff, per-country cutoffs, cross-validation scores, test statistics |
| `model/metric_f05.csv` | cross-validation score table: predict nothing, cosine rule, LightGBM, LightGBM + one owner, ceiling |
| `model/feature_importance.csv` | gain and split count of every feature |
| `model/oof_pairs.parquet` | every training pair with its out-of-fold probability (`p`), label (`is_true`), fold and keep decision |
| `test_probabilities/<country>.parquet` | every test pair: `s1_entity_id`, `candidate_entity_id`, `rank`, `p` (used by notebook 04) |

---

## `rethreshold/`, written by `04_rethreshold.ipynb`

| File | Content |
|---|---|
| `matching_results.tsv` | the matches re-decided with `COUNTRY_THRESHOLDS` (e.g. a stricter France cutoff), same format and rules as 03's. Submit it together with 03's unchanged `candidate_pairs.tsv` |

---

## Sample experiments (not needed for a submission)

### `embeddings/<model>/<sample>/`, written by `02_e5_embeddings.ipynb`

One folder per sample, named from its size (e.g. `sample_100k`, `sample_s1-50k_s2-200k_s3-200k_India`), so samples of different sizes sit side by side.

The sample notebook embeds a diverse sample (default 100K per source; 500K was used on Kaggle).

| File | Content |
|---|---|
| `s1_embeddings.npy`, `s2_embeddings.npy`, `s3_embeddings.npy` | fp16, one 384-number row per sampled record |
| `s1_meta.parquet`, `s2_meta.parquet`, `s3_meta.parquet` | row-aligned: `entity_id`, split, country, `kind` (linked / no_match / singleton / …), true S1 link, diversity tags, embedded text |
| `manifest.json`, `quick_check.csv` | settings, counts, tag shares; recall of the true S1 per slice |
| `faiss/s1_buckets.parquet`, `faiss/s1_buckets_exact.parquet`, `faiss/s1_buckets_ivfpq.parquet` | top-k buckets from exact search and (optionally) FAISS IVF-PQ |
| `faiss/index_<country>.faiss`, `faiss/mean_<country>.npy` | the IVF-PQ index and the centering mean (IVF-PQ mode only) |
| `faiss/bucket_recall.csv`, `faiss/metric_f05.csv` | true matches in the buckets; F0.5 table for simple rules |
| `faiss/s1_buckets_all.parquet`, `faiss/extra_recall.csv` | (with `EXTRA_SEARCHES`) e5 top-30 + every extra search in 02 full's bucket format, for 03 (sample) with `BUCKETS=all` / `sub2`; e5 top-k vs e5 top-30 + each search (pairs per S1, true matches inside, ceiling) |

### `lgbm_matcher/<model>/<sample>/`, written by `03_lightgbm_matcher.ipynb`

One folder per sample it was trained on (plus `_max<N>` when `MAX_S1` limits the S1, and `_all` / `_sub2` for
those `BUCKETS`; `BUCKETS=e5` keeps the plain name).

| File | Content |
|---|---|
| `lgbm_matcher.txt`, `matcher_config.json` | model and settings trained on the sample buckets (44 features, 51 with `BUCKETS=all` / `sub2`), candidates per S1 |
| `oof_pairs.parquet`, `metric_f05.csv`, `feature_importance.csv` | out-of-fold predictions, score table, importance |
| `sample_matching_results.tsv` | out-of-fold matches of the sampled S1, in the submission format |

---

## Submission file formats (from the problem statement)

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00003	
```

Both files:
- are tab-separated and UTF-8, with `\n` line endings
- have exactly one row per test S1, including France
- contain only S2-/S3- IDs, with no duplicates inside a list

The matches must be a subset of the candidates. 03 (full) guarantees this, and
`validate_submission.py` checks it.
