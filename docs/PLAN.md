# Amazon ML Challenge 2026: Business Entity Resolution Plan

*Status: preprocessing and a small embedding benchmark exist; blocking, matching, decision logic, and final outputs are not implemented yet. Written/updated 2026-09-24.*

## Context
The repo now contains the challenge material plus preprocessing/embedding notebooks under `code/business_entity_resolution/`; the blocking, matcher, decision, and output stages remain to be implemented. The goal: for every test Source 1 (S1) record, output the S2/S3 records that refer to the same business. The score is **macro F0.5 per S1**; a correct empty list scores 1.0 and any false match on a singleton scores 0. We also have to ship `candidate_pairs.tsv`, runnable code and a methodology doc.
Constraints: no external data or APIs, final model must be MIT/Apache and ≤8B parameters, 5 submissions/day over Sep 25–27. Compute: Kaggle/Colab with 2× T4 and ~30 GB RAM.

## What the data tells us (EDA done)
| | Train | Test |
|---|---|---|
| S1 | 2.21M (US 1.32M, IN 0.88M) | 1.73M (US 0.66M, **FR 0.26M**, IN 0.81M) |
| S2 | 5.03M | 4.89M |
| S3 | 5.29M | 5.08M |

- **Each S2/S3 record matches at most one S1 in the observed training labels** (max reuse = 1). Use this as a strong many-to-one assignment prior, but validate it on held-out groups and do not merge two S1 entities.
- **Singletons are only 5.6%** of train S1. Match counts peak at 3–4 (mean ≈3.5, max 11); split S2 48% / S3 52%.
- **~26% of S2/S3 records are distractors** that match nothing. These are the main source of false positives.
- **There are no cross-country matches in the observed training links** (0 of 7.64M checked), so same-country retrieval is a strong primary prior; keep a documented fallback/stress test for unseen test behavior.
- **Leakage audit:** a normalized exact `(name, address, country)` comparison found zero test↔train-S1 pair overlap; about 33.8% of test S1 names collide with train-S1 names, so name-only lookup is unsafe. Recompute with a semantic/near-duplicate audit before relying on this result.
- Empty address rate: S2/S3 ~3%, S1 0%.
- Noise we saw (it looks synthetic and systematic, so it can be learned):
  - junk prefixes/wrappers: `***`, `>>`, `--`, `<<`, `##`, `[Services]`, `(ID: 1072)`, `<NULL>`
  - honorifics: `Smt`, `Shri`
  - typos: `Adndi`, `Wcstgrove`, `Tocmroing`
  - accent injection: `Prívate`, `Léarning`
  - word shuffles and duplicated tokens
  - legal suffixes: Pvt/Private, Ltd/Limited
  - **name replaced by a domain** (`eelegal.com`, `projectsmanipalindia.com`)
  - **name in Devanagari** (`गुरु इंफोटेक प्रा. लि.`)
  - **state in native script** (`தமிழ்நாடு`, `महाराष्ट्र`, `दिल्ली`)
  - **completely unrelated name, address-only match** (`Evoavi`)
  - address changes: component reordering, abbreviations (`Rd`, `Cir`, `R.`), injected `H.no 780` / `PO Box` / `Unit`, dropped city or PIN, state code ↔ name (`MH`, `RJ`, `Texas`/`TX`)
- **France never appears in train**, so every component has to be country-agnostic.

## Approaches considered
| # | Approach | Pros | Cons | Verdict |
|---|---|---|---|---|
| A | Rules/deterministic: normalize, then exact or fuzzy keys plus thresholds | Fast, gives a Day-1 submission | Misses transliteration, domain and address-only cases; brittle on FR | Baseline only |
| B | Multi-pass blocking + hand-crafted pair features + **LightGBM** | Proven for ER, cheap, interpretable, handles 30M+ pairs on CPU | Weak on Devanagari names and unseen FR patterns unless features cover them | **Core** |
| C | Dense retrieval (multilingual bi-encoder + FAISS-GPU) for blocking | High recall across scripts and domains; one index per country | Needs GPU embedding/indexing of roughly 12M records; benchmark actual throughput and memory rather than relying on a fixed time estimate | **Core (blocking)** |
| D | Fine-tuned **cross-encoder** (mMiniLM / mDeBERTa-v3-base / XLM-R-base, MIT/Apache) over candidate pairs | Best semantic accuracy on hard pairs; multilingual backbones are a plausible France-transfer option, but must be tested on a country-held-out proxy | Inference over tens of millions of pairs is heavy, so run it only on LightGBM's uncertain band | **Stage 2 booster** |
| E | Graph/cluster resolution over S1–S2–S3 (S2↔S3 duplicate links, connected components) | Uses the one-to-one structure and transitivity | Can propagate errors; adds complexity | Post-processing |
| F | LLM ≤8B (e.g. Qwen2.5-7B, Apache) as judge | Strong reasoning | Far too slow for millions of pairs on T4 | Skip, or only a tiny ambiguous set |

## Recommended pipeline: B + C for the core, then D and E layered on top

### 1. Normalization (`src/normalize.py`)
- Unicode NFKC, then strip accents (fixes `Prívate` and French text), lowercase.
- Strip the junk prefixes/wrappers, `(ID: n)` and `<NULL>`.
- Canonicalize honorifics and legal suffixes into a separate `legal_form` field. Include FR forms such as SARL, SAS, S.A.S, SA, EURL, and SCI; treat `Groupe` as an optional alias/feature rather than automatically deleting it.
- Keep the legal-stripped "core name".
- Domain names: detect `.com`/`.in`/`.fr` and store the stem. Compare it against the S1 core name with spaces removed.
- Transliterate Indic scripts to Latin with `indic-transliteration` (MIT) or `anyascii` (ISC). **Avoid GPL `unidecode`.** Also learn a native-script state → state-code map from train co-occurrences; keep the map fold-safe and document its provenance because organizer treatment of test-derived/transductive maps should be confirmed.
- Address parsing:
  - Extract house/plot numbers, ZIP (US 5-digit), PIN (IN 6-digit), CP (FR 5-digit), city and state.
  - Expand abbreviations for EN, IN and FR (`R.`→rue, `Bd`→boulevard, `Av.`→avenue).
  - Map state code ↔ name.
  - Drop injected noise tokens (`H.no`, `Hn`, `Door No`, `PO Box`).
- **Implementation gap:** the current `01_preprocess.ipynb` prototype exposes numeric address tokens and state, but not a reliable house/ZIP/PIN/CP/city parser. Recompute the address EDA with country-aware parsing before using the current PIN-present percentages.
- Process one country at a time with polars/pyarrow to stay inside 30 GB.

### 2. Candidate generation (`src/blocking.py`), per country, as a union of passes
Query **S2/S3 → S1** (10M queries against a 1.7M index), because each S2/S3 record has at most one owner. Then invert the results into per-S1 lists.
1. **Dense name+address**: embed `name | address` with `intfloat/multilingual-e5-small` (MIT) or `paraphrase-multilingual-MiniLM-L12-v2` (Apache), fp16 on both T4s, then FAISS-GPU inner-product search, top-k≈10.
2. **Char 3-gram TF-IDF on core name**: sparse top-k (`sparse_dot_topn`, or a GPU matmul by chunks). Catches typos and domain stems.
3. **Address-only key/TF-IDF**: (house no + street tokens + city/ZIP/PIN), top-k. Catches `Evoavi`-style cases where the name is unrelated.
4. **Exact keys**: (house no, ZIP/PIN), plus phonetic core-name + city.
- Tune k per pass on full country indexes and held-out S1 groups. Measure edge recall, full-list recall, candidate count, oracle macro-F0.5, and hard-slice recall; do not extrapolate from the 10K-index embedding benchmark or impose an arbitrary 15–25 cap.
- Whatever reaches the final model becomes `candidate_pairs.tsv`, which by rule must be the last stage.
- Treat S1↔S2, S1↔S3, and S2↔S3 evidence as separate graphs; use mutual/reciprocal high-confidence links for tie-breaking and consistency features, not unconditional transitive merging.
- Keep address parsing as an auditable feature-producing stage; never silently rewrite the raw address.

### 3. Pair features + LightGBM (`src/features.py`, `src/train_lgbm.py`)
- **Name features** (all with `rapidfuzz`, MIT): Jaro-Winkler, Levenshtein ratio, token_set/sort ratio, char-3gram cosine, token Jaccard, legal_form agree/conflict, domain-stem vs name similarity, transliterated-name similarity, phonetic match.
- **Address features**: house-number exact/conflict, ZIP/PIN/CP match/conflict/missing, street-token overlap, city match, normalized-state match, address TF-IDF and embedding cosine.
- **Structural features (key for the one-owner constraint)**:
  - rank of this S1 among the S2/S3 record's candidates
  - score margin to the best and second-best S1
  - number of candidates
  - which blocking passes produced the pair
  - source (S2/S3), missing-field flags
  - name/address frequency, IDF, block size, postal/street collision count, and other uniqueness features
  - high-confidence S2↔S3 sibling similarity/consistency features (used as evidence, never as an unconditional merge)
- **Do not use country identity as a fixed categorical feature.** Use dynamic country partitions plus generic `same_country` and missing/unknown features; compute name/address frequency and uniqueness features separately. French records must pass through the same retrieval and inference path.
- Train with GPU LightGBM. Split by S1 group (e.g. 90/10) after blocking the full train set, so candidate density matches the test set.

### 4. Cross-encoder refinement (`src/cross_encoder.py`), Day 2
- Fine-tune `mdeberta-v3-base` (MIT) or a multilingual MiniLM on an initial 200K–500K hard-example set, expanding only if the full-index baseline justifies it. Use the observed synthetic transformations as augmentation and randomize pair/source order. Input: `name [SEP] address` for each side, max 96 tokens, fp16, both T4s.
- Measure the uncertain-band size before inference. With 150–250M candidate pairs, a nominal 0.05–0.95 band can still contain tens of millions; score only a measured small set of top-1/top-2 disagreements or high-impact singleton cases.
- Blend the two scores with logistic stacking on out-of-fold validation.

### 5. Decision layer for macro F0.5 (`src/decide.py`)
- **Owner decision:** compare no assignment, dirty-record argmax, and a sparse many-to-one assignment with an explicit no-match option. Never use a standard one-to-one solver that would force two S1 entities to compete or merge.
- **Singleton/cardinality gate:** train an S1-level `has_match`/cardinality head; predict an empty list for likely singletons.
- **Listwise matcher:** for each S2/S3 query, rank its S1 candidates together with a reject option using LambdaRank/softmax-style training; this matches the observed one-owner structure better than random pair negatives.
- **Metric-aware output:** for each S1, choose the prefix of ranked candidates that maximizes **expected F0.5** under calibrated probabilities, with the singleton case handled explicitly.
- Optional graph step (E): use high-confidence S2↔S3 evidence to re-score or remove **existing candidates**; never output an ID absent from the final `candidate_pairs.tsv`, and do not blindly union connected components.
- Tune every threshold and assignment policy with an exact macro-F0.5 scorer on out-of-fold predictions (`src/metric.py`).

### 6. France generalization
- **Proxy/stress test:** leave-one-country-out (train US → evaluate India and reverse) is a useful distribution-shift test, not a literal France simulation because language, script, and address conventions change together. Add native-script versus Latin holdouts inside India and a France-like held-out transformation proxy. Keep only features and normalizations that survive these tests.
- The FR abbreviation/legal dictionaries are intended as hand-written domain knowledge, not external lookup. Because the rules do not explicitly define every borderline case, record the provenance and obtain written organizer clarification before shipping them.
- Optional: self-training on the test set's high-confidence FR matches, to adapt thresholds per country. This is transductive rather than ordinary train-only validation; keep it as a separate ablation and ship it only after confirming that organisers allow it.

## Project layout (matches the required zip)
```
code/business_entity_resolution/
  src/ normalize.py blocking.py embed.py features.py train_lgbm.py
       cross_encoder.py decide.py metric.py run_pipeline.py write_outputs.py
  README.md  requirements.txt (pinned)
output/ matching_results.tsv  candidate_pairs.tsv
Documentation_template.md (filled)
```
Reuse `Data/student_resource/utils/validate_submission.py` before every upload.

## Timeline (15 submissions total)
- **Today (Sep 24), prep:**
  - Set up the environment and upload the data to Kaggle.
  - Write normalize, metric, rule baseline (A) and the blocking recall harness.
- **Day 1 (Sep 25):**
  - Consolidate the preprocessing implementation, audit the address parser, and quantify source-specific noise.
  - Build blocking passes 1–4 and measure full-index edge/full-list recall on realistic query samples.
  - Build the listwise/pair feature pipeline and LightGBM baseline.
  - Submissions: rule baseline, then LightGBM.
- **Day 2 (Sep 26):**
  - Run the leave-one-country-out check and harden FR handling.
  - Add the cross-encoder, stacking and the expected-F0.5 decision.
  - Submissions: threshold variants.
- **Day 3 (Sep 27):**
  - Graph post-processing, final blend, keeping submissions stable across the public and private splits (avoid overfitting the public LB).
  - Package the zip and write the documentation.

## Verification
- `src/metric.py` reproduces the PDF example: 0.714.
- Blocking: report pair recall, the recall ceiling on validation F0.5, and reduction ratio, per country.
- Hold-out macro F0.5 on the 10% S1 group split, plus leave-one-country-out scores.
- Run `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids` and require PASS with no errors or warnings; the official rules say invalid/nonexistent S2/S3 IDs are not acceptable even if the helper's default mode treats them as diagnostic.
- Reproducibility: a clean Kaggle run of `run_pipeline.py` regenerates both TSVs.
