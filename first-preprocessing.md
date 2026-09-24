# First Preprocessing — Detailed Run Report (full preprocessing.md coverage)

Branch: `eda-preprocessing` | Notebook: `01_eda_preprocessing.ipynb` (executed copy: `01_eda_preprocessing.executed.ipynb`)
Date: 2026-09-25 | Mode: EDA on **full data** (Polars lazy scans; GT explode + joins over 7.6M pairs) + normalization **executed per-country (40k samples/country)**.
`FULL_RUN=true` runs the identical code chunked over every row → full per-country parquet. Companion: `first_preprocessing.md` (step/impact table).

## 1. What was done, in order

1. Cloned `Nikunjgoyal07/Amazon-ML-Challenge-2026`, created/pushed branch `eda-preprocessing`.
2. Built `01_eda_preprocessing.ipynb` from scratch via `build_notebook.py` (nbformat-generated valid JSON), 8 cells: paths → EDA basics → EDA structural → true-pair similarity → normalization → per-country run → verify.
3. Bugs found and fixed during execution: empty-`DATA_ROOT` resolving to repo dir; eager-vs-lazy `.collect()`; legal-only names ("Private") stripping to empty core (now kept as core, matching the spec example); soundex vowel handling.
4. Executed headlessly (`nbconvert --execute`): all cells PASS. Outputs in `processed/` (gitignored).

## 2. Data inventory (measured)

| File | Rows |
|---|---|
| train S1 / S2 / S3 / GT | 2,206,821 / 5,034,616 / 5,285,603 / 2,206,821 |
| test S1 / S2 / S3 | 1,732,544 / 4,887,273 / 5,082,316 |

Schema: `entity_id | business_name | business_address | country` (verified); prefixes S1-/S2-/S3- align with **0 mismatches** in all 6 files.

## 3. EDA findings (all full-data unless noted)

- **Country.** Train S1: US 1,323,633 + India 883,188. S2 train: US 3,016,817 + India 2,017,799. S3 train: US 3,170,056 + India 2,115,547. Test adds France: S1 259,452; S2 703,378; S3 731,615.
- **Nulls.** S1 train/test: 0/0. S2 train addr null 168,967 (3.36%), S3 train 175,916 (3.33%); test S2 129,408, S3 136,098. Names never null.
- **Cardinality.** Mean 3.461, max 11, singletons 5.58% (123,247); peak at 3 (530,841) / 4 (484,115). S2 share **48.4%** (3,693,619 pairs) / S3 **51.6%** (3,944,746).
- **Structure (verified, not assumed).** Each S2/S3 reused at most **1×** (`reuse_max=1` over 7,638,365 pairs) → distractors **25.99%**. **0 cross-country mismatches / 7,638,365 checked** → per-country blocking is safe.
- **Lengths (20k/source).** Name char median 24–25, token median 4; address char median 37–42.
- **Completeness/country (30k/source).** US PIN-present ~11–12.5%, India ~0.4–1.2% (PIN regex misses spaced Indian PINs — known heuristic gap); null-addr ~3–4% in S2/S3 both countries.
- **True-pair similarity (20k pairs, rapidfuzz).** token_set mean **84.6** (66.1% ≥90, 13.3% <60), fuzz mean 78.4, address Jaccard mean **0.576** → names carry most signal; ~13% hard pairs need address/embeddings. 15-pair noise catalog printed in notebook.
- **Vocabulary (100k S1).** Top tokens `& / and / care / associates / group / center / services…`; top legal `limited / llc / inc / ltd / llp / corp / co…`.

## 4. Normalization (all preprocessing.md §3 steps)

`base_clean` (NFKC→accent-strip→lower→junk→collapse) → honorific strip **+ flag** → legal split (EN/IN + hand-listed FR: sarl/sas/sa/eurl/sci/groupe) → domain-stem → **`name_lat` via anyascii (ISC)** → **soundex `phonetic_code`** → address house/PIN parse → PO-box drop → EN+FR abbreviation expansion → **`city`/`state_code` heuristic** (US codes/names, IN codes/names, 5-entry native-script map: Tamil Nadu, Maharashtra, Delhi, West Bengal, Karnataka) → `street_tokens` + `empty_addr/missing_pin/missing_city/landmark_flag`.
`#12`-style houses supported; legal-only names kept as core. **13 asserts PASS**, incl. Devanagari-anyascii, FR `Bd→boulevard`/`R.→rue`, `SARL→legal`, landmark, junk+accent combo.

## 5. Outputs (`processed/`)

Per-country parquet: `train_s1_{US,India}`, `test_s1_{US,India,France}`, `train_s2/s3_{US,India}` (40k each in sample mode) + `eda_summary.json` (all §3 numbers) + `processed_counts.json`. Schema frozen at 21 columns: raw 4 + `name_norm, core_name, legal_form, domain_stem, name_lat, phonetic_code, has_honorific, addr_norm, street_tokens, house_no, zip_pin_cp, city, state_code, empty_addr, missing_pin, missing_city, landmark_flag`.
Verified: `High Point/NC`, `Tahlequah/OK`, `Phoenix/AZ` extracted; `<< Team Ecole → team ecole`, `ZNB Club SARL → znb club + sarl`. Known heuristic limits: bare street numbers can read as PIN (`17560 Ellis Road → pin 17560`); India spaced PINs under-detected.

## 6. How to reproduce / run on Kaggle

Upload TSVs preserving `train/`+`test/`, set `DATA_ROOT=/kaggle/input/<dataset>`, run top-to-bottom (`FULL_RUN=false` validates in minutes; `FULL_RUN=true` + re-run processing cell for full per-country parquet). Blocking consumes `processed/train_s1_{US,India}.parquet` etc.
