# First Preprocessing — EDA + Normalization (executed)

Branch: `eda-preprocessing` | Notebook: `01_eda_preprocessing.ipynb` (executed copy: `01_eda_preprocessing.executed.ipynb`)
Run mode: EDA on **full data** (lazy scans) + normalization validated on **4 x 100k samples** (`FULL_RUN=true` on Kaggle does full chunked run).
Outputs: `processed/eda_summary.json`, `processed/processed_counts.json`, `processed/*_sample_processed.parquet` (100k each: train S1/S2/S3 sample, test S1 sample).

## Measured EDA facts (from this run)

- Rows: train S1 **2,206,821** / S2 **5,034,616** / S3 **5,285,603**; test S1 **1,732,544** / S2 **4,887,273** / S3 **5,082,316**.
- Country: train = US **1,323,633** + India **883,188**; test = US **663,106** + India **809,986** + **France 259,452** (unseen in train).
- Nulls: S1 name/addr **0**; S2 addr null **168,967 (3.36%)**; S3 addr null **175,916 (3.33%)**; names never null.
- Ground truth: mean matches/S1 **3.461**, max **11**, singletons **5.58% (123,247)**; distribution peaks at 3 (530,841) and 4 (484,115).
- Lengths (50k sample): name median **24** chars, address median **41** chars; 0 empty addresses in S1 sample.
- Verified examples: `"B+ Retail Inc" -> core "b+ retail" + legal "inc"`; `"H.no 780, Voltaire, 75011 Paris" -> house "780", pin "75011"`.

## Step table

| Step taken | What it does | Why we did it | How it helps us | Impact score |
|---|---|---|---|---|
| 1. TSV load with `sep="\t"` + lazy `scan_csv` row counts | Reads tab-separated files without silent single-column corruption; counts all 7 files without loading into RAM | Addresses/ID-lists contain commas; full files are 2.35 GB and don't fit naive pandas loads | Correct row counts (2.2M/5M/1.7M) ground blocking index sizes and chunk plan | High (5/5) |
| 2. Country distribution scan (train vs test) | `group_by(country).len()` on S1 train + test | Test adds France (259k) unseen in train; hard-coded `{US, India}` logic would silently drop/mishandle it | Forces country-agnostic features and per-country blocking including FR | High (5/5) |
| 3. Null audit (name/address per source) | Counts null `business_name`/`business_address` in S1/S2/S3 | S2/S3 have ~3.3% null addresses vs S1 zero — matcher can't assume address always present | Adds `empty_addr`/`missing_pin` flags so model down-weights missing address instead of crashing | High (4/5) |
| 4. Ground-truth cardinality analysis | Splits `matched_entity_ids`, computes mean/max/singleton% + full 0–11 distribution | Singletons are only 5.58% (not ~40%) and mean is 3.46 — singleton-gating alone can't win; multi-match ranking matters | Sets decision-layer design: one-owner assignment + expected-F0.5 prefix, not just empty/non-empty | High (5/5) |
| 5. Length + noise sampling (50k + 5 raw rows) | Median name/address lengths; eyeball raw pairs for junk patterns | Quantifies difficulty (short names = hard) and catalogs noise (`***`, `[Services]`, accents, `H.no`, legal suffixes) | Each observed noise maps to one normalization rule below — no blind rules | Medium (3/5) |
| 6. Unicode NFKC + accent strip + lowercase (`base_clean`) | Canonicalizes `Prívate -> private`, French accents, case; collapses whitespace | Same business written with accents/case/punctuation across sources would otherwise never match | Raises name/address overlap for blocking recall; required for FR generalization | High (5/5) |
| 7. Junk-wrapper strip (`***`, `>>`, `--`, `<<`, `##`, `[Services]`, `(ID:n)`, `<NULL>`) | Regex-removes synthetic wrappers/prefixes | Observed as systematic noise; tokens pollute TF-IDF and embeddings | Cleaner `core_name`/`addr_norm`, fewer false blocking misses | High (4/5) |
| 8. Honorific strip (`Smt/Shri/Mr/Mrs/Dr`) | Removes leading honorifics, keeps behavior consistent | Honorifics appear on one source only, breaking token overlap | Prevents `shri`/`smt` from creating false distinctions between same business | Medium (3/5) |
| 9. Legal-suffix split (`core_name` vs `legal_form`, incl. FR SARL/SAS/EURL/SCI/Groupe) | Separates `guru infotech` from `pvt ltd`; FR forms hand-listed (domain knowledge, not external data) | Legal suffix is inconsistent across sources (`Pvt` vs `Private`, missing vs present) | Enables `legal agree/conflict` feature instead of noisy raw token match; unit-tested | High (4/5) |
| 10. Domain-stem extraction (`eelegal.com -> eelegal`) | Detects name-is-a-domain, stores stem | Some records replace business name with a website; plain token match fails completely | Stem-vs-core comparison recovers these otherwise-unmatchable pairs | Medium (3/5) |
| 11. Address parse: house_no + ZIP/PIN/CP + abbreviation expansion + PO-box drop | Extracts `house_no`, 5/6-digit codes, expands `Rd->road, Bd->boulevard, R.->rue`, drops `PO Box/Unit/H.no` wrappers | House number + PIN is the strongest address signal but buried in noise/abbreviations; `Evoavi`-style pairs match on address only | Gives exact-match keys for blocking (`house+pin`) and `match/conflict/missing` tri-state features | High (5/5) |
| 12. Missing-field flags (`empty_addr`, `missing_pin`) | Binary flags per record | ~3.3% of S2/S3 lack addresses; model must distinguish "missing" from "different" | Lets LightGBM learn missingness handling; avoids imputing fake similarities | Medium (3/5) |
| 13. Unit tests (6 asserts: legal split, domain stem, house/pin, empty addr) | Fails fast if any normalization rule regresses | Preprocessing bugs silently cap blocking recall (missed true matches are unrecoverable) | All PASS before any data run; safe to scale to `FULL_RUN=true` on Kaggle | High (4/5) |
| 14. Chunked parquet write (200k chunks; 100k samples locally) | Applies steps 6–12, writes `*_processed.parquet` + `eda_summary.json` + `processed_counts.json` | Full S1 is 2.2M rows — one-shot UDF would OOM/time out; samples validate pipeline in minutes | `processed/` is the frozen contract for notebook 02 (blocking); row preservation verified (100k in = 100k out) | High (5/5) |
| 15. Before/after verification (5 rows: raw -> norm + house/pin) | Prints `business_name -> core_name + legal` and `address -> addr_norm + house/pin` | Proves each rule fired correctly on real data (e.g. `B+ Retail Inc -> b+ retail + inc`) | Human sign-off that normalization helps rather than destroys signal | Medium (3/5) |

## How to reproduce / run on Kaggle

1. Upload TSVs as a Kaggle dataset (keep `train/` + `test/` layout).
2. Open `01_eda_preprocessing.ipynb`, set env `DATA_ROOT=/kaggle/input/<dataset>` (or `KAGGLE_MODE=true`).
3. Run top-to-bottom with `FULL_RUN=false` first (validates in minutes), then set `FULL_RUN=true` and re-run cell 4 for full parquet.
4. Next: blocking notebook consumes `processed/train_s1_processed.parquet` (+ S2/S3 full once generated).
