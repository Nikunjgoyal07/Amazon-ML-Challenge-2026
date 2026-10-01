# 01 · Preprocessing (`submission-4/01_eda_preprocessing.ipynb`)

**What it does:** measures the raw data (EDA), then cleans every record and writes one parquet file per split,
source and country to `processed/`. Every later notebook reads these files.

```
raw TSVs (train/test × source 1/2/3)  ──►  01  ──►  processed/<split>_<source>_<country>.parquet
                                                      + eda_summary.json, processed_counts.json
```

**Run:** on Kaggle (CPU is enough) or locally, top to bottom. For a submission, `FULL_RUN=true`, so every row is
processed. The default (`false`) processes a sample for experiments; the EDA cells always read the full files.

Why the cleaning is done this way, and what it gained: [SOLUTION_WRITEUP.md §5](../SOLUTION_WRITEUP.md).

---

## Cell by cell

| Cell | What it does | Key names |
|---|---|---|
| 1 | Settings, locating the competition files, sampling helpers | `find_data_file`, `FULL_RUN`, `SAMPLE_*`, `stratified_sample` |
| 2 | EDA basics: row counts, columns, ID prefixes, empty fields, matches per S1 | writes `eda_summary.json` |
| 3 | EDA structure: S2/S3 share, "one owner" check, distractors, cross-country check, lengths, French examples, frequent words | `pairs` (the exploded ground truth, reused in cell 6) |
| 4 | Similarity of 20,000 true pairs, and 15 printed example pairs (the noise catalogue) | `eda["sim_20k"]` |
| 5 | **The cleaning functions**, with unit tests | `normalize_name`, `normalize_address`, `extract_city_state`, `bare_house_no` |
| 6 | Runs the cleaning on every file (or a sample) and writes the parquet files | `normalize_frame`, `process_full_per_country`, `process_sample_per_country`, `SPEC` |
| 7 | Spot checks: before/after examples (US and France) and the final column list | |

---

## What the EDA measured (full data)

| Fact | Value | Consequence for the design |
|---|---|---|
| Rows (train) | S1 2,206,821 · S2 5,034,616 · S3 5,285,603 | |
| Rows (test) | S1 1,732,544 (France 259,452) · S2 4,887,273 · S3 5,082,316 | France exists only in test |
| Matches per S1 | mean 3.46, max 11; 5.58% have none (singletons) | |
| Reuse | every S2/S3 record belongs to **at most one** S1 | the "one owner" rule in 03 |
| Distractors | 25.99% of S2/S3 records match nothing | the main source of false matches |
| Cross-country matches | 0 of 7,638,365 | every search runs per country |
| Matches by source | S2 48.4%, S3 51.6% | |
| Empty addresses | S1 0%, S2/S3 about 3.3% | a dedicated search in 02 |
| True-pair similarity (20K pairs) | name token-set 84.6 on average (66% ≥ 90, 13% < 60); address word overlap 0.58 | names carry most of the signal; about 13% of pairs need the address or embeddings |

---

## The cleaning, step by step (cell 5)

### Names: `normalize_name(raw)`

1. **`base_clean`**, a common first pass for names and addresses:
   - Unicode NFKC normalization, then accents removed (`Prívate` → `private`) and lower case.
   - Junk wrappers removed: `***`, `>>`, `--`, `<<`, `##`, `[services]`, `(id: 1072)`, `<null>`.
   - Spaces collapsed, and leading or trailing `, . -` trimmed.
2. **Honorific** (`smt`, `shri`, `mr`, `mrs`, `miss`, `dr`) removed and flagged in `has_honorific`.
3. **Website names:** `projectsmanipalindia.com` gives `domain_stem = projectsmanipalindia`.
4. **Legal form:** words from `LEGAL` are split off into `legal_form`. These cover English and Indian forms (`pvt`,
   `ltd`, `llc`, `llp`, `inc`, …) and French ones (`sarl`, `sas`, `sasu`, `eurl`, `sci`, `ei`, `cie`, …). The rest
   is `core_name`; a name that is only a legal word ("Private") keeps it as its core.
5. **`name_lat`:** `core_name` transliterated to Latin letters with `anyascii`.
6. **`phonetic_code`:** Soundex of the first core word.

| Raw | `core_name` | `legal_form` | other |
|---|---|---|---|
| `*** Prívate *** [Services] (ID: 1072)` | `private` | | |
| `GURU INFOTECH PVT. LTD.` | `guru infotech` | `pvt ltd` | |
| `Thermal & Fils SASU` | `thermal & fils` | `sasu` | |
| `Shri Ram Traders` | `ram traders` | | `has_honorific = 1` |
| `projectsmanipalindia.com` | `projectsmanipalindia.com` | | `domain_stem = projectsmanipalindia` |

### Addresses: `normalize_address(raw)`

1. `base_clean`, as for names.
2. **House number:** first a marker (`h.no 780`, `plot 12`, `door 5`, `#12`). Without one, `bare_house_no` takes the
   first plain 1-4 digit number of the first comma part that has one. That lifts house-number coverage from almost
   0% to about 99% (France) and 91% (US).
3. **Postcode** (`zip_pin_cp`): a 6-digit number (India PIN), otherwise a 5-digit one (US ZIP, French CP).
4. **Landmark flag:** `near`, `opp`, `temple`, `station`, `mall`, … and the French `gare`, `eglise`, `hopital`, ….
5. **Noise removed:** `p.o. box …`, `unit 5`, `door no`.
6. **Abbreviations expanded:** `rd` → road, `st` → street, `ave`/`av` → avenue, `blvd`/`bd` → boulevard, `r` → rue,
   `cir` → circle. The result is `addr_norm`, and its sorted unique words are `street_tokens`.
7. **City and state:** `extract_city_state` reads the last comma part. It recognizes:
   - US state codes and names
   - Indian state names and codes (`MH` → maharashtra)
   - the 18 French regions (longest name first: "pays de la loire" before "loire")
   - Indian state names written in their own script (`தமிழ்நாடு` → tamil nadu)

   The part before the state is the city. If no part ends with a state (the parts were reordered), every part is
   searched for a region.

Examples (from the unit tests):

| Raw address | Result |
|---|---|
| `H.no 780, R. Voltaire, 75011 Paris` | house 780, postcode 75011, `r.` → rue |
| `175 Boulevard Bd Saint-Germain, Paris` | house 175, "boulevard" in `addr_norm` |
| `#12, Lake Town, Kolkata` | house 12 |
| `20 Rue Parmentier, Dunkerque, Hauts-de-France` | city dunkerque, state hauts-de-france |
| `12 Rue de la Gare, Lille, Hauts-de-France` | landmark (gare) |

### Unit tests

Cell 5 ends with 18 `assert`s covering the cases above: junk + accents, legal forms (incl. SARL/SASU), website
stem, honorific, house number with and without markers, postcode, abbreviations, landmark, empty address, French
regions, and that `anyascii` handles Devanagari. Run the cell after any change; a failing assert stops the notebook
before hours of processing.

---

## The output files (cell 6)

One parquet per `(split, source, country)`, e.g. `train_s2_India.parquet`, `test_s1_France.parquet`. The six
inputs in `SPEC` are train S1/S2/S3 and test S1/S2/S3; the test S2/S3 files are needed for the submission.

| Column | Meaning | Used by |
|---|---|---|
| `entity_id`, `business_name`, `business_address`, `country` | the original record | 02, 03 |
| `name_norm` | cleaned full name | 02 (embedding text), 03 |
| `core_name` | name without legal form or honorific | 02 (name searches), 03 |
| `legal_form`, `domain_stem`, `name_lat`, `phonetic_code` | see above | 03 |
| `has_honorific` | 0/1 | 03 (flag) |
| `addr_norm`, `street_tokens` | cleaned address; its sorted unique words | 02 (embedding text), 03 |
| `house_no`, `zip_pin_cp`, `city`, `state_code` | parsed address parts | 03 |
| `empty_addr`, `missing_pin`, `missing_city`, `landmark_flag` | 0/1 flags | 03 (empty, landmark) |

Also written: `eda_summary.json` (every EDA number) and `processed_counts.json` (rows per file and the settings
used).

**Full run** (`process_full_per_country`): reads each file in chunks of 200,000 rows, cleans them, writes one
temporary file per country per chunk, then joins them per country.

---

## The known weakness: Indian scripts

`strip_accents` removes Unicode "combining marks". For Indian scripts, the vowel signs are combining marks, so
`name_norm` and `addr_norm` come out garbled for those records (`दिल्ली` loses its vowels). 01 was left as is.
Instead, 02 and 03 build the text of any Indian-script name or address from the **original** text, transliterated
with `anyascii` plus two fixes and a learned word map. See [02_CANDIDATES.md](02_CANDIDATES.md#text-given-to-the-model).

**Lesson:** a fix in 01 only matters if 02 or 03 actually read that column. 03 recomputes the house number from the
raw address itself, so 01's `bare_house_no` doesn't reach the model's `house_number_same` feature (it does reach
`house_no_same`).

---

## Sample mode (`FULL_RUN=false`)

For quick experiments, cell 6 processes a sample instead of every row:

| Setting | Default | Meaning |
|---|---|---|
| `SAMPLE_SIZE` | `40k` | S1 rows per country; each S2/S3 file keeps the full data's ratio to S1 (about 2.3×) |
| `SAMPLE_MODE` | `linked` | `linked`: random train S1, then **all their true matches** plus random other records (realistic pairs and distractors). `random`: random rows. `head`: first rows |
| `SAMPLE_DIVERSE` | `true` | every random draw is stratified by name type (Indian script / website / honorific / plain) and, for train S1, by match count, so rare cases are present even in small samples |
| `PROCESS_SPLITS` | `train,test` | `train` = skip the test files (experiments only) |
| `PROCESSED_OUT` | `processed` | output folder |

**Warning:** a sample has fewer lookalikes per S1 than the full data, so every score measured on it is too
optimistic (0.974 on a sample vs 0.946 at full size). Use samples to compare two versions with each other, never to
predict the leaderboard.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `FULL_RUN` | `false` | `true` for a submission: every row |
| `DATA_ROOT` | searched | folder with the competition TSVs (flat or `train/` + `test/`); on Kaggle found under `/kaggle/input` |
| sample settings | see above | only when `FULL_RUN=false` |

## Changing the cleaning

1. **Edit cell 5.** Add an `assert` for the case you are fixing.
2. **Check it:** run with `FULL_RUN=false` and look at cell 7's before/after examples.
3. **Rerun everything after it:** 01 with `FULL_RUN=true`, then 02 full and 03, because the buckets and features
   depend on these columns.
4. **Check the change reaches the model:** see the lesson above.
