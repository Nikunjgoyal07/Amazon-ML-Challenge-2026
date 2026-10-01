# Making a submission

The full run uses three Kaggle notebooks, one after the other. You then download the two result
files, validate them on your computer, and upload `matching_results.tsv` to the portal.

```
01_eda_preprocessing.ipynb  (FULL_RUN=true)   →  processed/            cleaned train + test records
02_full_e5_buckets.ipynb                       →  embeddings_full/      top-30 + extra candidates per S1 (train + test)
03_full_lightgbm_submission.ipynb              →  submission/           matching_results.tsv, candidate_pairs.tsv
04_rethreshold.ipynb   (optional)              →  rethreshold/          same matches with another France cutoff
```

The four notebooks are in [`submission-4/`](../submission-4/). How it works: [SOLUTION_WRITEUP.md](SOLUTION_WRITEUP.md),
and notebook by notebook in [pipeline/](pipeline/).
What each output contains: [DATA_AND_OUTPUTS.md](DATA_AND_OUTPUTS.md).

## Versions

The current notebooks are the **final pipeline**: public leaderboard **0.967** with
`COUNTRY_THRESHOLDS = {"France": 0.9}` in 03; cross-validation 0.9796 on India + US. Earlier submissions
(0.935, 0.960) and what changed between them: [SOLUTION_WRITEUP.md §1 and §10](SOLUTION_WRITEUP.md).

---

## Step 1: notebook 01 on all data (skip if already done)

If your `processed/` output already has `test_s2_*` and `test_s3_*` files (the run used for
version 1 did), reuse it and skip this step.

- **Inputs:** the competition data.
- **Settings:** `FULL_RUN=true`, e.g. add `import os; os.environ["FULL_RUN"] = "true"` as the first
  cell.
- **Accelerator:** none needed (CPU).
- **Check:** the `DONE:` line lists `s2te` and `s3te` with France, India and US. Full-size row counts:

  | File | Rows |
  |---|---|
  | test S1 | 1,732,544 (France 259,452 · India 809,986 · US 663,106) |
  | test S2 | 4,887,273 (France 703,378 · India 2,312,565 · US 1,871,330) |
  | test S3 | 5,082,316 (France 731,615 · India 2,405,000 · US 1,945,701) |

## Step 2: 02_full_e5_buckets.ipynb

- **Inputs:** the competition data and the output of step 1.
- **Accelerator:** GPU T4 x2. **Internet:** on, to download `intfloat/multilingual-e5-small` and
  install `anyascii`.
- **Run all.** The first cell should print `device: 2 x Tesla T4 | fp16: True | batch: 1024 | ...
  text mode: fixed`.
- **What it does:**
  1. Transliterates Indian-script names and addresses to Latin letters.
  2. Encodes all ~24M train and test records on both GPUs.
  3. Keeps the top 30 candidates of every S1, adds the extra candidates (reverse search, address key,
     name 3-grams), and saves them in `embeddings_full/multilingual-e5-small/buckets/`.
- **Time:** estimate 1.5–2 hours (the extra sources add roughly 30–45 minutes, mostly the name search). It prints texts/s while encoding. If the session stops,
  re-running in the same session continues after the last finished bucket file.
- **Check before moving on:**
  - five bucket files: `train_India`, `train_US`, `test_France`, `test_India`, `test_US`
  - each bucket prints `new pairs per S1 by source`: a few per source (France's address key may add
    more, crowded cities share addresses)
  - the last table: India's "ceiling macro F0.5" for `e5 top-30` should be clearly above version 1's
    0.946 (about 0.97 on the local test bed), and `e5 top-30 + all extra (saved)` higher still
    (about 0.974 on the test bed without the reverse search). The US should stay around 0.994.
  - to judge a source, compare its row with `e5 top-40`/`e5 top-50` at similar "pairs per S1". A
    source that adds pairs without raising the share of true matches can be dropped from
    `EXTRA_SOURCES`.
  - `e5 top-30 + number` (India) and `e5 top-30 + empty` / `+ empty_rev`: on submission 2's full train
    buckets, the number key found about 13% of India's misses for ~4 pairs per S1, and the empty-address
    search about 40% of the misses whose candidate has no address ("... empty-address candidates" column)
    for ~2 pairs per S1.
  - India's "... Indian-script candidates" vs "... Latin-script candidates": on the test bed, 17% of
    Indian-script true matches were missing from the buckets vs 4% for Latin script. If the gap is
    still large at full size, a better transliteration is the obvious next step. IndicXlit was tried
    and did not help ([experiments/INDICXLIT.md](experiments/INDICXLIT.md)).

## Step 3: 03_full_lightgbm_submission.ipynb

- **Inputs:** the competition data, the output of step 1 and the output of step 2.
  Optional: upload `Data/student_resource/utils/validate_submission.py` as a small dataset and
  attach it too. The notebook then runs the official validator itself.
- **Accelerator:** not needed (CPU work). **Internet:** on, to install `anyascii` if missing.
- **Run all.**
- **What it does:**
  1. Trains LightGBM on 200,000 train S1 per country with 85 features each.
  2. Cross-validates and picks the cutoff.
  3. Predicts every test pair (30 e5 candidates per S1 plus the extra ones) and writes both files.
  4. Checks the files.
- **Time:** estimate 3–4 hours. Version 1 took about 1 hour for test prediction with 20 candidates
  and 44 features; version 2 has 50% more pairs and heavier features.
  Version 3 adds about 15% more pairs (the extra candidates) and the typo-tolerant features, roughly
  30–40 minutes more (estimate). The number key and the empty-address search add about 15% more pairs in
  India and 4% in the US.
- **Check before downloading:**
  - **Cross-validation table:** the row "LightGBM p ≥ t + one owner (used for the submission)" is
    the expected score for India + US. Version 1 had 0.946; expect about 0.96.
  - **The France check (test table vs training table):** France has no labels, so this is the only
    view of it.

    | | S1 with a predicted match | Matches per matched S1 |
    |---|---|---|
    | Version 1: France | 95.8% | 3.67 |
    | Version 1: US | 94.1% | 3.52 |
    | Version 1: India | 92.1% | 3.23 |

    France predicted more than the others, a sign of lookalike matches. With version 2's new
    comparisons, France's numbers should move toward the US's.
  - **Last cell:** `matching_results.tsv: OK`, `candidate_pairs.tsv: OK`, `matches that are not
    candidates: none`.

Download `submission/output/matching_results.tsv` and `submission/output/candidate_pairs.tsv`.

## Step 4: validate on your computer

Put the two files in `Data/student_resource/output/`, then run from `Data/student_resource/`:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

It must print `PASS`. `--check-ids` also checks that every ID exists in the test files. It uses a few
GB of memory; drop `--candidate` if it runs out.

## Step 5: submit

Upload `matching_results.tsv` in the portal.

## Step 6: tune France on the leaderboard (optional, recommended)

France can't be tuned on data (no labels), but the leaderboard can compare cutoffs:

1. Attach 03's output and the competition data to `04_rethreshold.ipynb`, keep
   `COUNTRY_THRESHOLDS = {"France": 0.80}` and run it (minutes).
2. Validate `rethreshold/matching_results.tsv` together with 03's `candidate_pairs.tsv`, and submit
   it.
3. Compare with step 5's score. Only France changed, so the difference comes from France alone.
   - **Higher:** try 0.85 or 0.90.
   - **Lower:** try a France cutoff below 03's tuned one, or keep 03's file.

## Final zip

The zip needs:
- `output/` with both TSV files
- `code/business_entity_resolution/` with the notebooks (`src/`), a README and requirements
- the filled-in `Documentation_template.md`

[SOLUTION_WRITEUP.md](SOLUTION_WRITEUP.md) covers most of what the template asks for.

---

## If something goes wrong

| Problem | Fix |
|---|---|
| 02: `WARNING: no test_s2_* / test_s3_* files` | step 1 ran with the old 01. Re-run the updated 01 with `FULL_RUN=true`. |
| 02: only `1 x Tesla T4` | set the accelerator to GPU T4 x2 |
| 03: out of memory | lower `TRAIN_S1_PER_COUNTRY` (e.g. 100000) or `PAIRS_PER_CHUNK` (e.g. 1000000) in the first cell |
| 03: runs too long | set `CANDIDATES_PER_S1 = 20` (about a third less work; the local score drops from 0.967 to 0.964) |
| 03: `[LightGBM] [Fatal] CUDA Tree Learner was not enabled` | harmless: the notebook is testing for GPU support and falls back to the CPU |
| Validator: `required S1 entity(ies) missing` | should not happen (every S1 of `test_source1.tsv` gets a row). Check that step 3 read the right `test_source1.tsv` (printed in its first cell). |
| You want another cutoff | notebook 04 re-writes the matches from `submission/test_probabilities/` without recomputing |

## Tested locally

- **End to end on a 1% sample** (two simulated GPUs): 01 → 02 (full) → 03 (full) → 04.
  `validate_submission.py --check-ids` printed `PASS` on the files from 03 and 04.
- **Accuracy on a dense regional test bed:** all businesses of Arizona + Utah, Kerala + Telangana
  (train, labeled) and Pays de la Loire (France, test). Each change was kept only if it scored
  better there:

  | Change | India + US cross-validation |
  |---|---|
  | Version 1 | 0.9427 |
  | + Indian-script transliteration and 13 new comparisons | 0.9644 |
  | + 30 candidates per S1 | **0.9667** |
