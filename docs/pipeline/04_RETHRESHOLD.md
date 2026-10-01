# 04 · Other cutoffs without recomputing (`submission-4/04_rethreshold.ipynb`)

**What it does:** rewrites `matching_results.tsv` from the probabilities 03 saved, with a different cutoff for one
or more countries. It takes minutes instead of 03's hours.

```
03's submission/test_probabilities/<country>.parquet + model/matcher_config.json  ──►  04  ──►  rethreshold/matching_results.tsv
```

**Why:** France has no labels, so its cutoff can't be tuned on data. The leaderboard can do it:
1. Submit 03's file.
2. Submit one from 04 with a different France cutoff.
3. The difference in the public score shows which way France should go.

Change only one country per submission, so the difference can be attributed. The final submission used France
0.9; the other countries kept 03's tuned cutoff (about 0.7).

## Cell by cell

| Cell | What it does |
|---|---|
| 1 | **The setting to edit**, `COUNTRY_THRESHOLDS = {"France": 0.80}`. Also finds 03's output (`SUBMISSION_DIR`, `submission/`, or any Kaggle input) and reads 03's tuned cutoff from `matcher_config.json` |
| 2 | Per country: each S2/S3 record goes to its highest-`p` S1 (one owner), and pairs with `p ≥` the country's cutoff are kept. It prints the share of S1 with a match and matches per matched S1 |
| 3 | Writes the file (every test S1 once, in `test_source1.tsv` order) and checks it: one row per S1, no duplicate IDs, only S2/S3 IDs, no record used twice |

`candidate_pairs.tsv` doesn't change, because the candidates are the same. Submit 04's `matching_results.tsv`
together with 03's `candidate_pairs.tsv`, and run `validate_submission.py` on that pair.

## Reading the printed table

Compare each country's "S1 with a predicted match" and "matches per matched S1" with the training data's rates,
which 03 prints: about 94% of S1 have a match, with about 3.5 matches each.

- **Much higher than training** (France has been): probably false matches between lookalikes. Raise the cutoff.
- **Much lower:** the cutoff is too strict for that country.

## When 03's probabilities change

A cutoff only means something for the model that produced the probabilities. After rerunning 03 with new features,
new buckets (e.g. a fine-tuned embedding model) or a blend of models (`experiments/03c_ensemble.ipynb`), tune France
again: 0.9 for one model is not 0.9 for another.
