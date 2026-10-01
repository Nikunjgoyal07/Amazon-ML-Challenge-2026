# Ensemble matcher: LightGBM + XGBoost + CatBoost + neural network (`experiments/03c_ensemble.ipynb`)

**What it does:** 03's matcher with three more models next to LightGBM, and two ways of blending them. Everything
else is identical to 03: buckets, the 85 features, the 200,000 training S1 per country, the 3 folds (same seed), the
decision rule and the submission files. Every model is compared on exactly the same held-out pairs.

**Status:** smoke-tested locally on a small sample, where it runs end to end. It has not yet been run at full size.

**Expectation:** modest. The 85 features limit the model more than its size does: 127 leaves instead of 63 gave
+0.0001. A second learner on the same columns tends to make the same mistakes. The interesting number is the
country-transfer table (France's stand-in), where models can differ more.

---

## The four models

| Model | Settings | Hardware |
|---|---|---|
| LightGBM (`lgbm`) | 03's: 63 leaves, learning rate 0.1 | as 03 |
| XGBoost (`xgb`) | depth-8 trees, learning rate 0.1, 80% of rows and features per tree | first GPU |
| CatBoost (`cat`) | depth-8 symmetric trees, learning rate 0.1 (`gpu_ram_part=0.6`, so XGBoost's memory cache still fits) | first GPU |
| Neural network (`nn`) | 3 layers (512, 256, 128) with batch norm, SiLU and dropout; AdamW, one-cycle learning rate, `NN_EPOCHS`=6 passes | second GPU |

**Rounds:**
- **Boosting models:** stop 50 rounds after the held-out fold stops improving. The final model trains for 1.1 × the
  folds' mean best round count, as in 03.
- **Network:** trains the same number of passes in cross-validation and in the final model.

**Network inputs:** trees don't care about the scale of a feature; a network does. Each feature is mapped to its
rank among the training pairs (-1 to 1), and empty features become 0 plus a 0/1 "is empty" column. Two details
make it train well on any data size:
- the batch is at most 8,192, but small enough for about 200 steps per pass;
- the output starts at the share of true pairs instead of 50%.

In the local test, this brought its held-out log-loss from 0.056 to 0.002-0.004, the trees' level.

## The blends

| Option | How |
|---|---|
| `mean` | average of the models' probabilities |
| `stack` | logistic regression on the models' log-odds. For the comparison it is fitted on the other folds' held-out predictions, so its own score is honest too. For the test it is fitted on all of them (at most 2M rows) |

## Choosing what to submit

The comparison table shows each model and blend with:
- **cutoff and macro F0.5:** each option gets its own cutoff;
- **gain vs lgbm ± 2 SE:** the difference from LightGBM alone, with two standard errors. The difference is paired
  over S1, since every option is scored on the same S1;
- **S1 scored differently:** how many S1 get a different score than with LightGBM;
- **held-out log-loss**, and the correlation between the models' predictions.

`ENSEMBLE=auto` (default) picks the best option **only if its gain is larger than its ± 2 SE**; otherwise it keeps
LightGBM. A gain within noise isn't worth giving up the probabilities the France cutoff was tuned on.
`ENSEMBLE=stack` / `mean` / `xgb` / … forces a choice. That is worth doing if the transfer table shows a clearly
better transfer to the unseen country.

**Noise is real here:** LightGBM on the CPU isn't perfectly repeatable. In two local runs on identical data, one
fold stopped at 97 and 106 rounds. Differences of 0.0001-0.0003 mean nothing.

## Running it

- **Where:** Kaggle, GPU T4 x2, the same inputs as 03.
- **Time (estimate):** 03's ~5 h plus 1-2 h, because each extra model is trained 6 times (3 folds, 2 country
  transfers, final).
- **Shortcuts:**
  - `CROSS_COUNTRY_CHECK=false` saves about a third of the extra time.
  - `PREDICT_TEST=false` stops after the comparison and the transfer, with no submission files.
- **Other settings:** `MODELS` (default `lgbm,xgb,cat,nn`), `ENSEMBLE`, `NN_EPOCHS`, and `COUNTRY_THRESHOLDS` as JSON
  (e.g. `{"France": 0.9}`).
- **Outputs:** the same files as 03, plus:
  - `model/model_comparison.csv`;
  - one column per model (`p_lgbm`, `p_xgb`, `p_cat`, `p_nn`) in `oof_pairs.parquet` and `test_probabilities/`;
  - the models used (`xgb_matcher.json`, `cat_matcher.cbm`, `nn_matcher.pt`).

**If a blend is chosen, retune France with 04.** A blend's probabilities are not LightGBM's, so 0.9 means something
else.

## Local smoke test (4,000 S1, so the scores are not meaningful)

- **End to end** in about 2.5 minutes: cross-validation, blends, transfer, final models, test prediction, file
  checks.
- **LightGBM's held-out predictions were bit-identical to 03's** on the same data. That confirms the same S1, folds
  and features.
- **Every option landed within ±0.001 of LightGBM**, inside the noise, so `auto` kept LightGBM.
