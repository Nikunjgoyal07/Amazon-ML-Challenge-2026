# Fine-tuning the embedding model (notebook 02b)

This page explains `02b_finetune_e5.ipynb`: what it does, why, how the training works, and how we check that its
score is honest. It assumes you know the pipeline (01 → 02 full → 03 → 04), but not how neural networks are
trained. Technical terms are explained where they first appear; the glossary at the end collects them.

A sister notebook, [02c](EMBEDDING_PRETRAINING.md), builds an embedding model from scratch instead of adapting e5.

**Status:**
- **Training:** done on Kaggle (2 × T4, 33 minutes).
- **Evaluation:** the first run's evaluation crashed on a GPU memory setting, which is fixed. It needs one more run,
  which can reuse the trained model.

---

## 1. What an embedding model does in our pipeline

An embedding model turns a piece of text into a list of 384 numbers, a **vector**. Think of it as a point in space.
The model is built so that texts about the same thing land close together:

```
"sun energy pvt ltd | 12 mg road pune"      ─┐
"sun enrgy private limited | 12 m.g. rd pune" ├─ close together
"सन एनर्जी लिमिटेड | पुणे" (transliterated)   ─┘

"moon traders | 4 station road delhi"       ── far away
```

Closeness is measured by **cosine similarity**, a score up to 1.0; the higher, the more alike.

Our pipeline uses this twice:

1. **Finding candidates (02 full).** Every S1 looks up its 30 nearest S2/S3 records. Those, plus the five extra
   searches, form its **bucket**. A true match that doesn't reach the bucket is lost for good.
2. **Judging (03).** LightGBM's two strongest clues are built from these similarities:
   - `cand_margin` (48.6% of the model's gain): is this S1 the candidate's closest S1, and by how much?
   - `rev_rank` (25.5%): searching from the candidate's side, where does this S1 rank?

A better embedding model therefore helps twice: more true matches in the buckets, and sharper clues for the judge.

## 2. Why fine-tune

We use `intfloat/multilingual-e5-small`, a general-purpose model trained by its authors on web text in about 100
languages. It knows language well, but it has never seen our data's particular noise:
- `pvt` vs `private`, `rd` vs `road`
- typos (`Wcstgrove`)
- names replaced by a website (`vtseeds.com`)
- Hindi or Tamil names written in their own script
- lookalike businesses in the same street

The numbers show it:
- **Misses:** its top 30 misses 5.3% of India's true pairs (recall 0.9471) and 1.7% of the US's (0.9833), per
  02 full's train report.
- **Where the misses are:** mostly Indian-script names, records with no address, and lookalikes in crowded cities.

**Fine-tuning** means continuing e5's training on our own examples. Its idea of "similar" then moves toward the
competition's idea of "same business". The model keeps everything it knew and adjusts its numbers a little.

## 3. The training examples

Each training example (a **row**) has three texts:

| Part | What it is | Role |
|---|---|---|
| **anchor** | an S1 record | the "question" |
| **positive** | one of its true S2/S3 matches, from `train_ground_truth.tsv` | the right answer |
| **negative** | a lookalike from the S1's bucket that is *not* a match | a tempting wrong answer |

Real rows from the Kaggle run (anchor ‖ positive ‖ negative, texts cut at 70 characters):

```
India  vt seeds llp | blkno/surno:801 westport nr.sankalpsquare-3 shilaj dask
       ‖ vtseeds.com | blkno/surno:801 westport nr.sankalpsquare-3 shilaj daskr
       ‖ perfect services technologies private limited | westport near sankalp ...

India  sustainable agro pvt ltd | pl.no 32 devi illam thirupathi garden pedda
       ‖ sustainable agro pvt ltd | pl.no. 32, devi illam, thirupathi garden pe
       ‖ environment agri pvt ...

US     rippey beyond inc | 1301 lebanon pike nashville tn
       ‖ beyond rippey | 1301 lebanon pike nashville tn
       ‖ hernandezfederalbread.com | 1301-d lebanon pike nashville tn
```

The negatives are exactly the hard cases: same building, similar name, different business.

**How rows are chosen:**
- **Count:** up to 500,000 random true pairs per country (India and US), so 1,000,000 rows.
- **Which S1:** only S1 **outside** LightGBM's training set (section 7 explains why).
- **The negative:** drawn from the S1's 10 best-scored bucket candidates that are not its matches. A negative whose
  text is identical to the anchor's or the positive's is skipped, and an S1 with no usable lookalike gets a random
  record of its country.
- **The text:** built by the same functions as 02 full, copied unchanged: `query: name | address`, Indian scripts
  transliterated, the word map applied, cut at 128 tokens. The model is trained on exactly the text it will later
  embed.

## 4. How the model learns

### The "pick the right one" game

Training runs in **steps**. Each step takes a **batch** of 256 rows. For every anchor in the batch, the model
compares it with all 512 candidate texts of the step: the 256 positives and the 256 negatives. It then plays a
multiple-choice game: *which of these 512 is my match?*

```
anchor: "vt seeds llp | ... shilaj"
  candidates:  vtseeds.com | ... shilaj daskr        ← the right answer
               perfect services technologies ...     ← its lookalike
               (510 other texts from the batch)      ← easy wrong answers
```

The **loss** is a number that is low when the right answer gets the highest similarity by a clear margin, and high
otherwise. After each step, the model's 118 million internal numbers (its **parameters**) are nudged slightly in
the direction that lowers the loss. Over thousands of steps, "same business" texts move together and lookalikes
move apart. The technical name of this game is `MultipleNegativesRankingLoss`.

### The details that make it work on our data

| Detail | Setting | Why |
|---|---|---|
| **In-batch negatives** | automatic | the other rows' texts are free wrong answers: 510 per anchor per step |
| **Hard negatives** | 1 lookalike per row | easy wrong answers teach little; lookalikes are what LightGBM struggles with |
| **Scale** | 20 | multiplies the similarities before the game. Without it, 0.90 vs 0.85 looks like a near-tie; with it, a clear difference |
| **No duplicates** | on | a text never appears twice in a batch, so two matches of the same S1 never land in one batch, where one would count as a "wrong answer" for the other |
| **Same-country batches** | on | an Indian S1 competes against Indian records, as it does in the real search |
| **Learning rate** | 2e-5, first 10% of steps ramping up (**warm-up**) | small steps keep what e5 already knows |
| **Length** | 1 pass (**epoch**) over the 1M rows = 3,906 steps | enough for the loss to flatten (below) |

### What the run showed

| Step | Loss |
|---|---|
| 500 | 0.054 |
| 1,000 | 0.036 |
| 2,000 | 0.029 |
| 3,000 | 0.026 |
| 3,900 | 0.025 |

The loss falls fast, then flattens. That is the healthy shape: the model learned quickly what it could, and more
steps would add little.

## 5. Training on two GPUs

Kaggle gives two T4 GPUs with 14.6 GB of memory each.

**First attempt: one GPU, out of memory after 32 minutes.** Each step encodes 256 rows × 3 texts × up to 128
tokens. That used 13.6 of 14.6 GB, and a batch with longer texts pushed it over.

**What runs now: one process per GPU** (called DDP, distributed data parallel):
1. Each GPU encodes half the batch: 128 rows.
2. The two GPUs swap their encoded texts, so every anchor still plays against all 512 texts of the step, exactly as
   on one big GPU.
3. Each GPU works out how to nudge the model, the two proposals are averaged, and both copies of the model take the
   same step.

**Result:** 3,906 steps in 33.3 minutes, with half the memory per GPU.

**Safety nets:**
- **Fallback:** if the two-GPU run fails, the notebook retrains on one GPU with a memory-saving version of the same
  loss. It computes the game 64 rows at a time, with the same result.
- **Reuse:** `FINETUNED_MODEL=<folder>` skips training entirely and only evaluates an already trained model.

## 6. How we measure it

**Recall@k:** the share of true pairs whose record is among the S1's k nearest records. Example: an S1 has 4 true
records, and 3 of them are in its top 30. Its recall@30 is 3/4 = 0.75. We report k = 5, 10 and 30. 30 is what 02 full
keeps; 5 and 10 show whether matches also move toward the top, which sharpens LightGBM's clues.

**Full size, never a sample:**
- **Index:** each evaluated S1 searches **all** 4,133,346 India S2/S3 train records, as in the real pipeline.
- **Why:** samples inflate recall, because fewer lookalikes compete. We learned this the hard way (a model that
  scored 0.974 on a sample scored 0.946 at full size).

**Two groups of S1:**

| Group | Size | What it tells us |
|---|---|---|
| **unseen** | 20,000 S1 from LightGBM's set, with at least one match | **the honest number**: the model never trained on these S1 or their matches |
| **seen** | 5,000 S1 whose pairs were trained on | a memorization check: if "seen" is far above "unseen", the model learned those pairs by heart |

**Slices:** results are also split by the kind of true record: Indian-script name, empty address, Latin script with
an address. That shows where the gain comes from.

**Base vs fine-tuned:** both models run on the same S1, the same index and the same code. The base model's unseen
recall@30 should come out close to 0.947 (India); that doubles as a sanity check of the evaluation itself.

**Decision rule:**
- **Gain of +0.005 or more** at unseen recall@30 (India): worth running 02 full and 03 with the new model.
- **Smaller or negative:** keep e5. Fine-tuning then only adds risk (France, memorization).

## 7. Is the evaluation honest? (leak audit)

A **leak** would be the model being tested on examples it was trained on. The score would then look better than it
will on test. Here is everything the training touches, checked against what the evaluation uses:

| Question | Answer |
|---|---|
| Are the evaluation's S1 ever training anchors? | **No.** Training uses only S1 outside LightGBM's set; the evaluation's "unseen" S1 come from inside it. The notebook stops if any training pair involves one of them. |
| Are their true records ever training positives? | **No.** Each S2/S3 record belongs to at most one S1, so a record matched to an evaluation S1 can't be another S1's match. The notebook checks this too. |
| Can their true records appear as training *negatives*? | **Yes, occasionally.** A record can be a lookalike in another S1's bucket. That only teaches "this record is *not* that other business", which can't make it easier to find for its own S1. |
| Can another entity have exactly the same texts as an evaluation pair? | **Rarely.** A business listed twice can make an evaluation pair identical, as text, to a training pair. The notebook counts these (`evaluation pairs with a trained pair's exact texts`); expect 0 or a handful. |
| Is the "seen" group training data? | **Yes, on purpose.** It exists to measure memorization and is never the headline number. |
| Does the word map leak? | **Marginally, and equally for both models.** The word map (transliterated word → English word) was learned in 02 full from all train pairs, including the evaluation S1's. It prepares the text the same way for the base and fine-tuned models, so their comparison stays fair. Each mapping needs at least 20 supporting pairs, so no single S1 decides one. At most, the absolute Indian-script recall is a hair optimistic, and 03's cross-validation has always shared this. |

**Reproducing LightGBM's S1 exactly.**
- **How 03 draws them:** 200,000 random S1 per country, with seed 42, in the S1 order of its bucket files.
- **How 02b repeats the draw:** same seed, same order. When 02 full's buckets are attached, the notebook first checks
  that their S1 order is identical to 01's files; if they differed, the notebook would stop.

**One downstream caveat, outside this notebook.** The fine-tuned S1 are still in the train buckets that 03 learns
from, and the new model places their own matches closer to them than it will on test. When one of LightGBM's S1
competes with a fine-tuned S1 for a record, the contest is slightly easier in training than on test. 03's
cross-validation could therefore be a little optimistic. The seen-vs-unseen gap shows how big this effect can be:
small gap, small effect.

**Verdict:** the "unseen" number is not measured on training data. The only shared ingredient is the word map, which
treats both models the same.

## 8. Using the result

If the gain is worth it:
1. **Attach:** add 02b's output (`e5_finetune/`) to a new notebook as a dataset.
2. **Buckets:** run 02 full with `E5_MODELS=/kaggle/input/<dataset>/e5_finetune/e5-small-er`. The new buckets go to
   `embeddings_full/e5-small-er/`.
3. **Matcher:** run 03 full with `BUCKET_DIR=<that run>/embeddings_full/e5-small-er/buckets`, keeping its defaults
   (`TRAIN_S1_PER_COUNTRY=200000`, seed 42). Those defaults are what keep LightGBM on S1 the fine-tuning never saw.
4. **Compare:** 03's cross-validation against 0.9796 (base e5).
5. **France cutoff:** retune it with 04. The e5 scores change, so the old cutoff may no longer fit.

**France has no labels, so the effect there can't be measured directly.** The only proxy is to fine-tune on one
country and evaluate on the other: `FINETUNE_COUNTRIES=India`, `EVAL_COUNTRIES=US`. If fine-tuning on India hurts
the US, expect it to hurt France too.

## 9. Settings and files

| Setting | Default | Meaning |
|---|---|---|
| `FINETUNE_COUNTRIES` | `India,US` | countries whose pairs are trained on |
| `MAX_PAIRS_PER_COUNTRY` | 500,000 | training rows per country |
| `HARD_NEGATIVES`, `HARD_NEG_TOP` | on, 10 | lookalike negatives from the S1's 10 best non-matching candidates |
| `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `SCALE` | 1, 256, 2e-5, 20 | training |
| `TRAIN_GPUS`, `MINI_BATCH` | all, 64 | GPUs used; chunk size of the memory-saving loss (one-GPU fallback) |
| `FINETUNED_MODEL` | empty | reuse a trained model: evaluation only |
| `EVAL_COUNTRIES`, `EVAL_S1`, `SEEN_EVAL_S1` | India, 20,000, 5,000 | evaluation |
| `LGBM_SEED`, `LGBM_TRAIN_S1_PER_COUNTRY` | 42, 200,000 | must equal 03's settings |

| Output | Content |
|---|---|
| `e5-small-er/` | the fine-tuned model |
| `eval_recall.csv` | recall@5/10/30 by group and slice, base vs fine-tuned |
| `train_log.csv`, `finetune_config.json` | loss curve, settings, run summary |
| `lgbm_train_s1.txt`, `finetune_s1.txt` | LightGBM's S1 (never trained on) and the trained S1 |
| `train_e5.py` | the training script the notebook launched |

## 10. History of the notebook's runs

| Run | What happened | Fix |
|---|---|---|
| 1 | Out of GPU memory after 32 min: 256 rows × 3 texts on one T4 | two GPUs (one process each), plus the memory-saving loss for the one-GPU fallback |
| 2 | Training succeeded (33.3 min, both GPUs, shared negatives). The evaluation then crashed: "CUDA driver error: operation not permitted" | a GPU memory setting added for run 1 broke the two-GPU search. It now applies to training only. `FINETUNED_MODEL` lets the evaluation rerun without retraining |
| 3 | to do: evaluation (about 25-30 min with a reused model) | |

## Glossary

| Term | Meaning |
|---|---|
| Embedding, vector | the list of 384 numbers a model produces for a text |
| Cosine similarity | how close two vectors point, up to 1.0 |
| Bucket | an S1's shortlist of candidate records (02 full) |
| Anchor, positive, negative | the question, the right answer, a tempting wrong answer |
| In-batch negatives | the other rows' texts in the same step, used as wrong answers |
| Hard negative | a wrong answer that looks right (a lookalike) |
| Loss | the number training tries to lower; low = right answers ranked first |
| Parameters | the model's internal numbers, adjusted by training (118M for e5-small) |
| Step, batch, epoch | one update; the rows used in it; one full pass over all rows |
| Learning rate, warm-up | how big each update is; a gentle start with smaller updates |
| GPU, DDP | graphics card doing the math; one training process per GPU, kept in sync |
| Recall@k | share of true pairs found in the top k |
| Unseen / seen S1 | evaluation S1 never trained on / trained on (memorization check) |
| Leak | testing on what the model was trained on, which inflates the score |
