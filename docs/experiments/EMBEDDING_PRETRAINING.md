# Pretraining our own embedding model (notebook 02c)

This page explains `02c_pretrain_embedding.ipynb`, which builds an embedding model **from scratch**, trained only on
the competition's records. It covers English, Hindi (and the other Indian scripts in the data) and French. The page
assumes you know the pipeline (01 → 02 full → 03 → 04), but not how neural networks are trained.

Read [the fine-tuning page](EMBEDDING_FINETUNING.md) first if you can. It explains what an embedding model does in
our pipeline and the "pick the right one" training game, which this notebook reuses.

**Status:** first full run done on Kaggle (30 Sep). Its outputs (model, logs, `eval_recall.csv`) are in
[`submission-5/`](../../submission-5/); results in section 0 below. The maintained notebook is
`experiments/02c_pretrain_embedding.ipynb`. It adds the leak check (section 9) to the copy that ran; the pair
selection itself is unchanged. The next step, running 02 full and 03 with this model, hasn't been done yet.

## 0. Results of the first run

India, full size: each evaluated S1 searches all 4.1M train S2/S3 records. The **unseen** S1 are LightGBM's training
S1; the model never trained on their pairs.

| Unseen S1, kind of true record | True pairs | e5-small recall@5 / @10 / @30 | **er-embed-small** recall@5 / @10 / @30 |
|---|---|---|---|
| all | 73,654 | 0.836 / 0.922 / 0.947 | **0.927 / 0.987 / 0.995** |
| Indian-script name | 13,249 | 0.823 / 0.906 / 0.939 | **0.962 / 0.998 / 1.000** |
| empty address | 2,836 | 0.166 / 0.351 / 0.491 | **0.629 / 0.911 / 0.969** |
| Latin script, with address | 57,590 | 0.872 / 0.953 / 0.971 | **0.934 / 0.989 / 0.995** |

How to read it:
- **The evaluation itself checks out.** e5's unseen recall@30 (0.9467) matches 02 full's train report (0.9471).
- **No sign of memorization.** The seen group scores the same as the unseen one (all, recall@30: 0.996 vs 0.995).
- **Biggest gains on e5's weak spots:** names in Indian scripts, which our tokenizer reads directly, and records
  with no address, which the noisy copies taught it to match on the name alone.
- **Only half the answer:** recall decides the ceiling. The end-to-end effect (03's cross-validation, then the
  leaderboard) is still to be measured.

**The run:**

| Part | Result |
|---|---|
| Data | 24M records (France 1.7M, India 10.5M, US 12.0M) → 26.9M corpus texts; texts average 20 tokens |
| Stage 1 | 9.6M texts, 9,378 steps on 2 GPUs, 21 min; loss 8.04 → 2.75 |
| Stage 2 | about 4M pair rows, 89 min on **1 GPU** (see the known issue below); loss 1.42 → 0.02 |
| Pairs | `matches` 1M each for India and the US (100% with a hard negative from the bucket); `noisy` 500K per country; `script` and `script_matches` 300K each for India. France's 9 `script` rows were left out as too few |

**Known issue: stage 2 on two GPUs.**
- **What happened:** the two training processes hung on a synchronization step until the 10-minute timeout, and the
  notebook fell back to one GPU as designed.
- **Likely cause:** the pair files have different shapes. `matches` rows have three texts, the others two. At the
  same step the two GPUs can get batches of different kinds and make a different number of sync calls. 02b,
  with a single kind of pair file, ran fine on two GPUs.
- **Fix (not done yet):** give every pair file the same columns, or turn off cross-GPU sharing when the files
  differ. Until then, stage 2 takes about 1.5 hours instead of about 45 minutes.

---

## 1. The idea, and how it differs from fine-tuning

The fine-tuning notebook (02b) takes e5, a model someone else trained on huge amounts of web text, and adjusts it to
our data. This notebook starts from **random numbers** and learns everything from our own records.

| | 02b: fine-tune e5 | 02c: our own model |
|---|---|---|
| Starting point | e5-small, trained by its authors on around a billion text pairs | random numbers |
| Tokenizer (how text is cut into pieces) | e5's: 250,000 pieces for ~100 languages | our own: 32,000 pieces learned from our records |
| Size | 118M parameters (96M of them the piece table) | ~23M parameters |
| Training data | our true pairs (India, US) | our records' text, then our pairs, plus noisy copies (India, US, France) |
| French | only what e5 already knew | learned from the French records' text |
| Time on 2 × T4 | ~35 min training | ~3-4 h in total (estimate) |
| Main risk | forgets a little, memorizes a little | too little training to match e5's general knowledge |
| Main hope | fewer misses on our noise | better on our scripts and vocabulary, and faster to run |

## 2. The building blocks

### The tokenizer: cutting text into pieces

A model can't read letters directly. The **tokenizer** cuts text into pieces, called **tokens**, from a fixed list,
and each piece gets a number. Frequent words stay whole; rare words are cut into frequent pieces. For example
(illustrative, the real cuts depend on the data):

```
"sun energy pvt ltd | 12 mg road pune"   →  sun · energy · pvt · ltd · | · 12 · mg · road · pune
"wcstgrove" (a typo)                      →  w · ##cst · ##grove       (## = continues the previous piece)
"boulangerie sarl | rue de la gare"       →  boulangerie · sarl · | · rue · de · la · gare
```

**Why our own tokenizer:**
- **e5's list is spread thin.** Its 250,000 pieces cover about 100 languages, so our words compete with Japanese,
  Arabic and Swahili for room.
- **Ours spends every piece on this data.** Its 32,000 pieces are learned from our records, so words like `pvt`,
  `nagar`, `sarl` or `rue`, city names, and Hindi or Tamil words get whole pieces of their own.

The method is called **WordPiece**:
- **Normalization:** text is normalized (NFKC) and lowercased.
- **Splitting:** it is split on spaces and punctuation.
- **Alphabet:** up to 3,000 distinct characters are kept, enough for every Indian script and French letter in the
  data.

### The network: 6 layers that read the pieces together

The model is a small **BERT**, the standard design for this job:

1. **Piece table.** Each piece's number is looked up in a table of 32,000 × 384 numbers, so every piece becomes
   384 numbers.
2. **Six layers.** In each, every piece "looks at" the other pieces of the text and updates its own 384 numbers.
   This is the **attention** mechanism: after a few layers, `pvt` next to `ltd` means something different from `pvt`
   alone, and `12` next to `road` is read as a house number. Each layer has 6 attention "heads", which can each
   follow a different kind of relation.
3. **One vector per text.** The pieces' final numbers are averaged (**mean pooling**) into one list of 384 numbers:
   the text's embedding.

About 23 million parameters in all, half of them the piece table. e5-small has the same width (384) and 12 layers,
but its piece table alone is 96M.

## 3. Step 1: the text

**Same text as 02 full.** The model is trained on exactly the text 02 full embeds: `query: name | address` from 01's
cleaned fields, with Indian-script names and addresses transliterated and names passed through the word map. The
model can then replace e5 in 02 full with nothing else changed.

**Plus the original script.** For records written in an Indian script, the original text is used too, for example
`सन एनर्जी लिमिटेड | पुणे` next to its transliteration `sun energy limited | pune`. That is what makes the model truly
multilingual.

**Train and test records.**

| | S1 | S2 | S3 |
|---|---|---|---|
| Train | 2.21M | 5.03M | 5.29M |
| Test | 1.73M (France 0.26M) | 4.89M | 5.08M |

About 24 million records in all. France exists **only in the test set**, so without the test records' text the
model would never see French. Only their **text** is used; test labels don't exist, and nothing is inferred about
which records match. `USE_TEST_TEXT=false` turns this off. See section 10 about the rules.

## 4. Step 2: learning the tokenizer

**Sample:** 3,000,000 texts, 1,000,000 from each country, so each language gets its share.

**Result:** 32,000 pieces. The notebook prints a few texts cut into pieces, including Indian-script ones, as a check.

**Time:** a few minutes.

## 5. Step 3, stage 1: fill in the blanks

A model with random numbers knows nothing: not that `ltd` and `limited` are related, not that `411001` is a Pune
PIN code, not even which pieces tend to appear together. Training it straight away on "same business or not" works
poorly. It needs to learn the language first.

**The exercise** is called masked-language modelling (MLM):
1. **Hide:** 15% of the pieces in each text are hidden. Of those:
   - 80% are replaced by a `[MASK]` piece
   - 10% by a random piece
   - 10% are left as they are
2. **Guess:** the model has to guess the original pieces.

```
sun energy [MASK] ltd | 12 mg [MASK] pune maharashtra 411001
          → guess: "pvt", "road"
```

To guess well, it must learn:
- spelling
- abbreviations
- which words go together (a city with its state and PIN, `pvt` with `ltd`, `rue` with `de la`)
- how each script works

That is exactly the knowledge an embedding needs.

**Settings:**
- **Data:** up to 4,000,000 texts per country (about 10M in total), one pass.
- **Batches:** 1,024 texts per step, 512 per GPU, grouped by length, so that little space is wasted on padding.
- **Learning:** rate 5e-4, a 6% warm-up, then a slow decrease to zero. Updates are capped in size ("gradient
  clipping") to keep training stable.
- **Speed-up:** only the hidden positions are scored, which is faster than scoring every position and gives the same
  learning.

**What the log shows:** the loss, which should fall, and the **masked accuracy**, the share of hidden pieces guessed
right, which should rise steadily. My estimate is about 45 minutes.

**Output:** the encoder and its tokenizer are saved in `er-embed-small-mlm/`, every 5,000 steps and at the end. If a
later step fails, `MLM_MODEL=<that folder>` restarts from here.

## 6. Step 4, stage 2: learning "same business"

Filling blanks teaches the language, but not what "same business" means. Stage 2 plays the fine-tuning notebook's
"pick the right one" game:
- **Batches:** 512 rows per step, shared by both GPUs, so each anchor chooses among all texts of the step.
- **Loss:** `MultipleNegativesRankingLoss`, with scale 20.
- **Batch rules:** no duplicate texts in a batch, and each batch drawn from one kind of pair.

It trains on four kinds of pairs:

| Pairs | Anchor ↔ positive | Countries | Rows (default) |
|---|---|---|---|
| `matches` | S1 ↔ one of its true records, plus a lookalike from its bucket as a tempting wrong answer. Only S1 outside LightGBM's set | India, US | up to 1,000,000 per country |
| `noisy` | a record ↔ the same record with synthetic noise (below) | India, US, France | 500,000 per country |
| `script` | a record's transliterated text ↔ its original Indian script | India | up to 300,000 |
| `script_matches` | S1 ↔ its true record in its original Indian script | India | up to 300,000 |

About 4 million rows and 8,000 steps.

**Learning rate: 1e-4.** That is higher than 02b's 2e-5. A small, new model has more to learn, and less hard-won
knowledge to protect, than e5.

### Noisy copies: the only lesson for France

France has no labels, so there are no true pairs to learn from. Instead, the notebook takes a record and makes a noisy
copy of it, with 1 to 3 kinds of noise found between true pairs in our data:

| Noise | Example |
|---|---|
| typo (swap, drop, double or replace a letter) | westgrove → wcstgrove |
| abbreviation or expansion | road ↔ rd, private ↔ pvt, rue ↔ r, saint ↔ st |
| legal form removed | sun energy pvt ltd → sun energy |
| address word dropped, last words (city, PIN) dropped, words moved | … pune 411001 → … |
| house-number format | 0058 → 58, 12 → no 12 |
| duplicated or swapped name words | sun energy → sun sun energy |
| name written as a website, or address removed | sun energy pvt ltd → sunenergy.com |

The model learns that these differences do **not** change the business, while every other record in the batch is a
different business. The notebook prints real examples when it runs.

**The limit:** this noise is our best reconstruction of the real noise, not the real noise. Whether it teaches
enough for France can only be seen through the full pipeline and the leaderboard.

## 7. Two GPUs

Both stages run as separate scripts, with one process per GPU (DDP, as in 02b):
- **The work:** each GPU handles half of every batch, and their updates are averaged, so both copies of the model
  stay identical.
- **Stage 2 sharing:** the two GPUs also swap their encoded texts, so every anchor competes with the whole batch.
- **Fallback:** if a two-GPU run fails, the notebook retries that stage on one GPU.

The scripts are saved next to the model (`stage1_mlm.py`, `stage2_contrastive.py`) so the run can be inspected or
repeated.

## 8. Evaluation: our model vs e5

The same evaluation as 02b:
- **Recall@5/10/30:** each evaluated S1 searches all 4.1M India S2/S3 train records.
- **Groups:** unseen S1 (from LightGBM's set) and seen S1.
- **Slices:** Indian-script name, empty address, Latin script with an address.
- **Models:** our model is compared with `multilingual-e5-small`, and with any other model listed in
  `COMPARE_MODELS`, for example 02b's fine-tuned e5.
- **Gain:** the `gain` column is ours minus e5.

The notebook also prints each model's encoding speed. A faster model makes 02 full faster too.

**How to read it:**
- **"unseen, all, recall@30" (India):** the number that matters. For reference, e5-small scores 0.9471.
- **"Indian-script name":** where e5's general tokenizer is weakest, and where our own model should gain first.
- **"seen" far above "unseen":** the model memorized the `matches` pairs.

## 9. Is the evaluation honest? (leak audit)

| Question | Answer |
|---|---|
| Are the evaluation's S1 or their true records in the `matches` / `script_matches` pairs? | **No.** Those pairs use only S1 outside LightGBM's set, and a record has at most one owner. The notebook stops if any pair involves an evaluation S1 or one of its records, before any training starts. |
| Did the model see the evaluation records' **text**? | **Yes, by design, with no pairing.** The tokenizer, stage 1 (fill in the blanks), and the `noisy` and `script` pairs use every record's text, including the evaluation S1 and their true records. None of these tells the model which record matches which S1. Each record is only paired with a copy of itself. The test records get exactly the same treatment, so this mirrors test conditions instead of inflating the score. |
| Could the noisy copies help the evaluation pairs indirectly? | If anything, they hurt slightly. An evaluation S1 and its true record can land in the same `noisy` batch, where the game treats them as *different* businesses. |
| Identical texts shared with a training pair? | Counted by the notebook (`evaluation pairs with a trained pair's exact texts`); expect 0 or a handful. |
| The word map? | Learned in 02 full from all train pairs, and it shapes the text of both our model and e5 in the same way. The comparison stays fair; see the fine-tuning page for details. |
| The "seen" group? | Training data on purpose: a memorization check, never the headline. |

**Downstream caveat (as for 02b):** the `matches` S1 stay in the train buckets that 03 learns from. When one of
LightGBM's S1 competes with them for a record, the contest is slightly easier than on test, so 03's cross-validation
could be a little optimistic. The seen-vs-unseen gap shows how large that can be.

**Verdict:** the "unseen" number is not measured on pairs the model trained on. The model has read the evaluation
records' text, like it reads the test records' text, but has never been told which ones match.

## 10. Rules

- **The model is our own:** ~23M parameters, trained from random numbers on competition data only. That is far below
  the 8B limit, with no external data or pretrained weights.
- **Libraries:** PyTorch (BSD), transformers, tokenizers, sentence-transformers (Apache 2.0).
- **Test text:** using the test records' text (never labels) is the one point to confirm against the competition
  rules before submitting anything built on this model. It is the same kind of use as 03's word rarity, which is
  computed over the test S1. `USE_TEST_TEXT=false` removes it, at the price of no French.

## 11. Running it

On Kaggle, with the accelerator **GPU T4 x2** and internet on (the e5 comparison downloads e5):

1. **Attach:** 01's `processed/`, the competition data, and 02 full's output (hard negatives and word map).
2. **Quick check:** `QUICK_RUN=true`. Every step runs on small samples, in about 15-20 minutes. It finds mistakes
   before the long run.
3. **Full run:** `QUICK_RUN=false`. My estimates:

| Part | Time |
|---|---|
| Reading records, building the corpus and pairs | ~30 min |
| Tokenizer + cutting 10M texts into pieces | ~10 min |
| Stage 1 (fill in the blanks) | ~45 min |
| Stage 2 (pairs) | ~60 min |
| Evaluation (e5 and ours on 4.1M records) | ~30 min |

The training steps print the time left.

**Rerunning part of it:**
- `MLM_MODEL=<attached>/er_embed/er-embed-small-mlm` skips the tokenizer and stage 1.
- `TRAINED_MODEL=<attached>/er_embed/er-embed-small` skips all training and only evaluates.

**Main settings:**

| Setting | Default | Meaning |
|---|---|---|
| `LAYERS`, `HIDDEN`, `HEADS`, `VOCAB_SIZE` | 6, 384, 6, 32,000 | model and tokenizer size (`LAYERS=12` for a bigger model) |
| `MLM_TEXTS_PER_COUNTRY`, `MLM_EPOCHS`, `MLM_BATCH`, `MLM_LR` | 4M, 1, 1,024, 5e-4 | stage 1 |
| `PAIRS_PER_COUNTRY`, `NOISY_PAIRS_PER_COUNTRY`, `SCRIPT_PAIRS` | 1M, 500K, 300K | stage 2 data |
| `CL_EPOCHS`, `CL_BATCH`, `CL_LR`, `SCALE` | 1, 512, 1e-4, 20 | stage 2 training |
| `USE_TEST_TEXT` | on | test records' text for tokenizer, stage 1, noisy copies |
| `COMPARE_MODELS` | e5-small | models evaluated next to ours |

## 12. What to expect, and what to do with the result

**Honest expectation.**
- **What e5 has on us:** its authors trained it on vastly more text than we can in a few hours on two T4s. A small
  model trained from scratch is not guaranteed to beat it overall.
- **Where ours is most likely to gain:** the Indian-script slice, our vocabulary, and speed.

**If it beats e5** (+0.005 or more at unseen recall@30, India):
1. Run 02 full with `E5_MODELS=/kaggle/input/<dataset>/er_embed/er-embed-small`.
2. Run 03 full with `BUCKET_DIR=<that run>/embeddings_full/er-embed-small/buckets`.
3. Compare 03's cross-validation with 0.9796, and retune France's cutoff with 04.

**If it doesn't:**
- **More stage 1:** more texts or more passes (`MLM_TEXTS_PER_COUNTRY`, `MLM_EPOCHS`).
- **More stage 2 data:** `PAIRS_PER_COUNTRY`.
- **A bigger model:** `LAYERS=12`.
- **Save time:** rerun only stage 2 with `MLM_MODEL`.

**If it's close to e5 but different** (better on some slices, worse on others), it might still help as a second
search. 02 full can already build buckets with both models (`E5_MODELS=small,<our model>`, one folder each), but
merging the two sets of candidates for 03 would need new code.

## Glossary

| Term | Meaning |
|---|---|
| Token, tokenizer | a piece of text from a fixed list; the tool that cuts text into those pieces |
| WordPiece | a way to learn that list: frequent words whole, rare words in pieces |
| BERT, layer, attention | the standard text-reading network; one reading pass; each piece looking at the others |
| Mean pooling | averaging the pieces' numbers into one vector per text |
| MLM (masked-language modelling) | stage 1: hide pieces, guess them back |
| Masked accuracy | share of hidden pieces guessed right |
| Contrastive training | stage 2: pull matching texts together, push the others apart |
| Noisy copy | a record with synthetic noise, used as its own positive |
| Parameters | the model's internal numbers, adjusted by training |
| DDP | one training process per GPU, kept in sync |
| Unseen / seen S1 | evaluation S1 never trained on / trained on (memorization check) |
| Leak | testing on what the model was trained on, which inflates the score |
