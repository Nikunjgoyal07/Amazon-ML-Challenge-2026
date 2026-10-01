# Submission-5 WWMD — in Simple Language but Full Detail

> WWMD = **What it is, Why it exists, How it works, Measure of success, Do — what you need to run it.**

Notebook: `pretrain-embedding-model.ipynb` (23 cells)
Saved output: `PRETRAIN-embed/er_embed/`

---

## 1. WHAT is submission-5?

In one line: **We built our own small search-brain for shop/company names and addresses, instead of borrowing Google's e5 brain.**

Long version:

The competition asks: for every business in Source-1, find the same business in Source-2 and Source-3, even when the spelling is different, words are missing, or the name is in Hindi / Tamil / French.

Old way (submission-2/3): use a ready-made model called `multilingual-e5-small` to turn every record into a number-list (embedding). Records with similar number-lists are treated as same shop.

New way (submission-5): **train our own number-maker from zero, only on this competition's shops.**

It is called `er-embed-small` and lives here:

```
submission-5/PRETRAIN-embed/er_embed/er-embed-small/
  config.json
  model.safetensors
  tokenizer.json, tokenizer_config.json
  1_Pooling/, modules.json, sentence_bert_config.json ...
```

It is a drop-in replacement. Wherever old code said `E5_MODELS = e5-small`, you can now say `E5_MODELS = er-embed-small` and everything else stays same.

There is also a half-finished version saved:

```
er-embed-small-mlm/
```

That is the brain after school (stage-1) but before practice (stage-2). You can restart from there.

Other files in `er_embed/`:

* `pretrain-embedding-model.ipynb` logic split as `stage1_mlm.py`, `stage2_contrastive.py`
* `eval_recall.csv` — score card: our brain vs e5 brain
* `mlm_log.csv` — school marks over time
* `contrastive_log.csv` — practice marks over time
* `embed_config.json` — full settings + sizes + times
* `lgbm_train_s1.txt` — list of shops LightGBM will use later (we never train on these)
* `matches_s1.txt` — list of shops we DID train on
* `translit_wordmap.json` — dictionary of 370 words like `praivet -> private`
* `__huggingface_repos__.json` — record that we downloaded e5 only for comparison

---

## 2. WHY does it exist?

e5 is very smart in general English, but weak in 3 places that matter here:

1. **Indian scripts:** `सन एनर्जी प्राइवेट लिमिटेड` — e5 tokenizer breaks it into tiny useless pieces.
2. **Empty address:** when address is blank, only name is there. e5 recall@10 was only `0.35` — means it found the true shop only 35 times out of 100.
3. **Shop-talk words:** `pvt, ltd, nagar, sarl, rue, st` — e5 does not give them full importance.

So idea: learn a tokenizer and brain **only from these shops**, with their Hindi, French, typos, abbreviations.

Simple analogy: e5 is a tourist who knows 100 languages a little. Our model is a local postman who knows only 3 streets — English, Hindi-scripts, French — but knows every house, shortcut, nickname on those streets.

Result (India test, shops model never saw in training):

* All shops, top-10: `0.9219 -> 0.9872` (+6.5%)
* Indian-script name: `0.9063 -> 0.9981` (almost 100%)
* Empty address: `0.3505 -> 0.9111` (3x better — biggest win)
* Latin with address: `0.9534 -> 0.9885`

That means for candidate shortlisting (top-30 per S1), we miss almost nothing now.

---

## 3. HOW does it work? Step by step

### Step 0: What text goes in?

Exactly same text as `02_full_e5_buckets` uses, so model can replace e5 without change:

```
query: sun energy pvt ltd | 12 mg road pune 411001
```

* `query: ` is added in front of everything.
* `name | address` comes from `01` cleaning (`name_norm`, `addr_norm`).
* If name/address was in Indian script, we first convert to English letters with `anyascii` + 2 fixes:
  - `റ്റ` -> `tt` (Malayalam)
  - nasal dot `ं` -> `n` before most letters
* Then name goes through word-map: `praivet -> private`, 370 such fixes.
* We also keep `native_text`: same record but in original script, example `सन एनर्जी लिमिटेड | ...`. This teaches the brain both scripts are same shop.

### Step 1: Collect corpus and pairs

For each country (France, India, US):

* **corpus:** every different text + every native text, from train AND test records. Sizes in this run: France `1,603,808`, India `13,615,026`, US `11,681,200` texts. Test text is used, but NO test labels — only words, no answers. Needed because France is only in test, else brain would never see French.
* **noisy pairs (500,000 per country):** take a record, make a dirty copy like real data: typo (`westgrove -> wcstgrove`), `road->rd, rue->r, private->pvt`, remove `pvt ltd`, drop city/PIN, move address words, `0058->58`, duplicate words, `sunenergy.com`, remove address. Brain learns: dirty copy = same shop.
* **script pairs (300,000 India):** transliterated text <-> original script text. Brain learns both scripts = same.
* **matches pairs (1,000,000 India, 1,000,000 US):** real true pairs `S1 <-> its true S2/S3` from `train_ground_truth.tsv`, PLUS a lookalike wrong shop from its bucket top-10 as hard negative. Only S1 **outside** LightGBM's 200k training set, so no leakage into later LightGBM step.
* **script_matches (300,000 India):** same but true record kept in original script.

France has no labels, so only noisy pairs — pure guess at noise. Can only be judged by final leaderboard.

### Step 2: Learn a tokenizer (dictionary)

WordPiece, 32,000 tokens, learned from 3,000,000 sampled texts (equal per country).

* Lower-case, NFKC, split on space/punctuation.
* Keeps 3000 letters so Hindi scripts + French `é è ê` all survive.
* Frequent shop words get whole tokens: `pvt, ltd, nagar, sarl, rue`.

Skipped if you give `MLM_MODEL=` — then old tokenizer is reused.

### Step 3: Stage-1 school — Masked Language Model

Small BERT: `6 layers, 384 width, 6 heads, 128 length, ~23M params`.

Teacher hides 15% words in a sentence: 80% become `[MASK]`, 10% random word, 10% stay. Student must guess hidden word. Only hidden spots are marked.

* Data: up to 4M texts per country = `9,603,808` total, 9378 steps.
* 2 GPUs (torchrun DDP), batch 1024, lr 5e-4, 1 epoch, fp16 on T4.
* Time: ~21 min. Loss `8.04 -> 2.75`, accuracy `0.17 -> 0.53`.

Why needed? If you start contrastive training from random weights, brain learns poorly. First teach it words, spellings, scripts.

Saved to `er-embed-small-mlm/` + `mlm_log.csv`. Script: `stage1_mlm.py`.

### Step 4: Stage-2 practice — Contrastive training

Take stage-1 brain + mean pooling. Show it pairs: anchor must be closer to its positive (true / noisy / script copy) than to every other text in same batch.

* Loss: `MultipleNegativesRankingLoss` (or cached version if batch >256 per GPU), cosine x `scale 20`.
* Batch 512 rows across all GPUs, 1 epoch, lr 1e-4 (higher than e5 fine-tune 2e-5 because fresh brain needs more learning), 5% warmup.
* `NO_DUPLICATES`: same text never twice in batch. `PROPORTIONAL`: each batch comes from one pair file, so negatives are same country+kind.
* Time: ~89 min, 1 GPU in this run. Loss `1.42 -> 0.02`.

Saved to `er-embed-small/` + `contrastive_log.csv`. Script: `stage2_contrastive.py`.

`QUICK_RUN=true` does all above on tiny samples (20k pairs, 100k texts) in 15-20 min to catch mistakes.

### Step 5: Exam at full size

Same exam code as `02b`: encode EVERY train S2/S3 record of a country, mean-center vectors, exact search.

Two groups:

* **unseen:** 20,000 S1 from LightGBM set — brain never trained on their pairs. This number will carry to test.
* **seen:** 5,000 S1 from training pairs — only counts trained pairs. If seen >> unseen, brain just memorized.

Report recall@5/10/30 overall + by type: Indian-script name, empty address, Latin with address. `gain` = ours minus e5.

Reading rule: **unseen all recall@30 is what matters.** Clearly above e5 by +0.005 => go to 02-full + 03-full pipeline.

---

## 4. MEASURE — what came out?

From `eval_recall.csv` (India):

| group | type | e5 r@10 | ours r@10 |
|---|---|---|---|
| unseen | all 73654 pairs | 0.9219 | 0.9872 |
| unseen | Indian-script 13249 | 0.9063 | 0.9981 |
| unseen | empty address 2836 | 0.3505 | 0.9111 |
| unseen | Latin+addr 57590 | 0.9534 | 0.9885 |
| seen | all 9506 | 0.9201 | 0.9887 |

e5 train top-30 report was 0.9471. Ours top-30 unseen all 0.9947.

Meaning: out of 100 true shops, e5 puts ~92 in top-10, we put ~98.7. For empty-address hard cases, e5 35/100, we 91/100.

---

## 5. DO — what do YOU need to run it?

### A. If you only want to USE the ready brain (no training):

You already have it. You still need these other existing files:

1. `01_...` output `processed/` with `FULL_RUN=true` — files `train_s1_India.parquet` etc + `test_...`. This is the raw text.
2. `02_full_e5_buckets.ipynb` — run it with `E5_MODELS=/kaggle/input/<dataset>/er_embed/er-embed-small`. It will make `embeddings_full/er-embed-small/buckets/`.
3. `03_full_lightgbm_submission.ipynb` — run it with `BUCKET_DIR=<above>/buckets`, keep `TRAIN_S1_PER_COUNTRY=200000, SEED=42` (must match `embed_config.json` `lgbm_seed`).
4. No GPU needed for just copying model, but embedding needs GPU.

You do NOT need `train_ground_truth.tsv`, `stage1/2.py`, or retraining.

### B. If you want to RETRAIN from zero:

1. `processed/` (required)
2. `train_ground_truth.tsv` (required — true pairs)
3. `02 full` output `buckets/train_*.parquet` (recommended — for hard negatives, else random negatives) + `translit_wordmap.json` (recommended — else relearned from pairs, same code)
4. Kaggle `GPU T4 x2 + internet ON` (to download e5 for comparison), packages `sentence-transformers 5.4.1, transformers 5.0.0, torch 2.10, datasets, tokenizers, rapidfuzz, anyascii, pyarrow`
5. Settings you can change by env var: `QUICK_RUN, USE_TEST_TEXT (must be true for French), VOCAB_SIZE 32000, HIDDEN 384, LAYERS 6, HEADS 6, MAX_LEN 128, TOKENIZER_TEXTS 3M, MLM_TEXTS_PER_COUNTRY 4M, MLM_EPOCHS 1, MLM_BATCH 1024, PAIRS_PER_COUNTRY 1M, NOISY 500k, SCRIPT 300k, CL_EPOCHS 1, CL_BATCH 512, CL_LR 1e-4, SCALE 20, LGBM_SEED 42`
6. Shortcuts: `MLM_MODEL=<er-embed-small-mlm folder>` skips tokenizer+stage1, `TRAINED_MODEL=<er-embed-small folder>` skips to eval only.

### Warnings in simple words:

* Test words are used, test answers are never used. Same as old `03` word-rarity. Still check competition rules.
* LightGBM shops are never used in `matches` — no cheating into next step.
* France has no answers — noisy copies are a guess. Only full pipeline + leaderboard can tell if France worked.

### Next step after this:

1. Upload `er_embed/` as Kaggle dataset.
2. Run `02 full` with this model.
3. Run `03 full` with those buckets.
4. Compare LightGBM CV `0.9796 with e5` vs new, and `02 full` train report.

That is the whole submission-5.
