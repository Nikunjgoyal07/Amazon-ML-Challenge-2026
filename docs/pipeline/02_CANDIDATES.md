# 02 · Candidate generation (`submission-4/02_full_e5_buckets.ipynb`)

**What it does:** for every S1, builds a shortlist of S2/S3 records that could be the same business: its
**bucket**. 03 only ever judges pairs inside a bucket, so a true match missing here is lost for good.

```
processed/  ──►  02 full  ──►  embeddings_full/<model>/buckets/<split>_<country>.parquet
                                + translit_wordmap.json, bucket_recall_train.csv, manifest.json
```

**Run:** Kaggle, GPU T4 x2, internet on (it downloads e5). About 24M texts are encoded; my estimate is about an hour
of encoding plus 30-50 minutes for the extra searches. A bucket file that already exists is skipped, so a rerun in
the same session continues where it stopped.

Why it works this way, and how much each part gained: [SOLUTION_WRITEUP.md §6](../SOLUTION_WRITEUP.md). This page is
about how the code works.

---

## Cell by cell

| Cell | What it does | Key names |
|---|---|---|
| 1 | Settings, input discovery, model selection | `set_model`, `E5_MODEL_IDS`, `EXTRA_SOURCES`, … |
| 3 | Inventory of `processed/`: which (split, source, country) files exist | `INVENTORY` |
| 5 | Encoding on all GPUs (one process per GPU) | `get_pool`, `embed_all` |
| 7 | Exact nearest-neighbour search on the GPUs, and pair scores | `exact_search`, `pair_score`, `pair_cosine` |
| 9 | Ground truth, used only for the train recall report | `owner_rows`, `recall_row` |
| 11 | The five extra searches | `street_of`, `address_pairs`, `name_pairs`, `number_pairs`, `empty_pairs`, `sparse_top` |
| 13 | Indian-script transliteration | `translit`, `latin` |
| 15 | Word map for Indian-script names (learned, or loaded if attached) | `learn_word_map`, `translit_name` |
| 16 | **Building the buckets**, one (split, country) at a time | `read_texts`, `build_model_buckets` |
| 18 | Runs `build_model_buckets` for every model in `E5_MODELS` | |
| 20 | Compares the models' train recall reports (when several were run) | |

---

## Text given to the model

`read_texts` builds one text per record. With the default `TEXT_MODE=fixed`, that text is
`name | address`, or just `name` when the address is empty:

| Record | Name part | Address part |
|---|---|---|
| Latin script | 01's `name_norm` | 01's `addr_norm` |
| Indian script | the **original** `business_name`, transliterated (`translit`) and word-mapped (`translit_name`) | the original `business_address`, transliterated (`latin`) |

Why not 01's text for Indian scripts: 01's accent removal deletes their vowel signs (see
[01_PREPROCESSING.md](01_PREPROCESSING.md#the-known-weakness-indian-scripts)).

**`translit(s)`** is `anyascii` with two corrections, measured on true pairs:
- Malayalam `റ്റ` becomes `tt`. anyascii gives `rr`, so "limited" came out as "limirrd".
- The nasal dot (anusvara) becomes `n` before most consonants. anyascii always gives `m`, so "Anand" came out as
  "anmd".

**The word map** (`learn_word_map`, cell 15) learns transliterated word → English word from the train true pairs:
1. For every Indian-script candidate name with a known true S1, each transliterated word is aligned to the most
   similar word of the S1's name (rapidfuzz ratio ≥ 55).
2. A mapping is kept when it is seen in at least 20 pairs and wins at least 60% of that word's occurrences.

The result is about 370 words (`solyusns` → solutions, `praivet` → private), saved as `translit_wordmap.json`. 03
loads the same file, so both stages see identical text.

Every text gets the prefix `query: ` (e5's convention for "compare like with like") and is cut at 128 tokens.

---

## Encoding (cell 5)

- **One process per GPU:** `get_pool` starts one sentence-transformers worker process per GPU, and `embed_all` sends
  it chunks of texts.
- **Each distinct text encoded once:** duplicates are common, so texts are factorized first.
- **Longest texts first:** batches then hold texts of similar length, which wastes less padding.
- **Storage:** fp16 on the T4s, vectors normalized to length 1.

## Exact search (cell 7)

`exact_search(base, queries, mu, k)` returns, for every query vector, the k most similar base vectors.

1. **Mean-centering:** `mu` (the average S2/S3 vector of the country) is subtracted from both sides, and the vectors
   are re-normalized. e5 vectors share a large common direction; removing it makes the similarities more
   discriminative.
2. **Base on each GPU:** the whole base matrix (all of the country's S2/S3 vectors) is loaded onto each GPU.
3. **Query blocks:** queries are processed in blocks sized so that one score block uses at most 1/8 of GPU memory.
   Even blocks go to GPU 0 and odd blocks to GPU 1, one thread per GPU.
4. **Two passes:** a fast fp16 matrix product shortlists the best k + 32, and those are re-scored in fp32 to get
   the final top k.

It is exact, with no approximation. FAISS IVF-PQ was tested and lost 10 points of recall
([SOLUTION_WRITEUP.md §6.2](../SOLUTION_WRITEUP.md)).

---

## The five extra searches (cell 11)

These use simpler text than e5. Names are 01's `core_name`, lowercased; Indian scripts are transliterated and
word-mapped. Addresses are the original text, transliterated if needed.

| Search | Function | How it finds pairs |
|---|---|---|
| `reverse` | `exact_search` with the roles swapped | every S2/S3 record looks up its 5 closest S1 (`REVERSE_K`). The pair joins the bucket when the S1 is among the record's best `REVERSE_TOP` = 2 |
| `address` | `address_pairs` | key = house number + first street word, from `street_of`: `15 bis Rue de Barbieux, Roubaix` gives `15|barbieux`. S1 and candidates with the same key are paired. A key shared by more than `ADDRESS_KEY_CAP` = 20 candidates is too common and skipped |
| `name` | `name_pairs` → `sparse_top` | character 3-gram TF-IDF vectors of the names (3-grams in more than 2% of names ignored), and the `NAME_TOP` = 5 most similar candidate names per S1. The similarity is a sparse matrix product on the GPUs, in blocks of 256 S1; a block that runs out of GPU memory is split in two and retried |
| `number` (India) | `number_pairs` | keys = every (number, word of 3+ letters) combination in an address, e.g. `206|pune` (leading zeros dropped, street and stop words excluded). Keys shared by more than `NUMBER_KEY_CAP` = 10 candidates are skipped |
| `empty` | `empty_pairs` | the name search again, but only over candidates with **no address**, in both directions: each S1's 2 closest names among them (`EMPTY_S1_TOP`), and each such candidate's 3 closest S1 names (`EMPTY_NAME_TOP`) |

`street_of` (also copied into 03) takes the first comma part that has a number or a street word, then removes
numbers, street words (rue, road, avenue, …) and stop words (de, la, the, …). What remains is the street name.

---

## Building a bucket (`build_model_buckets`, cell 16)

For each split (train, test) and each country:

1. **Read and encode:** read the S1 and S2/S3 texts, then encode both.
2. **e5 search:** the top `TOP_K` = 30 candidates per S1. For train, it searches up to top-50, so the recall report
   can compare k = 20/30/40/50.
3. **Extra searches:** each gives a list of (S1 row, candidate row) pairs, encoded as one integer key,
   `s1_row × n_candidates + candidate_row`.
4. **Union:** the e5 keys and every extra search's keys, deduplicated.
5. **Score and sort:** every pair gets its e5 `score` (mean-centered cosine) and `cosine` (plain cosine). The rows
   are sorted by S1, then by score, and `rank` is the position within the S1's bucket.
6. **Found by:** for each search, the value it gave the pair, or empty when it didn't find it.
7. **Save:** write `buckets/<split>_<country>.parquet` and record it in `manifest.json`. A rerun skips buckets
   already in the manifest, as long as the settings are unchanged.

### Bucket file columns

| Column | Meaning |
|---|---|
| `s1_entity_id`, `candidate_entity_id`, `country` | the pair |
| `rank` | position in the S1's bucket (0 = best score) |
| `candidate_source` | `S2` or `S3` |
| `score` | e5 similarity after mean-centering (used for ranking) |
| `cosine` | plain e5 cosine |
| `e5_rank` | rank in the e5 top-30 (empty: not found by e5) |
| `rev_rank` | this S1's rank among the candidate's 5 closest S1 (reverse search) |
| `addr_key`, `num_key` | 1 = found by the address key / number key |
| `name_rank` | rank among the S1's closest names (name 3-grams) |
| `empty_rank`, `empty_rev_rank` | ranks in the empty-address name search, from the S1's side / the candidate's side |

The rows are grouped by S1 in the order of 01's `train_s1_<country>.parquet`. 03 relies on this to recreate its
random choice of training S1, and 02b/02c check it.

### Train recall report

For train, `recall_row` measures, for each candidate set, the share of true matches inside the buckets. It also
breaks this down by script and for candidates with an empty address, and computes the **ceiling**: the score a
perfect judge would get on these buckets. The candidate sets are e5 top-20/30/40/50, top-30 plus each search, and
all of them. The result is `bucket_recall_train.csv`; full-size numbers are in
[SOLUTION_WRITEUP.md §6.4](../SOLUTION_WRITEUP.md).

Embeddings are not saved: all 24M vectors would take about 18.6 GB, close to Kaggle's 20 GB output limit. Only the
buckets (a few hundred MB) are kept.

---

## Several models in one run

`E5_MODELS=small,base` (or a folder path, e.g. a fine-tuned model) runs the whole build once per model, each into
its own `embeddings_full/<model name>/` folder. The word map and ground truth are computed once and shared. Cell 20
then compares the recall reports of every model found in `EMB_FULL_DIR`. Each extra model roughly adds a full run's
time.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `SPLITS` | `train,test` | which splits get buckets (a submission needs both) |
| `TOP_K` | 30 | e5 candidates kept per S1 |
| `E5_MODELS` | `small` | `small` / `base` / `large` (multilingual-e5), a model id, or a folder; comma-separated |
| `TEXT_MODE` | `fixed` | `fixed` = as above; `processed` = 01's text only; `latin`, `raw` = older variants |
| `EXTRA_SOURCES` | `reverse,address,name,number,empty` | which extra searches run; `""` = e5 only |
| `REVERSE_TOP` | 2 | reverse search: add the pair when the S1 is among the candidate's best 2 |
| `ADDRESS_KEY_CAP` | 20 | skip address keys shared by more candidates |
| `NAME_TOP` | 5 | name 3-gram matches per S1 |
| `NUMBER_KEY_CAP`, `NUMBER_COUNTRIES` | 10, `India` | number key: cap; where it runs |
| `EMPTY_NAME_TOP`, `EMPTY_S1_TOP` | 3, 2 | empty-address search, candidate side / S1 side (0 = off) |
| `WORD_MAP` | `true` | apply the word map to Indian-script names |
| `COUNTRIES` | all | testing aid: only these countries |
| `PROCESSED_DIR`, `DATA_ROOT`, `EMB_FULL_DIR` | searched / `embeddings_full` | inputs and output |

## Adding a new search

1. **Write a function** in cell 11 that returns `(s1_rows, candidate_rows)`, and optionally a value per pair, such
   as a rank.
2. **Call it** in `build_model_buckets` under a new `EXTRA_SOURCES` name, store it in `extra[...]`, and `attach` its
   value as a new bucket column.
3. **Use it in 03:** add the column to `read_buckets` and to the "found by" features.
4. **Measure it:** compare the recall report line `e5 top-30 + <new>` with the others. A good search finds many
   missed matches for few extra pairs per S1.
