# Research notes

Notes written **during** the competition, kept as they were written. They explain why some ideas were built and
others weren't. For what the pipeline does now, read [SOLUTION_WRITEUP.md](../SOLUTION_WRITEUP.md) and the
[pipeline docs](../pipeline/).

| Note | What it covers | What came of it |
|---|---|---|
| [RESEARCH_APPROACHES.md](RESEARCH_APPROACHES.md) | survey of winning solutions to similar problems (Foursquare Location Matching, Shopee) and of entity-resolution research: candidate generation, pairwise matchers, decision rules | the overall design: retrieve candidates with embeddings, judge pairs with gradient boosting, decide with the one-owner rule |
| [BLOCKING_METHODS.md](BLOCKING_METHODS.md) (was `buckets.md`) | catalogue of candidate-generation methods, each rated for this data | **built:** exact dense search, address key, character 3-gram name search (plus the reverse, number and empty-address searches from later miss analyses). **Tested and rejected:** FAISS IVF-PQ. **Closest relatives built later:** supervised contrastive blocking (SC-Block) ≈ `experiments/02b_finetune_e5.ipynb`; self-supervised dense blocking ≈ the noisy copies in `experiments/02c_pretrain_embedding.ipynb`. **Not built:** sorted neighbourhood, phonetic blocking, BM25 over full records, HNSW |
| [MODIFICATIONS_REVIEW.md](MODIFICATIONS_REVIEW.md) (was `modifs.md`) | evidence-based review of proposed changes to blocking, preprocessing, matcher and embeddings | **built:** the 3-gram name search; address keys; house-number extraction (`bare_house_no` in 01, `street_of` in 02/03); working around 01's accent removal for Indian scripts by transliterating the original text in 02/03 |
| [INDICXLIT_PLAN.md](INDICXLIT_PLAN.md) (was `hybridplanner.md`) | plan for a hybrid anyascii + IndicXlit transliteration | explored in `submission-3/` and tested as `01_keshav.ipynb` / `02_keshav_exp.ipynb` (now in the git history); not adopted ([experiments/INDICXLIT.md](../experiments/INDICXLIT.md)) |

**Old file names in these notes:**

| Old name | Now |
|---|---|
| `ARCHITECTURE.md` | [SOLUTION_WRITEUP.md](../SOLUTION_WRITEUP.md) and [pipeline/](../pipeline/) |
| `FILES.md` | the repository map in the [README](../../README.md), the submission folders' READMEs, and the settings tables in [pipeline/](../pipeline/) |
| `PLAN.md`, `explanation.md`, `first-preprocessing.md` | removed (superseded); in the git history |
| `buckets.md`, `modifs.md`, `hybridplanner.md` | renamed as in the table above |
