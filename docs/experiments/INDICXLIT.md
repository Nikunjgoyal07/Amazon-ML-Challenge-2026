# IndicXlit transliteration (tested, not adopted)

**What it is:** a teammate's idea. Indian-script text would be transliterated with **IndicXlit**, AI4Bharat's neural
transliteration model (MIT), instead of `anyascii` rules alone. The research track, with its own README, is in
[`submission-3/`](../../submission-3/). The original plan is in
[research/INDICXLIT_PLAN.md](../research/INDICXLIT_PLAN.md).

**Result:** on a 1M sample, bucket recall and ceilings matched the anyascii pipeline within ±0.001. Full results:
[SOLUTION_WRITEUP.md §10.10](../SOLUTION_WRITEUP.md).

---

## How the hybrid worked

The test notebooks were `01_keshav.ipynb` and `02_keshav_exp.ipynb`: our 01 and the sample version of 02, with
transliteration routed through IndicXlit.

- **Which words:** a word with an Indian script, and more than 4 characters, went through IndicXlit. Short words, and
  words when no model was loaded, used `anyascii` plus the two fixes (Malayalam `റ്റ` → `tt`, nasal dot → `n`).
- **One decode per distinct word**, cached. The data is templated: only about 1,550 distinct Indian-script words
  exist in the whole competition data. Decoding them all took about 3 minutes once, on a laptop GPU.
- **Junk guard:** a word with no Indian script at all was never sent to the model. Mojibake apostrophes and dashes
  had been decoded as if they were Hindi.
- **Settings:**
  - `XLIT_DIR`: a checkpoint folder. Without one, the notebooks behaved exactly like the anyascii versions.
  - `XLIT_AUTO_DOWNLOAD`: downloads the ~124 MB checkpoint once.

## Why it didn't move the score

02 and 03 never use 01's transliteration for the embedding text or the main comparisons. They transliterate the
**original** text themselves (see [pipeline/01_PREPROCESSING.md](../pipeline/01_PREPROCESSING.md)), so a change in
01 only reaches legal form, phonetic code, city and state for Indian-script records.

The 02 variant did change the embedding text. But the word quality was mixed:
- better: `solutions` instead of `solyusns`
- worse: `egeneerig` for engineering

In aggregate it was a wash.

The from-scratch embedding model of `submission-5` attacks the same weakness from the model side. Its own tokenizer
lifted India's Indian-script recall@30 from 0.94 to 0.9998 ([EMBEDDING_PRETRAINING.md](EMBEDDING_PRETRAINING.md)).

## Rerunning it

The two notebooks were removed from the repo. Restore them from the git history; the sample notebooks they pair
with are still in `submission-4/`:

```bash
git checkout 4b7ddf4 -- 01_keshav.ipynb 02_keshav_exp.ipynb
```
