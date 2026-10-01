<div align="center">

![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=170&section=header&text=Submission-3&fontSize=44&desc=hybrid+transliteration+track+%C2%B7+IndicXlit+%2B+anyascii&descAlignY=62)

[![typing](https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1200&color=0066FF&center=true&vCenter=true&width=640&lines=Fix+the+weakest+slice%3A+Indian-script+names;Hybrid+transliteration%2C+acceptance-gated;Anchored+sampling+for+honest+recall)](https://github.com/Nikunjgoyal07/Amazon-ML-Challenge-2026)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-1.44-CD792C?style=for-the-badge&logo=polkadot&logoColor=white)
![IndicXlit](https://img.shields.io/badge/IndicXlit-hybrid_engine-FF6B35?style=for-the-badge)
![anyascii](https://img.shields.io/badge/anyascii-ISC-6AA84F?style=for-the-badge)
![Status](https://img.shields.io/badge/Track-research_experiment-blue?style=for-the-badge)
![Kaggle](https://img.shields.io/badge/Runs_on-Kaggle_CPU-20BEFF?style=for-the-badge)

[![Skill Icons](https://skillicons.dev/icons?i=python,pytorch,git&theme=light)](https://skillicons.dev)

</div>

## 📌 What is this?

**Submission-3 is the transliteration-upgrade experiment track**: Indian-script names are the search stage's weakest slice (17% of true matches missed vs 4% for Latin script), so this folder tests a **hybrid transliteration engine** — IndicXlit for native-script tokens, anyascii for Latin/French — behind strict acceptance gates, plus **anchored dev sampling** so recall numbers are honest. No leaderboard score is claimed here; winners graduate into the main pipeline (see `docs/hybridplanner.md`).

| Item | Detail |
|---|---|
| 🎯 Goal | Close the 17%-vs-4% miss gap on Indian-script true matches |
| 🔤 Engine | Hybrid: IndicXlit (Devanagari/Telugu/Tamil/…) + anyascii (Latin/French) |
| 🛡️ Safety | `XLIT_DIR` unset → engine fully disabled, byte-identical baseline |
| 📐 Method | Anchored sampling (GT matches guaranteed inside the sample) |
| 📊 Test | Simple `country \| phonetic \| core_name[:4]` blocking stand-in, ON vs OFF × native vs Latin |

## 🧠 Approach

Three ideas, each fixing a real trap from earlier experiments:

1. **Hybrid transliteration.** anyascii romanizes Indian scripts crudely (`praivet limited` for प्राइवेट लिमिटेड). Routing native-script tokens through a pure-torch port of **IndicXlit** while keeping Latin/French on anyascii should produce spellings closer to how the S1 side writes them — raising name similarity before any embedding is involved. The engine is a drop-in helper: with `XLIT_DIR` empty, `XLIT_MODEL is None` and every code path returns exactly today's output.
2. **Acceptance gates, run every time.** Regex probes always run; IndicXlit anchor checks run when a model is loaded. If any assert fails, stop and recheck the engine against the fairseq sources — never "fix" it by tweaking weights or thresholds. This keeps a transliteration experiment from silently corrupting the pipeline.
3. **Anchored sampling.** The old per-file sampler took `head(quota)` from S1 and S2/S3 *separately*, so a sampled S1's true matches often weren't in the S2/S3 sample at all — recall computed on that was meaningless. `build_anchored_sample` guarantees each sampled S1's GT matches land inside the sample, so ON-vs-OFF recall deltas are real.

```mermaid
flowchart LR
    A["raw TSVs"] --> B["normalize_frame<br/>(NEW_COLS helper)"]
    B --> C{"token script?"}
    C -->|native| D["IndicXlit<br/>(needs XLIT_DIR)"]
    C -->|latin/french| E["anyascii<br/>(as today)"]
    D --> F["romanized text"]
    E --> F
    F --> G["acceptance gates<br/>(regex + anchors)"]
    G --> H["anchored sample<br/>(GT inside)"]
    H --> I["baseline blocking key<br/>country|phon|core[:4]"]
    I --> J["recall: ON vs OFF<br/>x native vs latin"]
```

## 📁 Contents

| Notebook | What it does |
|---|---|
| [`preprocessing-3.ipynb`](preprocessing-3.ipynb) | 01 preprocessing with sample controls (`SAMPLE_SIZE` 40k default · `SAMPLE_MODE` linked/random/head · `PROCESSED_OUT`), full EDA always on all rows |
| [`amazon-ml (2).ipynb`](<amazon-ml%20(2).ipynb>) | Preprocessing + **hybrid transliteration engine** + anchored sampling + baseline blocking recall (hybrid ON vs OFF, native-vs-Latin split) |

Sample modes: `linked` (S1 sample + their true S2/S3 matches first, then distractors — real pairs to learn from) · `random` · `head` (fastest, almost no pairs).

## 🚀 Run

```bash
# byte-identical baseline (default — no model assets needed)
# Run all: FULL_RUN=true for the real processed/ ; sample settings otherwise

# hybrid engine experiment (needs the unzipped indicxlit assets)
XLIT_DIR=/path/to/indicxlit-assets  # unset/empty => hybrid disabled, nothing changes
```

> Test files must be processed with `FULL_RUN=true` before any final submission — every test entity must be present regardless of dev-time sampling.

## 📊 How to read the result

* **Native-script recall delta (ON − OFF)** is the only number that matters; Latin-slice recall must not drop.
* The blocking key here is a deliberately simple stand-in for notebook 02's embedding search — fast enough to iterate, good enough to tell whether hybrid transliteration moves recall *before* wiring it into the expensive stage.
* Promotion rule (from the planner): only a clear, gated, anchored win gets merged into `02_full_e5_buckets`'s transliteration path.

## 🧭 Where it fits

* Builds on **submission-2**'s pipeline (anyascii transliteration + word map).
* Successor ideas live in `docs/hybridplanner.md`; the from-scratch tokenizer of **submission-5** ultimately attacks the same weakness from the model side.

---

<div align="center">

[![Repo Card](https://github-readme-stats.vercel.app/api/pin/?username=Nikunjgoyal07&repo=Amazon-ML-Challenge-2026&theme=tokyonight)](https://github.com/Nikunjgoyal07/Amazon-ML-Challenge-2026)

*Engine sources: IndicXlit (fairseq-based) · anyascii (ISC) · Polars (MIT).*

</div>
