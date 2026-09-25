# Amazon-ML-Challenge-2026

Business entity resolution: for every Source 1 business, find the Source 2 / Source 3 records that
are the same business.

**Pipeline:** `01_eda_preprocessing` (clean) → `02_full_e5_buckets` (embed + shortlist 30 candidates
per S1) → `03_full_lightgbm_submission` (LightGBM judges each pair → submission files) →
`04_rethreshold` (optional per-country cutoffs).

## Documentation

| Document | Answers |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How does the solution work, and why was it built this way? |
| [docs/DATA_AND_OUTPUTS.md](docs/DATA_AND_OUTPUTS.md) | What does each folder and output file contain? |
| [docs/FILES.md](docs/FILES.md) | What does each file in the repo do? |
| [docs/SUBMISSION_STEPS.md](docs/SUBMISSION_STEPS.md) | How do I run a submission on Kaggle and validate it? |
| [docs/RESULTS_EXPLAINED.md](docs/RESULTS_EXPLAINED.md) | What do the printed scores mean? |
| [docs/first-preprocessing.md](docs/first-preprocessing.md) | What did the preprocessing run find and do? |
| [docs/PLAN.md](docs/PLAN.md), [docs/RESEARCH_APPROACHES.md](docs/RESEARCH_APPROACHES.md) | What approaches were considered? |
