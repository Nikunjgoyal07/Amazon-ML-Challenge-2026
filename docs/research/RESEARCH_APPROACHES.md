# Business Entity Resolution: approaches from winning solutions and research

*Written 2026-09-24. Survey for the Amazon ML Challenge 2026 team.*

**Bottom line:** strong solutions to problems like ours share four stages:

1. **Candidate generation.** A high-recall union of cheap blockers (text/TF-IDF keys plus embedding nearest neighbours).
2. **Gradient-boosted matcher.** A model on string-similarity and embedding features.
3. **Transformer cross-encoder.** It re-scores the uncertain pairs.
4. **Graph-level post-processing.** It enforces consistency across all pairs (one owner per record, transitivity, cleanup).

Most top teams in the closest Kaggle competition (Foursquare Location Matching) used a version of this. The 3rd-place team instead went end to end with learned embeddings. The research literature explains why each stage works and where it breaks. Section 9 maps all of it onto our pipeline and lists the next experiments by expected value.

## Contents
1. [Our problem in the literature's terms](#1-our-problem-in-the-literatures-terms)
2. [Closest competitions and what won](#2-closest-competitions-and-what-won)
3. [Candidate generation (blocking)](#3-candidate-generation-blocking)
4. [Pairwise matching models](#4-pairwise-matching-models)
5. [Clustering, global consistency and post-processing](#5-clustering-global-consistency-and-post-processing)
6. [Choosing matches to maximize F-beta](#6-choosing-matches-to-maximize-f-beta)
7. [Business names, addresses and scripts](#7-business-names-addresses-and-scripts)
8. [Rules check: what is allowed](#8-rules-check-what-is-allowed)
9. [What this means for our pipeline](#9-what-this-means-for-our-pipeline)
10. [Sources](#10-sources)

---

## 1. Our problem in the literature's terms

| Our situation | Literature name | Why it matters |
|---|---|---|
| S1 is deduplicated; S2/S3 contain several records per business | **Clean–dirty, multi-source ER** | Each S2/S3 record belongs to **at most one** S1 (verified: 0 reuse in 7.6M train links). Every S2/S3 record needs one best S1 or none, a bipartite assignment ([Papadakis et al., EDBT 2022](https://arxiv.org/abs/2112.14030); [optimal F-score bipartite linkage, 2023](https://arxiv.org/abs/2311.13923)). |
| Score = F0.5 per S1, averaged over all S1 | **Per-query (macro) metric**, like Shopee (mean F1 per posting) and Foursquare (mean IoU per record) | Every S1 counts equally. A false match on a singleton costs a full point, so decision rules should be set per S1 (§6). |
| Only names and addresses, no coordinates or IDs | **Textual / dirty EM** | Deep models help most on textual and dirty data ([Ditto](https://www.vldb.org/pvldb/vol14/p50-li.pdf); [DeepBlocker](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)). |
| France appears only in test | **Domain shift / zero-shot matching** | This favours models that generalise, such as multilingual pretrained encoders. It argues against features tied to the training countries ([AnyMatch](https://arxiv.org/abs/2409.04073); [Peeters et al., EDBT 2025](https://www.uni-mannheim.de/media/Einrichtungen/dws/DWS_News/Documents/Peeters-Entity-Matching-using-LLMs-EDBT2025.pdf)). |
| ~10M records per split | **Scalable ER** | This requires a blocking stage ([Papadakis et al., ACM CSUR 2020 survey](https://arxiv.org/abs/1905.06167)). |

---

## 2. Closest competitions and what won

### 2.1 Foursquare Location Matching (Kaggle 2022) — the best analogue

**The task:** about 1.5M place records (name, address, category, coordinates) across many countries and scripts. Predict which records describe the same place. The metric is mean IoU per record, which is per-query like ours ([Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/); [7th-place write-up](https://future-architect.github.io/articles/20220720a/)).

| Place | Team | Pipeline | Source |
|---|---|---|---|
| 1st | re:waiwai | **3 stages + post-processing.** Candidates come from text similarity + geographic proximity. A **LightGBM with few features** cuts them to **40 per record**. A second LightGBM uses rich features (Levenshtein, Jaro-Winkler, distances, SVD-reduced embeddings). Then come **xlm-roberta cross-encoders**, fine-tuned iteratively. | [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/), [Kaggle write-up](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution) |
| 2nd | 2:30 (DeNA) | Coarse candidates come from geographic proximity + **TF-IDF**. A **Transformer-based blocking stage** reduces candidates while keeping recall. The final model is an ensemble of **LightGBM + XGBoost + BERT** models. | [DeNA](https://dena.ai/news/202207-kaggle-4sq/), [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/) |
| 3rd | psi (Philipp Singer) | **Metric learning with ArcFace loss** (predict the POI ID, so records of the same place embed close together). Then a **bi-encoder** trained on first-stage candidates, blended. | [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/), [Kaggle write-up](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/psi-3rd-place-solution) |
| 4th | Matiounine, Schuler, Viel | **Two-level boosting**: a level-1 prefilter, then a level-2 matcher, with CatBoost / LightGBM / XGBoost and post-processing. | [GitHub](https://github.com/TheoViel/kaggle_foursquare) |
| 7th | Future Architect | **32 candidates per record from 5 retrievers**: distance, distance+embedding, word and character bag-of-words, and embedding similarity. Their best-possible IoU was 0.9778. LightGBM uses 100+ features (Levenshtein, Gestalt, ROUGE, category). **Graph post-processing** removes low-centrality edges and keeps nodes within distance 2. Self-supervised embeddings (SimCSE-style) beat Universal Sentence Encoder, which *hurt*. | [Blog](https://future-architect.github.io/articles/20220720a/) |
| top finisher | Yuki Uehara | Candidates, then LightGBM filter, then 2 transformer pair classifiers, then **GNN node classification as post-processing**; the organizer reports a full-framework score change **0.907 → 0.946**, without isolating the GNN's individual contribution. | [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/) |

**Lessons that carry over:**
- **Recall first, then precision.** Every top team built a wide candidate set from *several* retrievers, then filtered in stages with models of increasing cost: cheap GBDT, then rich GBDT, then transformer.
- **Boosted models were the usual backbone.** String-similarity features plus a boosted model carried the 1st, 2nd, 4th and 7th places. Transformers added the last points for 1st and 2nd. The 3rd place shows an all-embedding route (metric learning + bi-encoder).
- **Graph post-processing can pay off.** Uehara's full four-stage framework was reported as 0.907 → 0.946, but the public account does not isolate the GNN's individual gain; it was also a core step for the 7th-place team.
- **Several text normalisations** (character level, word level, romanised) served different stages. That is the same idea as our `name_clean` / `name_core` / native-script dictionary.
- **Check train/test overlap early.** The 7th-place account discusses a reported/possible 67% Foursquare train-test duplication and the resulting re-evaluation; treat that as a warning about leakage, not as a clean, universal fact ([7th-place blog](https://future-architect.github.io/articles/20220720a/)).

### 2.2 Shopee Price Match Guarantee (Kaggle 2021)

**The task:** group product listings (title + image) that are the same product. The metric is mean F1 per listing, again per-query ([Kaggle](https://www.kaggle.com/c/shopee-product-matching/overview)).

The 1st-place ingredients were ([write-up summary](https://masatakashiwagi.com/blog/kaggle-shopee-solution/)):
- **Metric learning** (ArcFace / CurricularFace) on image and text encoders. The text models were xlm-roberta-large, Indonesian BERTs and multilingual BERT. Matches were found by cosine kNN.
- **Iterative Neighborhood Blending (INB).** Each item's embedding is replaced by a similarity-weighted blend of its neighbours, repeatedly, which sharpens cluster boundaries.
- **Rules tuned to the per-query metric.** The "min2" rule keeps an item's second-closest neighbour even when it falls outside the threshold, so each item returns at least one other item. That suits data where items almost always have a match. Our data is the opposite: 5.6% singletons and a precision-heavy metric, so we need a rule that *allows empty answers*.

**Lesson:**
- **Tune the post-processing to the exact metric.** Per-query metrics reward thresholds and rules set per query, and the winners spent most of their effort there.
- **Blend neighbours before thresholding.** INB-style blending can be applied to S2/S3 records that are near-duplicates of each other before assigning them to an S1.

---

## 3. Candidate generation (blocking)

| Approach | How it works | Evidence | Fit for us |
|---|---|---|---|
| **Standard / key blocking** | Records sharing a key (a token, postcode, phonetic code, house number + street…) are compared. | A classic family; surveyed in [Papadakis et al. 2020](https://arxiv.org/abs/1905.06167). | Cheap and interpretable. It catches the "name replaced, address identical" cases that embeddings miss. |
| **Top-k TF-IDF (Sparkly)** | For each record, retrieve the k highest TF-IDF/BM25 matches on the best attribute/tokenizer (Lucene). | Reports outperforming eight state-of-the-art blockers and emphasizes the recall/output-size trade-off ([Sparkly, PVLDB 2023](https://dl.acm.org/doi/10.14778/3583140.3583163)). | Strong, simple baseline. Char n-grams handle typos. Needs sparse top-k machinery at our scale. |
| **Canopy / q-gram TF-IDF** | Char 4-gram TF-IDF canopies, then clustering inside each canopy. | Used for **organisation-name disambiguation** at PatentsView scale ([PatentsView](https://patentsview.org/disambiguation/disambiguation-metrics)). | Directly on business names. |
| **Self-supervised DL blocking (DeepBlocker)** | Learn record embeddings **without labels** (autoencoder, cross-tuple training), then run kNN. | Autoencoder best on structured/dirty data, Hybrid best on textual. It beat labelled AutoBlock and an industrial blocker on dirty/textual data. **Combining DL and non-DL blockers did better still** ([DeepBlocker, PVLDB 2021](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)). | Supports a *union* of blockers. We have labels, so supervised variants are also possible. |
| **Pretrained sentence embeddings + kNN** | Encode each record with a sentence-embedding model and retrieve the top-k by cosine. | Ditto's "advanced blocking": Sentence-BERT + blocked matrix multiplication, top-10 per record ([Ditto](https://www.vldb.org/pvldb/vol14/p50-li.pdf)). **Our small 02b benchmark:** multilingual-e5-small reached 94.9% recall@5 on its fixed 10K-per-country sample, with hard slices deliberately oversampled; this is an optimistic model comparison, not full-data recall. | **Our main dense retriever to benchmark at full index size.** mE5 is MIT-licensed ([report](https://arxiv.org/html/2402.05672v1), [model](https://huggingface.co/intfloat/multilingual-e5-small)). |
| **Hybrid dense + learned sparse retrieval (BGE-M3)** | One multilingual encoder can produce dense, sparse, and multi-vector representations. | [BGE-M3](https://huggingface.co/BAAI/bge-m3) is MIT-licensed and supports hybrid retrieval. | Optional experiment only: it is much heavier than mE5-small, so benchmark it if the TF-IDF/dense union still misses the recall target. |
| **Supervised contrastive blocking (SC-Block)** | Fine-tune the encoder with supervised contrastive loss on labelled matches, then run kNN. | Reports smaller candidate sets at **99.5% pair completeness** and roughly 4× speedup on its large product benchmark; the advantage is scale-dependent and must be re-measured here ([SC-Block](https://arxiv.org/abs/2303.03132)). | High upside, but not a guarantee: we have 7.6M labelled links to fine-tune mE5 with, while the real index is much denser than the paper's examples. |
| **Metric learning (ArcFace)** | Treat each entity as a class so embeddings cluster by entity. | Foursquare 3rd place; Shopee 1st place (§2). | An alternative to contrastive fine-tuning; S1 IDs serve as classes. |
| **Neural LSH (NLSHBlock)** | A fine-tuned LM acts as a learned hashing function, with an LSH-based loss. | Significant gains over existing blockers ([Wang et al. 2024](https://arxiv.org/abs/2401.18064)). | Research-grade; lower priority. |
| **Transformer blocking stage** | A light transformer re-ranks coarse candidates before the expensive matcher. | Foursquare 2nd place (§2). | Could sit between the blocker and LightGBM. |

**What the evidence says:**
- **Use more than one blocker.** Lexical blockers (TF-IDF, keys) and semantic ones (embeddings) find different matches, and DeepBlocker, Foursquare 7th and our own prototype all show the union beats either alone.
- **Report blocking quality honestly.** Report pair completeness (recall) against the candidate count per record, per country and per hard slice.
- **Feed the matcher the rank and margin.** Top-k per query suits the "each record has one owner" structure, and the rank and score gap become matcher features.

---

## 4. Pairwise matching models

### 4.1 Probabilistic record linkage (Fellegi–Sunter)
Each field comparison is scored by its likelihood under "match" vs "non-match" distributions, estimated by EM with no labels needed. Production tool: **Splink** (UK Ministry of Justice), which implements Fellegi–Sunter in SQL on DuckDB or Spark and scales to large datasets ([Splink](https://moj-analytical-services.github.io/splink/index.html), [theory](https://moj-analytical-services.github.io/splink/topic_guides/theory/fellegi_sunter.html)).

- **Fit:** a solid unsupervised baseline and a way to create interpretable features ("match weight" per field). A supervised GBDT usually beats it once labels exist.
- **Related:** **ZeroER** (SIGMOD 2020) models match and non-match similarity vectors with Gaussian mixtures plus transitivity. It rivals supervised methods with zero labels ([ZeroER](https://arxiv.org/abs/1908.06049)). This is worth remembering for **France**, where we have no labels.

### 4.2 Feature engineering + classifier (GBDT)
The workhorse of every Foursquare top team, of Magellan / `py_entitymatching` (BSD) and of `dedupe`, which adds active learning ([py_entitymatching](https://github.com/anhaidgroup/py_entitymatching); [dedupe docs](https://docs.dedupe.io/en/latest/API-documentation.html)).

Features the sources used or recommend:

| Group | Features | Sources |
|---|---|---|
| String similarity | Levenshtein, Jaro-Winkler, Jaccard, token-set ratio, Gestalt/SequenceMatcher, ROUGE-N/L, TF-IDF cosine (word and char n-gram) | Foursquare 1st and 7th; PatentsView |
| Embedding similarity | cosine of sentence embeddings; SVD-reduced embeddings as raw features | Foursquare 1st; our 02b |
| Structure | candidate rank, score gap to the best candidate, number of candidates, which blocker produced the pair | Foursquare stage design (implied by top-k filtering); our plan |
| Domain parts | legal form agree/conflict, formerly-known-as / alias parts, address number match | Wasi & Flaaen `stnd_compname` / `stnd_address` ([paper](https://ideas.repec.org/a/tsj/stataj/v15y2015i3p672-697.html)) |

**Fit:** the core of our Solution B. LightGBM is MIT-licensed and fast. RAPIDS FIL gave the 7th-place team a 100× inference speed-up.

### 4.3 Pretrained-LM cross-encoders (Ditto family)
**Ditto** ([PVLDB 2021](https://www.vldb.org/pvldb/vol14/p50-li.pdf), [code](https://github.com/megagonlabs/ditto)) treats matching as sequence-pair classification with BERT/RoBERTa.

- **Input format:** `[COL] name [VAL] … [COL] address [VAL] …`, then `[CLS] left [SEP] right [SEP]`.
- **Three add-ons:**
  - **Domain knowledge:** tag or normalise important spans, such as numbers.
  - **Summarisation:** TF-IDF-based, keeps long text inside 256 tokens.
  - **Data augmentation:** span deletion or shuffling, attribute or entry swap, plus MixDA.
- **Results:**
  - Up to +29% F1 over the previous state of the art.
  - Matches the old state of the art with half the labels.
  - Largest gains on **dirty** data (+31.9 F1 on dirty Walmart-Amazon).
  - A **real company-matching task (789K vs 412K company records) reached 96.5% F1**.
  - Scoring only the top-10 candidates took 1,339 s vs 22,823 s for all pairs.

**Fit:** the natural stage-3 model on our uncertain pairs. Use a multilingual backbone (XLM-R or mDeBERTa, MIT). Ditto's augmentation matches our situation, because our noise looks synthetic and learnable: typos, legal-form swaps, prefixes, domains, aliases.

### 4.4 Contrastive and metric-learning bi-encoders
- **R-SupCon** (Peeters & Bizer, WWW 2022): supervised contrastive pre-training, then pair fine-tuning. It is more label-efficient than cross-encoders, reaching 94.29 F1 on Abt-Buy ([paper](https://dl.acm.org/doi/fullHtml/10.1145/3487553.3524254), [code](https://github.com/wbsg-uni-mannheim/contrastive-product-matching)).
- **Sudowoodo** (ICDE 2023): self-supervised contrastive representations, then few-label fine-tuning ([code](https://github.com/megagonlabs/sudowoodo)).
- **ArcFace metric learning:** Foursquare 3rd place, Shopee 1st place.

**Fit:** the same fine-tuned encoder can do both blocking (kNN) and matching (cosine as a feature). This is attractive given our 7.6M labelled links.

### 4.5 LLMs (generative models)
- **Zero-shot and few-shot:** this paper studies generative LLMs (GPT-4 and open models) as matchers that depend less on task-specific training data and are more robust than fine-tuned pretrained-LM matchers. GPT-4 can also explain its decisions and help analyse errors ([Peeters, Steiner & Bizer, EDBT 2025](https://www.uni-mannheim.de/media/Einrichtungen/dws/DWS_News/Documents/Peeters-Entity-Matching-using-LLMs-EDBT2025.pdf)).
- **Fine-tuning:** fine-tuning significantly improves *small* LLMs (LoRA r=64 on Llama-3.1-8B), with mixed results for large ones ([Steiner et al. 2024](https://arxiv.org/abs/2409.08185)). Note that Llama's licence is **not** MIT/Apache, so we cannot use it.
- **Tiny models:** **AnyMatch** fine-tunes GPT-2 (124M) for zero-shot matching. It reached 81.96 average F1 vs 86.36 for GPT-4-based MatchGPT, at **3,899× lower inference cost** ([AnyMatch](https://arxiv.org/abs/2409.04073)).
- **Budget-aware:** **Alper** (May 2026) combines cheap graph label propagation with a limited budget of LLM pair queries, chosen by constrained optimisation ([arXiv 2605.25814](https://arxiv.org/abs/2605.25814)).

**Fit:** an LLM cannot score millions of pairs on our hardware. It could arbitrate a small set of the most uncertain or highest-impact pairs, such as possible false matches on likely singletons. Allowed models (MIT/Apache, at most 8B parameters):
- Qwen2.5-7B-Instruct (Apache-2.0) ([Qwen](https://qwenlm.github.io/blog/qwen2.5/))
- Mistral-7B-Instruct-v0.3 (Apache-2.0) ([HF](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/tree/main))
- Phi-4-mini-instruct (MIT) ([HF](https://huggingface.co/microsoft/Phi-4-mini-instruct))

---

## 5. Clustering, global consistency and post-processing

| Technique | What it does | Evidence | Fit for us |
|---|---|---|---|
| **One-to-one / bipartite matching** | From a similarity graph between two clean sources, pick pairs so each record matches at most one (e.g. Unique Mapping Clustering, Ricochet variants) | 8 algorithms compared on 700+ similarity graphs from 10 datasets ([Papadakis et al., EDBT 2022](https://arxiv.org/abs/2112.14030)); a VLDB-J follow-up on one-to-one matching ([link](https://dl.acm.org/doi/10.1007/s00778-023-00791-3)) | Our constraint is one-sided: an S2/S3 record has one owner, while an S1 may own many records. Do not use standard one-to-one assignment or connected components as the final solver: keep S1 identities fixed and never merge two S1 entities. Use argmax/assignment only as a many-to-one owner decision. |
| **Multi-source clustering (FAMER)** | Holistic clustering across 3+ sources, including new schemes tailored to multi-source data | Distributed evaluation of 6–8 schemes ([Saeedi et al.](https://link.springer.com/chapter/10.1007/978-3-319-66917-5_19); [FAMER](https://dbs.uni-leipzig.de/research/projects/famer)) | Use S2↔S3 links as extra evidence: if an S2 and an S3 record match each other strongly, they should share an owner. |
| **Transitive-consistency cleaning (TransClean)** | Uses model evaluations and limited labeling/pseudo-labeling to find pairwise matches that break transitivity across sources | Reports **+24.42 average F1** in its multi-source experiments versus traditional pairwise matchers; this is not an expected gain for our pipeline ([TransClean](https://arxiv.org/abs/2506.04006)) | A possible precision booster for F0.5. Test it conservatively; it drops an S1←S2 link when the S2's near-duplicates point to a different S1. |
| **Graph centrality pruning** | Build a graph of predicted pairs, drop low-centrality edges, keep neighbours within distance 2 | Foursquare 7th place | A simple post-step to test. |
| **GNN post-processing** | Classify nodes/edges on the predicted-pair graph | Foursquare reported a full four-stage framework changing 0.907 → 0.946; the public account does not isolate the GNN's individual gain ([Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/)). | Worth trying once the base pipeline is stable, but do not attribute the whole improvement to the GNN. |
| **Neighbourhood blending (INB)** | Replace each embedding with a weighted blend of its close neighbours, iteratively | Shopee 1st place | Blend S2/S3 embeddings with their near-duplicates before assignment. |
| **Label propagation + LLM budget (Alper)** | Joint matching and clustering over one evolving graph | [arXiv 2605.25814](https://arxiv.org/abs/2605.25814) | Research-grade, recent. |

---

## 6. Choosing matches to maximize F-beta

With a per-S1 F0.5, the decision is which subset of an S1's candidates to output, including the empty set. The theory:

- **Two routes to F-measure.** "Empirical utility maximisation" tunes a threshold on validation. The "decision-theoretic approach" takes calibrated probabilities and picks the prediction with the best *expected* F. The paper also covers when each wins ([Ye et al., ICML 2012](https://icml.cc/2012/papers/175.pdf)).
- **An exact optimum exists under the modeled response distribution.** The General F-measure Maximizer (GFM) is Bayes-optimal for its binary/multilabel response model and requires a quadratic joint-distribution representation; it is inspiration for calibration/decision search, not a drop-in solver for our one-sided lists ([Waegeman et al., JMLR 2014](https://jmlr.org/papers/volume15/waegeman14a/waegeman14a.pdf)).
- **One-to-one linkage.** For bipartite record linkage where each record in both files links to at most one counterpart, there is a method that estimates matches by maximising expected F-score ([arXiv 2311.13923](https://arxiv.org/abs/2311.13923)). Our S1 side is many-to-one, so this is an inspiration rather than a direct solver.

**Practical recipe:**
- Calibrate the pair probabilities.
- For each S1, sort the candidates it owns, then choose the prefix (or empty set) with the highest expected F0.5. Compare this with plain global thresholds on out-of-fold predictions.
- Singletons deserve explicit handling. The empty set scores 1.0 when right and a single false match scores 0 (Shopee's min2 rule is the mirror image).

---

## 7. Business names, addresses and scripts

| Topic | Approach found | Source | Notes for us |
|---|---|---|---|
| Company-name standardisation | Rule files split a name into a core name, an **entity-type** part (Inc, Ltd…) and a **formerly-known-as** part, then standardise each | [Wasi & Flaaen, Stata Journal 2015](https://ideas.repec.org/a/tsj/stataj/v15y2015i3p672-697.html) | Our 01 preprocessing does the same: `name_legal`, alias (`dba`/`f/k/a`) handling, `name_core`. |
| Organisation disambiguation at scale | Jaro-Winkler + Jaccard + **char 4-gram TF-IDF** canopies, then hierarchical agglomerative clustering with a tuned threshold | [PatentsView](https://patentsview.org/disambiguation/disambiguation-metrics), [clustering](https://patentsview.org/disambiguation/disambiguation-clustering) | Char n-gram TF-IDF on `name_core` as both blocker and feature. |
| Address parsing | **libpostal**: a statistical parser/normaliser for 60 languages and 100+ countries, trained on OpenStreetMap/OpenAddresses; MIT code | [GitHub](https://github.com/openvenues/libpostal) | Rules risk (§8): its models are trained on **external address data**. We use our own rules instead. |
| Indic transliteration | **IndicXlit**: Roman↔Indic models for 21 languages, MIT, offline pip package `ai4bharat-transliteration` | [GitHub](https://github.com/AI4Bharat/IndicXlit), [Aksharantar paper](https://arxiv.org/abs/2205.03018) | Rules risk: trained on the external Aksharantar corpus. Our native→English dictionary is learned from the provided data without labels; any reported coverage must be fold-safe and treated as a diagnostic, not ground-truth recall. |
| Multilingual text | Multilingual encoders (mE5, XLM-R, mDeBERTa) | [mE5 report](https://arxiv.org/html/2402.05672v1) | mE5-small handled raw native scripts in our benchmark (94.7% recall@5). |
| Romanised normalisation | Separate romanised text versions for the neural models | Foursquare 7th place | Same idea as our `anyascii` fallback. |

---

## 8. Rules check: what is allowed

The challenge rules forbid external databases, APIs, geocoding and internet data augmentation. The final model must be MIT or Apache-2.0 licensed and at most 8B parameters. Also audit the licenses of all shipped code and model weights: a permissive model checkpoint does not automatically make a research repository or an external-data model permissible. Reimplement ideas from papers rather than copying code/data whose license is incompatible with the submission package. If the review treats every dependency as part of the submission, check ISC/GPL/AGPL helpers such as `anyascii`/`unidecode` rather than assuming the model-license rule covers them.

| Item | Status |
|---|---|
| String similarity, TF-IDF, LightGBM/XGBoost/CatBoost, FAISS-style kNN | Allowed (code, no external data) |
| multilingual-e5-small/base/large (MIT), XLM-R / mDeBERTa (MIT), MiniLM (Apache) | Allowed (pretrained model, permissive licence) |
| Qwen2.5-7B (Apache), Mistral-7B-v0.3 (Apache), Phi-4-mini (MIT) | Allowed as models; watch the compute cost |
| Llama-3.x, Gemma | **Not allowed** (licence is not MIT/Apache) |
| Hand-written dictionaries (abbreviations, legal forms, state names) | Generally domain knowledge rather than an external lookup, but the rules do not explicitly define every borderline case; obtain written organizer clarification and document provenance |
| Dictionaries learned from provided data (our native-script map) | Allowed |
| **libpostal, IndicXlit** | **Grey area.** The licences are fine, but the models embed external training data. Ask the organisers before using them, or skip them. |
| Geocoding, business registries, web search | **Forbidden** |
| Unlabelled test inputs (e.g. learning a dictionary from test records, self-training on France) | Not an external lookup, but transductive and not explicitly resolved by the written rules. Keep the safe submission train-only; seek clarification before shipping any test-derived learning. |

---

## 9. What this means for our pipeline

What we already have, and how it lines up with the evidence:

| Stage | Built so far | Evidence it's the right direction |
|---|---|---|
| Normalisation | `01_preprocess`: legal forms, aliases, domains/handles, digit-for-letter swaps, abbreviations, state codes, a native-script dictionary learned without labels | Wasi & Flaaen; Foursquare 7th (multiple normalisations) |
| Dense retriever choice | `02b` benchmark: **multilingual-e5-small** is the strongest overall in the small fixed sample, with some slice ties and optimistic sampling | Ditto advanced blocking; DeepBlocker; SC-Block |

Next experiments, in rough order of expected gain per hour:

1. **Build the union blocker and measure recall@k per slice.**
   - mE5 kNN (cleaned name+address), plus
   - char n-gram TF-IDF top-k on `name_core` (Sparkly / PatentsView style), plus
   - an address key (number set + street word, which our 02b pseudo-label rule already implements).

   *Why:* every top Foursquare team did this, and DeepBlocker found unions win.
2. **Two-stage LightGBM.**
   - A cheap LightGBM with few features to cut candidates to ~20–40 per record (Foursquare 1st).
   - Then a rich LightGBM with string, embedding, rank/margin and group-relative features.
3. **Decision layer for F0.5.** Argmax owner per S2/S3, a singleton gate, and per-S1 expected-F0.5 subset selection (§6), tuned on out-of-fold data.
4. **Consistency post-processing.** Use S2↔S3 near-duplicate links: TransClean-style removal of inconsistent links, and INB-style embedding blending. This is cheap and targets precision, which is what F0.5 rewards.
5. **Fine-tune mE5 on train links** (contrastive or ArcFace with S1 IDs as classes; SC-Block, R-SupCon, Foursquare 3rd). Treat this as an experiment to re-measure at our index density, not a guaranteed improvement; the real index is much denser than the paper examples.
6. **Ditto-style cross-encoder** (mDeBERTa/XLM-R) on pairs where LightGBM is uncertain.
   - Serialise with `[COL]`/`[VAL]`.
   - Augment with the observed noise operators.
   - **For France:** first build a validation-only French proxy from held-out training entities by applying the observed transformations (abbreviation, accent loss, legal-form and component-reordering noise). Do not use test-derived pseudo-labels in the safe submission; test-time self-training is transductive and should be used only if the organisers explicitly allow it and it is documented.
7. **Optional:**
   - GNN post-processing (Foursquare: +0.04).
   - An allowed ≤8B LLM as an arbiter on a small budget of high-impact uncertain pairs (Alper-style).
   - ZeroER-style unsupervised calibration for France.

**Validation hygiene** (from Foursquare's leak and our own dictionary lesson):
- Anything learned from data (dictionaries, embeddings, thresholds) is evaluated only on held-out S1 groups.
- Check whether any test records duplicate train records before trusting local scores.

---

## 10. Sources

**Competitions:**
- Foursquare Location Matching:
  - [Foursquare blog on winning solutions](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/)
  - [1st-place write-up](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution)
  - [3rd-place write-up](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/psi-3rd-place-solution)
  - [DeNA 2nd-place announcement](https://dena.ai/news/202207-kaggle-4sq/)
  - [4th-place code](https://github.com/TheoViel/kaggle_foursquare)
  - [7th-place write-up (Japanese)](https://future-architect.github.io/articles/20220720a/)
- Shopee:
  - [Competition page](https://www.kaggle.com/c/shopee-product-matching/overview)
  - [1st-place analysis (Japanese)](https://masatakashiwagi.com/blog/kaggle-shopee-solution/)

**Blocking:**
- [Papadakis et al., blocking and filtering survey (arXiv)](https://arxiv.org/abs/1905.06167)
- [Sparkly (PVLDB 2023)](https://dl.acm.org/doi/10.14778/3583140.3583163)
- [DeepBlocker (PVLDB 2021)](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)
- [SC-Block](https://arxiv.org/abs/2303.03132)
- [Neural LSH blocking](https://arxiv.org/abs/2401.18064)

**Matching:**
- [Ditto (PVLDB 2021)](https://www.vldb.org/pvldb/vol14/p50-li.pdf)
- [R-SupCon](https://dl.acm.org/doi/fullHtml/10.1145/3487553.3524254)
- [Sudowoodo](https://github.com/megagonlabs/sudowoodo)
- [Splink](https://moj-analytical-services.github.io/splink/index.html)
- [ZeroER](https://arxiv.org/abs/1908.06049)
- [py_entitymatching](https://github.com/anhaidgroup/py_entitymatching)
- [dedupe](https://docs.dedupe.io/en/latest/API-documentation.html)

**LLMs:**
- [Peeters et al., EDBT 2025](https://www.uni-mannheim.de/media/Einrichtungen/dws/DWS_News/Documents/Peeters-Entity-Matching-using-LLMs-EDBT2025.pdf)
- [Steiner et al. 2024](https://arxiv.org/abs/2409.08185)
- [AnyMatch](https://arxiv.org/abs/2409.04073)
- [Alper (2026)](https://arxiv.org/abs/2605.25814)

**Clustering and consistency:**
- [Bipartite matching for clean–clean ER](https://arxiv.org/abs/2112.14030)
- [One-to-one matching (VLDB-J)](https://dl.acm.org/doi/10.1007/s00778-023-00791-3)
- [Saeedi et al., multi-source clustering](https://link.springer.com/chapter/10.1007/978-3-319-66917-5_19)
- [FAMER](https://dbs.uni-leipzig.de/research/projects/famer)
- [TransClean](https://arxiv.org/abs/2506.04006)

**F-measure decisions:**
- [Ye et al., ICML 2012](https://icml.cc/2012/papers/175.pdf)
- [Waegeman et al., JMLR 2014](https://jmlr.org/papers/volume15/waegeman14a/waegeman14a.pdf)
- [Optimal F-score bipartite linkage](https://arxiv.org/abs/2311.13923)

**Names, addresses and scripts:**
- [Wasi & Flaaen 2015](https://ideas.repec.org/a/tsj/stataj/v15y2015i3p672-697.html)
- [PatentsView disambiguation](https://patentsview.org/disambiguation/disambiguation-metrics)
- [libpostal](https://github.com/openvenues/libpostal)
- [IndicXlit](https://github.com/AI4Bharat/IndicXlit)
- [Multilingual E5 report](https://arxiv.org/html/2402.05672v1)

**Model licences:**
- [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
- [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/)
- [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/tree/main)
- [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- Zingg is AGPL-3.0 ([licence](https://github.com/zinggAI/zingg/blob/main/LICENSE)); avoid it for code we ship.

*Some Kaggle write-ups could not be read directly (Kaggle pages need a browser). Their details come from the Foursquare blog, team pages and team repositories linked above. Where a source did not state a final rank, none is given.*

---

## 11. Verified research update (24 September 2026)

This addendum records the checks that should guide implementation. The earlier sections remain useful, but the following points are especially important for this dataset. The local `02b` embedding result is a model-bake-off on a small fixed index, not an estimate of full-data blocking recall; it must be replaced by a full validation candidate-recall measurement.

### 11.1 Status of an exact public solution

I found no credible public solution, winning write-up, or reusable challenge-specific repository for the **Amazon ML Challenge 2026 Business Entity Resolution** task as of 24 September 2026. The official challenge window starts on 25 September, so the closest evidence is:

1. the official challenge material in this repository;
2. the 2022 Foursquare Location Matching competition, which is the closest large-scale noisy place-matching analogue;
3. the 2021 Shopee Product Matching competition, which is useful for metric-aware post-processing; and
4. primary entity-resolution papers and open-source systems.

Do not transfer a reported Foursquare or Shopee score to this challenge. Foursquare had coordinates, self-matches, a different metric, and a different source structure. The value of those solutions is their architecture, not their leaderboard numbers.

### 11.2 Four corrections/clarifications to the current plan

#### A. Treat the one-owner constraint as empirical, not as an official guarantee

The training labels show that each S2/S3 record is reused by at most one S1 record (`reuse_max = 1` over 7,638,365 links), and the training data has no cross-country links. These are extremely useful priors, but the problem statement only explicitly says that an S1 may match zero, one, or many S2/S3 records.

Use the priors in two stages:

- measure hard one-owner assignment, dirty-record argmax, and no-assignment variants on held-out S1 groups;
- apply a hard assignment only if it improves the exact macro-F0.5 score; otherwise use it only to resolve close ties.

This protects against a test-generation change while exploiting the observed structure.

#### B. Country is open-set, but country equality is still useful

Do not one-hot encode a fixed set such as `{US, India}`. Partition dynamically by the observed `country` value and use generic features such as `same_country` and missing/unknown-country flags; compute frequency/uniqueness features for names and addresses separately. The absence of cross-country training links supports a same-country primary index, but it does not prove that a cross-country test match is impossible.

France must be processed through exactly the same retrieval and inference code path. French-specific normalization should add representations; it should not create a separate hard-coded model or discard the generic path.

#### C. `candidate_pairs.tsv` is an audit contract, not an early intermediate file

The file must contain the exact candidate edges fed into the final matching model. If a cheap LightGBM removes candidates before a richer model, the saved file is the output of that cheap stage, not the raw output of every early retrieval pass. The final matches must be a subset of the saved candidates.

#### D. Test-time learning is a fair-play grey area

Using test text for deterministic normalization and retrieval is different from learning pseudo-labels, thresholds, or model parameters from the test set. The safe submission should be train-only. If we test French self-training, keep it as a separately documented ablation and only ship it after confirming that the organisers regard it as allowed.

### 11.3 A metric-aware decision rule for this exact metric

The final decision should not be one global probability threshold. Let an S1 entity have `m` true dirty matches, and let its ranked candidate probabilities be:

```text
p1 >= p2 >= ... >= pk
```

If the probabilities are calibrated and approximately conditionally independent, and we select the first `j` candidates, the plug-in expected F_beta is:

```text
E[F_beta | j] ~= (1 + beta^2) * sum(p1..pj) / (m + beta^2 * j)
```

For this challenge (`beta = 0.5`):

```text
E[F_0.5 | j] ~= 1.25 * sum(p1..pj) / (m + 0.25 * j)
```

Therefore, for every S1 entity:

1. sort its candidate edges by calibrated probability;
2. evaluate `j = 0..k`;
3. choose the prefix with the best expected score;
4. handle the singleton case separately: if `m = 0`, the empty list has score 1 and every non-empty list has score 0.

The value of `m` is unknown at inference time. Estimate it with a combination of:

- the sum of calibrated edge probabilities;
- an S1-level `has_match` classifier;
- a learned cardinality distribution or a validation-derived prior by country/source/cardinality slice.

This rule is a better starting point than a single threshold because it naturally allows a high-confidence first match, a second marginal match, and an empty singleton decision to coexist. It must still be tuned on out-of-fold predictions because the independence and calibration assumptions are only approximations. The general F-measure theory is covered by [Waegeman et al.](https://jmlr.org/papers/v15/waegeman14a.html), and the one-owner linkage variant by [Bai et al.](https://arxiv.org/abs/2311.13923).

### 11.4 Recommended implementation architecture

| Stage | Recommended first version | Why it fits this data |
|---|---|---|
| Normalization | Preserve raw, Unicode/NFKC text, accent-folded text, and transliterated text. Split legal forms, aliases/DBA markers, domains, house numbers, postal codes, cities, and states. Never overwrite the raw representation. | The sample pairs contain native scripts, domain names, legal suffixes, aliases, missing addresses, component reordering, and corrupted house numbers. |
| Blocking | Per dynamic country partition, union: (1) character n-gram TF-IDF/BM25 on `name_core`; (2) address TF-IDF and number/street keys; (3) multilingual mE5-small dense retrieval on name, address, and combined text, using the model's required query/passage prefixes consistently; (4) exact/domain/phonetic high-precision keys. | Lexical retrieval catches typos and domains; address retrieval catches unrelated names; dense retrieval handles scripts and semantic/format variation. No single pass covers all observed cases. |
| Candidate direction | Query S2/S3 records against an S1 index, then invert the edges into the required per-S1 lists. Retain a reverse/ S1-side pass as a diagnostic for crowded S1 entities. | The training labels show one owner per dirty record, making dirty-to-S1 retrieval natural. The reverse pass protects against a single crowded query dominating top-k. |
| Matcher | LightGBM/XGBoost/CatBoost on string, address, missingness, source, rank, margin, retrieval-pass, and embedding features. Train with hard negatives and a group-aware validation split. | GBDT is fast, interpretable, robust to mixed feature types, and was the backbone of several top Foursquare solutions. |
| Owner ranker | A second listwise head grouped by each S2/S3 query, with an explicit reject option; use LambdaRank or a softmax/listwise objective. | It represents the observed one-owner structure and makes competition/margin features first-class, unlike independent binary decisions on easy random negatives. |
| Refinement | Fine-tuned multilingual cross-encoder (`microsoft/mdeberta-v3-base` or `xlm-roberta-base`) on the uncertain band only. mDeBERTa covers French and Hindi but not every script present in the data, so retain an XLM-R/mE5 path. | Ditto-style cross-encoders are strong on hard/dirty pairs, but scoring every candidate pair is too slow for this dataset. |
| Decision | Calibrate probabilities, use per-S1 expected-F0.5 prefix selection, then compare no assignment, argmax owner, and sparse global assignment. | The score is macro-averaged per S1 and precision-heavy; pair-wise accuracy alone is not the objective. |
| Consistency | Conservative S2↔S3 evidence and transitive-consistency checks. Prefer removing or downgrading an inconsistent edge; graph logic may only re-score/remove IDs already present in the final candidate file, never silently add an un-audited ID. | Graph evidence can improve precision, but one false edge can chain unrelated records. |

Use retrieval adaptively by evidence quality: dense/name retrieval should carry more weight when the address is missing or weak, while address-only retrieval should carry more weight when the name is replaced or corrupted. Tune k and score normalization per slice rather than using one global cutoff.

For a small unresolved subset with no credible same-country candidate, a cross-country dense retrieval fallback can be tested as a safety net. It should not be added blindly: the training data contains zero cross-country links, so the fallback must be accepted only if it improves a country-held-out validation proxy.

Do not assume that a nominal cross-encoder probability band such as `0.05 < p < 0.95` is small. With 150–250M candidate edges it can still contain tens of millions of pairs; measure the band on out-of-fold data and score only a bounded, high-value subset.

Normalization should be reversible at the feature level: retain raw, accent-folded, and transliterated views; do not remove potentially discriminative legal tokens such as `co`/`sa` everywhere; and do not treat `Groupe` or ambiguous abbreviations (`R`, `St`, `Av`, `Bd`) as globally equivalent without country context.

### 11.5 Features that deserve priority

#### Name and identity

- normalized/core exact equality, token Jaccard, containment, token-set ratio;
- RapidFuzz edit ratio, Jaro-Winkler, LCS, prefix/suffix similarity;
- character 3/4-gram cosine and overlap on several representations;
- legal-form equality/conflict (`Inc`, `Private`, `Limited`, `SARL`, `SAS`, etc.);
- alias/DBA/`trading as` marker agreement;
- domain stem versus the other record's core name;
- phonetic similarity as a low-weight feature, not a hard rule;
- script indicator, accent-folded similarity, transliterated similarity, and name-length ratios;
- first/last-token and initial/acronym features for reordered or abbreviated names.

#### Address

- normalized exact and component-set similarity;
- house-number exact, edit-distance, prefix, conflict, and missing flags;
- postal-code exact/prefix/missing/conflict features;
- city/state fuzzy agreement and native-script variants;
- street-token overlap after abbreviation expansion;
- PO box, unit, landmark, and injected-wrapper flags;
- address TF-IDF and dense-embedding similarities.

The observed examples show why an exact house-number key must not be the only address blocker: a true pair can have a different or corrupted number, while a non-match can share a building or common address.

#### Candidate/graph structure

- source indicator (S2 versus S3);
- rank of the S1 candidate for each dirty record;
- score gap to the best and second-best S1;
- candidate count and retrieval entropy;
- name/address frequency, IDF, block size, postal/street collision count, and other uniqueness features;
- which retrieval passes generated the edge;
- number of accepted S2/S3 records for the S1;
- maximum similarity to other dirty records assigned to the same S1;
- S2↔S3 similarity and consistency features;
- name-agreement/address-disagreement interaction features.

Do not use the S1 ID as a categorical feature: every test S1 identity is effectively unseen, and this would encourage memorization rather than text matching.

### 11.6 Training and validation design

1. Build a group-aware split by S1 entity before fitting dictionaries, IDF, thresholds, or calibration.
2. Generate candidates for the held-out groups using parameters learned only on the training folds. Include an always-empty baseline; on the full training set it should score approximately **0.0558**, the singleton fraction, and is a useful metric sanity check.
3. Include all positive links and several negative types: random, same-name/different-address, same-address/different-name, near-duplicate, missing-address, common-name, and same-source distractors.
4. Train a pair classifier and a per-dirty-record ranking model. The latter matches the observed one-owner structure and supplies useful rank/margin features. For pair/cross-encoder training, randomize left/right order and source order so the model does not memorize position or source-specific formatting.
5. Calibrate on out-of-fold predictions, separately for S2 and S3 at minimum.
6. Report macro-F0.5 and the following slices: singleton, cardinality 1, cardinality 3–4, maximum cardinality, native script, domain name, address-only, missing address, legal/DBA alias, crowded address, and France proxy.
7. Measure blocking with both edge recall and **full-list recall**: an S1 with five true matches can lose a full point if only four are retrieved. Run the first full-index comparison on at least 50K–100K realistic queries per country before trusting a candidate cap.
8. Use leave-one-country-out tests (train US, evaluate India and vice versa) as a distribution-shift stress test, not as a literal France proxy. Add native-script versus Latin holdouts inside India and a France-like transformation proxy. Do not use test labels or test-derived pseudo-labels.

A lightweight normalized overlap audit on the supplied files found **zero** exact `(lowercased/whitespace-normalized name, address, country)` matches between test records and train S1. It did find substantial name reuse (about 33.8% of test S1 names collide with a train-S1 name), while address collisions were much rarer. This argues strongly against name-only lookup and for frequency/IDF/uniqueness features; it is not a semantic-duplicate audit.

The current address EDA also needs a parser audit: a contiguous 5+ digit substring can be a house number rather than a ZIP/PIN, so recompute completeness and blocking statistics with country-aware house/postal/city parsing before drawing conclusions.

### 11.6.1 Reverse-engineer the synthetic generator and sibling structure

The observed noise appears systematic, so quantify it rather than hand-waving it. On training positives and sampled distractors, estimate by source/country:

- wrapper/prefix rates and legal-form substitutions;
- typo, token shuffle, duplicate-token, accent, and script-transliteration rates;
- address component deletion/reordering and house-number corruption rates;
- domain-name, alias/DBA, missing-address, and address-only rates;
- whether a transformation is more common in S2 than S3.

Use the resulting operator probabilities for **training-only augmentation** and hard-negative generation. Do not assume a rule generalizes to France until it survives the country/script stress tests.

Also build high-precision S2↔S3 sibling candidates (same normalized address, domain stem, or very close name/address). These links are valuable evidence for the shared owner, but they are not ground truth: franchises, shared buildings, and repeated business names can create false siblings. Feed sibling scores into the listwise matcher and graph-consistency step rather than unconditionally unioning components.

### 11.7 Recent research that changes the priority order

| Work | Main idea | Practical implication here |
|---|---|---|
| [SC-Block](https://arxiv.org/abs/2303.03132) | Supervised contrastive embeddings for blocking; reports candidate sets at 99.5% pair completeness and large runtime reductions. | A high-upside second-stage blocker because we have millions of positive links. Fine-tune a multilingual encoder only after the union baseline works. |
| [DIAL](https://arxiv.org/abs/2104.03986) | Jointly learns a transformer blocker and matcher with active learning. | Confirms that blocker and matcher objectives should be tuned together; manual active labelling is lower priority than hard-negative mining here. |
| [WDC Products](https://arxiv.org/abs/2301.09521) | Explicitly evaluates corner cases and unseen entities; all tested matchers degrade on unseen entities. | France is not merely a new country label; it is an out-of-distribution generalization test. Do not judge it only on US/India validation. |
| [TransClean](https://arxiv.org/abs/2506.04006) | Uses transitive consistency to find false positives in multi-source matching; reports an average +24.42 F1 in its experiments. | Promising precision-oriented post-processing, but test conservatively because its setting and metric differ. |
| [Heterogeneity in Entity Matching](https://arxiv.org/abs/2508.08076) | Separates representation and semantic heterogeneity and reviews robustness limits. | Supports keeping multiple text representations and evaluating script/format slices rather than one aggregate score. |
| [Entity Resolution in Practice](https://arxiv.org/abs/2607.26298) | 2026 preprint: no single matcher wins everywhere; precision and recall need separate fixes; graph merges must be re-verified. | Strong justification for a model bake-off, diverse blockers, and a final consistency audit. Treat it as a preprint, not settled guidance. |
| [ALER](https://arxiv.org/abs/2601.20664) | 2026 preprint using frozen bi-encoder embeddings, clustered active learning, and a lightweight classifier. | A possible way to spend a small manual-labeling budget on the most informative errors; not a Day-1 priority. |
| [Alper](https://arxiv.org/abs/2605.25814) | 2026 preprint combining graph refinement, label propagation, and budgeted LLM pair queries. | Research direction for a late-stage experiment; an LLM cannot be the main scorer for millions of pairs. |
| [UniBlocker](https://arxiv.org/abs/2404.14831) | Universal dense blocking pre-trained on a large external tabular corpus; sparse+dense ensembles improve recall. | Use the ensemble lesson only. Do not ship its external-pretraining checkpoint or corpus without organizer approval. |
| [Entity Matching using LLMs](https://arxiv.org/abs/2310.11244) and [AnyMatch](https://arxiv.org/abs/2409.04073) | LLMs/small fine-tuned models can be relatively robust to unseen entities, but prompts and inference cost matter. | Reserve an allowed ≤8B model for a few thousand high-impact uncertain pairs or error analysis, not the full candidate set. |

### 11.8 What I would submit if time is short

**Minimum viable high-quality path:**

1. dynamic country partitions and fold-safe normalization;
2. union of address, character TF-IDF, and mE5-small retrieval;
3. hard-negative LightGBM with rank/margin and missingness features;
4. per-S1 expected-F0.5 selection with a singleton gate;
5. validation, candidate audit file, and output validator.

**Only after that works:**

- multilingual cross-encoder on uncertain pairs;
- supervised contrastive/ArcFace retriever;
- one-owner assignment;
- conservative graph consistency;
- a tiny LLM-assisted error-analysis pass.

This ordering spends the limited compute on the two places where errors are most costly: losing a true edge during blocking and creating a false merge during the precision-heavy final decision.

### 11.9 Submission discipline

The official rules allow at most five submissions per day. Reserve submissions for materially different, locally validated configurations (for example: rules baseline, union blocker + GBDT, calibrated decision, and one final neural/graph variant). Log the data fingerprint, model revision, candidate parameters, thresholds, validation score, and output hash for every run. Do not use the public leaderboard as the only France validation signal.

### 11.10 Additional verified sources

- [Foursquare organizer write-up on the winning approaches](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/)
- [Foursquare 1st-place solution](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution)
- [Foursquare 3rd-place solution](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/psi-3rd-place-solution)
- [Foursquare 7th-place write-up](https://future-architect.github.io/articles/20220720a/)
- [Shopee solution analysis](https://masatakashiwagi.com/blog/kaggle-shopee-solution/)
- [Splink documentation](https://moj-analytical-services.github.io/splink/)
- [Dedupe API documentation](https://docs.dedupe.io/en/latest/API-documentation.html)
- [py_entitymatching](https://github.com/anhaidgroup/py_entitymatching)
- [DeepBlocker code](https://github.com/qcri/DeepBlocker)
- [Multilingual E5 model card](https://huggingface.co/intfloat/multilingual-e5-small)
- [mDeBERTa-v3-base model card](https://huggingface.co/microsoft/mdeberta-v3-base)
- [XLM-R base model card](https://huggingface.co/FacebookAI/xlm-roberta-base)
- [BGE-M3 model card (MIT; optional hybrid retriever)](https://huggingface.co/BAAI/bge-m3)
- [MultiEM code](https://github.com/ZJU-DAILY/MultiEM)
- [Continuous Filtering Benchmark code](https://github.com/gpapadis/ContinuousFilteringBenchmark)
- [BEACON code](https://github.com/nbpulsone/BEACON)
- [BI+CE address-matching code](https://github.com/avduarte333/adress-matching)

---

## 12. Additional recent literature and address-specific evidence

The following sources reinforce the same conclusion: the strongest practical system is a redundant **retrieve → cheap rank → selective cross-encode → calibrated resolve** pipeline. They also expose several claims that should not be copied without local validation.

### 12.1 Multi-source, filtering, and domain-shift research

| Work | Evidence | How to adapt it |
|---|---|---|
| [MultiEM](https://arxiv.org/html/2308.01927v1) | An unsupervised multi-table pipeline using attribute selection, Sentence-BERT, mutual top-k HNSW retrieval, table-wise hierarchical merging, and density pruning. It evaluates multi-source data up to the millions of records. | Use mutual top-k/ANN and multi-source reconciliation ideas. Do not copy its transitive merging blindly: our S1 identities are fixed, S2/S3 have a one-owner prior, and one false edge can chain unrelated businesses. |
| [Benchmarking Filtering Techniques for Entity Resolution](https://arxiv.org/html/2202.12521v5) | A systematic comparison of blocking workflows, sparse nearest-neighbor joins, and dense nearest-neighbor methods. Configuration/tuning has a large effect; sparse cardinality-based methods are strong baselines. | Establish carefully tuned BM25/TF-IDF, character n-gram, exact-key, and dense baselines before claiming that a neural blocker is better. Optimize recall at a candidate-count budget. |
| [A Deep Dive Into Cross-Dataset Entity Matching](https://openproceedings.org/2025/conf/edbt/paper-224.pdf) | Cross-dataset evaluation finds that fine-tuned small models can match prompted large models and that data-centric methods outperform model-centric approaches. | Prioritize hard negatives, domain mixtures, and calibration over a 7B model. This paper is CC BY-NC-ND; use the research result, not copied paper/code artifacts. |
| [BEACON](https://arxiv.org/html/2603.11391v2) | Budget-aware, distribution-aware sampling of labeled out-of-domain pairs using candidate embeddings. | We already have millions of labels, so use the principle to spend manual review on disagreements, rare scripts, likely France cases, and graph conflicts—not to build a large annotation loop. |
| [Beyond Scale and Generation](https://arxiv.org/abs/2607.24688) | A 2026 preprint with 1,215 controlled fine-tuning runs reports a consistent cross-encoder advantage, while larger models are not uniformly better and can exploit shortcuts. | Use a cross-encoder for the shortlist, not as a universal large-model replacement. Treat the result as preliminary and validate on our slices. |
| [ComEM](https://aclanthology.org/2025.coling-main.8/) | Compares LLM matching, pairwise comparison, and global candidate selection; selection benefits from record relationships but is sensitive to candidate order. | If an LLM is ever used, permute/repeat candidate order and audit position bias. It is not a replacement for the calibrated listwise model. |
| [GLEAM](https://dl.acm.org/doi/10.1145/3802066) | A 2026 framework combining graph blocking, adaptive connector logic, and bounded hierarchical LLM comparison. | Borrow adaptive throttling and graph representation only. The reported LLM setup is not an allowed direct solution for this challenge. |

### 12.2 Address-matching studies

- [BI+CE: Improving Address Matching using Siamese Transformer Networks](https://arxiv.org/html/2307.02300v1) uses a bi-encoder to retrieve ten candidates and a cross-encoder to rerank them. It reports strong door-level results on Portuguese addresses and roughly 4.5× GPU speedup over BM25 in its setting. This supports our retrieve-then-rerank design, but the country, labels, and metrics are different.
- [Contextual kNN Ensemble Retrieval for French Postal Addresses](https://ceur-ws.org/Vol-3770/paper9.pdf) combines CamemBERT/XLM-R representations continued with masked-language modeling on French address text and reports up to 96% top-10 retrieval. The data is private; use the architectural idea only, and do not import its corpus or models without permission.
- [AddrRaG](https://dl.acm.org/doi/10.1145/3748636.3764161) is a retrieval-plus-validation address framework evaluated on private French delivery data. It supports separating high-recall retrieval from final validation, not the reported numerical score.
- [AddrLLM](https://arxiv.org/abs/2411.13584) is an industrial address-rewriting system. The important lesson for us is to **preserve raw addresses and parse/validate them**, rather than silently rewriting correct source values. Its private training/deployment data and industrial assumptions are not available for this challenge.
- [Fighting Crime with Transformers](https://arxiv.org/html/2404.05632v2) finds that a well-tuned encoder with early stopping can beat generative LLM approaches for noisy multinational address parsing. This supports a constrained parser/token-classification model for house, postal, city, and region extraction if we later train one from permitted data.

### 12.3 Additional design consequences

1. **Tune the retriever, do not assume dense wins.** The filtering benchmark and the Foursquare solutions both favor a union and careful parameter selection. A dense model should be retained only if it improves full-index recall or the final OOF score.
2. **Use mutual or reciprocal evidence for high-confidence edges.** It is useful for S2↔S3 sibling links and owner tie-breaking, but not as a hard requirement for every true match.
3. **Keep graph operations constrained.** Build separate evidence graphs for S1↔S2, S1↔S3, and S2↔S3; fix S1 identities, require independent support before propagation, and split conflicting groups rather than merging them.
4. **Use explanation distillation only as a future research branch.** [Natural-language explanation distillation](https://aclanthology.org/2024.emnlp-main.352/) reports improved out-of-domain F1, but generating teacher explanations with an external LLM would create a fair-play, privacy, cost, and reproducibility burden. A permitted local model and train-only rationales would be required.
5. **Keep address parsing separate from identity matching.** A parser can create useful fields and training labels, but a wrong parse must be visible to the matcher through missing/conflict features rather than silently becoming ground truth.
6. **Do not use private address corpora or external registries.** The French address papers are useful evidence about retrieval/reranking and French encoders, not data sources for the submission.

### 12.4 Revised priority after the newer literature

```text
P0  full-index sparse + address + dense blocker bake-off
P0  group-safe OOF metric, calibration, and hard-negative mining
P1  listwise owner ranker + singleton/cardinality head
P1  S2↔S3 sibling evidence and conservative graph repair
P2  multilingual cross-encoder on a measured shortlist
P2  contrastive retriever fine-tuning
P3  local-LLM hard-case selection/explanation experiments
```

This ordering follows the papers above: data-centric improvements and calibrated retrieval/reranking come before expensive or unvalidated generative models.


