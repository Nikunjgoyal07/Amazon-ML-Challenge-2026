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
| top finisher | Yuki Uehara | Candidates, then LightGBM filter, then 2 transformer pair classifiers, then **GNN node classification as post-processing**: score **0.907 → 0.946**. | [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/) |

**Lessons that carry over:**
- **Recall first, then precision.** Every top team built a wide candidate set from *several* retrievers, then filtered in stages with models of increasing cost: cheap GBDT, then rich GBDT, then transformer.
- **Boosted models were the usual backbone.** String-similarity features plus a boosted model carried the 1st, 2nd, 4th and 7th places. Transformers added the last points for 1st and 2nd. The 3rd place shows an all-embedding route (metric learning + bi-encoder).
- **Graph post-processing pays off.** It gave +0.04 IoU for Uehara and was a core step for the 7th-place team.
- **Several text normalisations** (character level, word level, romanised) served different stages. That is the same idea as our `name_clean` / `name_core` / native-script dictionary.
- **Check train/test overlap early.** After the competition, 67% of Foursquare test records turned out to duplicate training records. That made local validation misleading ([7th-place blog](https://future-architect.github.io/articles/20220720a/)).

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
| **Top-k TF-IDF (Sparkly)** | For each record, retrieve the k highest TF-IDF/BM25 matches on the best attribute/tokenizer (Lucene). | Outperformed 8 state-of-the-art blockers; recall 86.6–99.96% ([Sparkly, PVLDB 2023](https://dl.acm.org/doi/10.14778/3583140.3583163)). | Strong, simple baseline. Char n-grams handle typos. Needs sparse top-k machinery at our scale. |
| **Canopy / q-gram TF-IDF** | Char 4-gram TF-IDF canopies, then clustering inside each canopy. | Used for **organisation-name disambiguation** at PatentsView scale ([PatentsView](https://patentsview.org/disambiguation/disambiguation-metrics)). | Directly on business names. |
| **Self-supervised DL blocking (DeepBlocker)** | Learn record embeddings **without labels** (autoencoder, cross-tuple training), then run kNN. | Autoencoder best on structured/dirty data, Hybrid best on textual. It beat labelled AutoBlock and an industrial blocker on dirty/textual data. **Combining DL and non-DL blockers did better still** ([DeepBlocker, PVLDB 2021](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)). | Supports a *union* of blockers. We have labels, so supervised variants are also possible. |
| **Pretrained sentence embeddings + kNN** | Encode each record with a sentence-embedding model and retrieve the top-k by cosine. | Ditto's "advanced blocking": Sentence-BERT + blocked matrix multiplication, top-10 per record ([Ditto](https://www.vldb.org/pvldb/vol14/p50-li.pdf)). **Our benchmark (02b):** multilingual-e5-small reached 94.9% recall@5 on cleaned text and read native scripts directly (94.7% on raw Indian-script names vs 36–46% for MiniLM). | **Our main dense retriever.** mE5 is MIT-licensed ([report](https://arxiv.org/html/2402.05672v1), [model](https://huggingface.co/intfloat/multilingual-e5-small)). |
| **Supervised contrastive blocking (SC-Block)** | Fine-tune the encoder with supervised contrastive loss on labelled matches, then run kNN. | Smaller candidate sets at **99.5% pair completeness**. Pipelines were 1.5–2× faster, and 8× faster on a large product benchmark ([SC-Block](https://arxiv.org/abs/2303.03132)). | High upside: we have 7.6M labelled links to fine-tune mE5 with. |
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
| **One-to-one / bipartite matching** | From a similarity graph between two clean sources, pick pairs so each record matches at most one (e.g. Unique Mapping Clustering, Ricochet variants) | 8 algorithms compared on 700+ similarity graphs from 10 datasets ([Papadakis et al., EDBT 2022](https://arxiv.org/abs/2112.14030)); a VLDB-J follow-up on one-to-one matching ([link](https://dl.acm.org/doi/10.1007/s00778-023-00791-3)) | Our constraint is one-sided: an S2/S3 record has one owner, an S1 has many. The **argmax-per-record** rule is the matching step, and Unique Mapping Clustering's greedy order is a refinement. |
| **Multi-source clustering (FAMER)** | Holistic clustering across 3+ sources, including new schemes tailored to multi-source data | Distributed evaluation of 6–8 schemes ([Saeedi et al.](https://link.springer.com/chapter/10.1007/978-3-319-66917-5_19); [FAMER](https://dbs.uni-leipzig.de/research/projects/famer)) | Use S2↔S3 links as extra evidence: if an S2 and an S3 record match each other strongly, they should share an owner. |
| **Transitive-consistency cleaning (TransClean)** | Removes pairwise matches that break transitivity across sources, without labels | **+24.42 average F1** in multi-source matching over pairwise matchers ([TransClean](https://arxiv.org/abs/2506.04006)) | A direct precision booster for F0.5. It drops an S1←S2 link when the S2's near-duplicates point to a different S1. |
| **Graph centrality pruning** | Build a graph of predicted pairs, drop low-centrality edges, keep neighbours within distance 2 | Foursquare 7th place | A simple post-step to test. |
| **GNN post-processing** | Classify nodes/edges on the predicted-pair graph | Foursquare (Uehara): 0.907 → 0.946 | Worth trying once the base pipeline is stable. |
| **Neighbourhood blending (INB)** | Replace each embedding with a weighted blend of its close neighbours, iteratively | Shopee 1st place | Blend S2/S3 embeddings with their near-duplicates before assignment. |
| **Label propagation + LLM budget (Alper)** | Joint matching and clustering over one evolving graph | [arXiv 2605.25814](https://arxiv.org/abs/2605.25814) | Research-grade, recent. |

---

## 6. Choosing matches to maximize F-beta

With a per-S1 F0.5, the decision is which subset of an S1's candidates to output, including the empty set. The theory:

- **Two routes to F-measure.** "Empirical utility maximisation" tunes a threshold on validation. The "decision-theoretic approach" takes calibrated probabilities and picks the prediction with the best *expected* F. The paper also covers when each wins ([Ye et al., ICML 2012](https://icml.cc/2012/papers/175.pdf)).
- **An exact optimum exists.** It is Bayes-optimal regardless of distribution, and is computed with the General F-measure Maximizer (GFM) ([Waegeman et al., JMLR 2014](https://jmlr.org/papers/volume15/waegeman14a/waegeman14a.pdf)).
- **One-to-one linkage.** For bipartite record linkage (each record links to at most one), there is a method that estimates matches by maximising expected F-score ([arXiv 2311.13923](https://arxiv.org/abs/2311.13923)).

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
| Indic transliteration | **IndicXlit**: Roman↔Indic models for 21 languages, MIT, offline pip package `ai4bharat-transliteration` | [GitHub](https://github.com/AI4Bharat/IndicXlit), [Aksharantar paper](https://arxiv.org/abs/2205.03018) | Rules risk: trained on the external Aksharantar corpus. Our native→English dictionary is learned from the provided data without labels, covering 92.6% of test native words at 10% data. |
| Multilingual text | Multilingual encoders (mE5, XLM-R, mDeBERTa) | [mE5 report](https://arxiv.org/html/2402.05672v1) | mE5-small handled raw native scripts in our benchmark (94.7% recall@5). |
| Romanised normalisation | Separate romanised text versions for the neural models | Foursquare 7th place | Same idea as our `anyascii` fallback. |

---

## 8. Rules check: what is allowed

The challenge rules forbid external databases, APIs, geocoding and internet data augmentation. The final model must be MIT or Apache-2.0 licensed and at most 8B parameters.

| Item | Status |
|---|---|
| String similarity, TF-IDF, LightGBM/XGBoost/CatBoost, FAISS-style kNN | Allowed (code, no external data) |
| multilingual-e5-small/base/large (MIT), XLM-R / mDeBERTa (MIT), MiniLM (Apache) | Allowed (pretrained model, permissive licence) |
| Qwen2.5-7B (Apache), Mistral-7B-v0.3 (Apache), Phi-4-mini (MIT) | Allowed as models; watch the compute cost |
| Llama-3.x, Gemma | **Not allowed** (licence is not MIT/Apache) |
| Hand-written dictionaries (abbreviations, legal forms, state names) | Allowed (domain knowledge, not a lookup) |
| Dictionaries learned from provided data (our native-script map) | Allowed |
| **libpostal, IndicXlit** | **Grey area.** The licences are fine, but the models embed external training data. Ask the organisers before using them, or skip them. |
| Geocoding, business registries, web search | **Forbidden** |
| Unlabelled test inputs (e.g. learning a dictionary from test records, self-training on France) | Not external data, but transductive. Document it in the methodology. |

---

## 9. What this means for our pipeline

What we already have, and how it lines up with the evidence:

| Stage | Built so far | Evidence it's the right direction |
|---|---|---|
| Normalisation | `01_preprocess`: legal forms, aliases, domains/handles, digit-for-letter swaps, abbreviations, state codes, a native-script dictionary learned without labels | Wasi & Flaaen; Foursquare 7th (multiple normalisations) |
| Dense retriever choice | `02b` benchmark: **multilingual-e5-small** is best on every slice | Ditto advanced blocking; DeepBlocker; SC-Block |

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
5. **Fine-tune mE5 on train links** (contrastive or ArcFace with S1 IDs as classes; SC-Block, R-SupCon, Foursquare 3rd). This should improve both blocking and a strong matcher feature.
6. **Ditto-style cross-encoder** (mDeBERTa/XLM-R) on pairs where LightGBM is uncertain.
   - Serialise with `[COL]`/`[VAL]`.
   - Augment with the observed noise operators.
   - **For France:** synthesise noisy copies of *test* French S1 records (abbreviations, accent loss, typos, prefixes) as extra training pairs. This is transductive, so document it.
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
