# Understanding the results of the matcher notebook

This page explains every score that `03_full_lightgbm_submission.ipynb` prints: what each number means,
where it comes from, and how to read it. No machine-learning background is needed.

The example numbers come from an early run of the matcher on a small sample (100,000 records per source, 44
features, 5 groups). 03 full prints the same tables, with 85 features and 3 groups, at full size. Your numbers
will differ, but they mean the same things.

---

## 1. What we are measuring

For every business in S1, we need to list the S2/S3 records that are the same business, or nothing
if there are none. The work happens in two steps:

1. **Search (notebook 02):** for each S1, collect the 20 most similar-looking S2/S3 records. This
   shortlist is called its **bucket**.
2. **Judge (notebook 03):** LightGBM looks at each record in the bucket and decides whether it is
   really the same business.

The scores tell us how good the final lists are, and whether points were lost in the search step or
in the judging step.

---

## 2. The competition score: F0.5

Every score in the main table is the competition's own score, between 0 (worst) and 1 (perfect).

### For one S1 business

We compare what we predicted with the true matches, using two questions:

- **Precision:** of the records we predicted, how many are right?
- **Recall:** of the true matches, how many did we find?

**F0.5** combines the two into one number and counts precision more than recall. **A wrong match
hurts more than a missed one.**

**Example** (from the competition PDF): a business has 2 true matches. We predict 3 records and
2 of them are right.

| | |
|---|---|
| Precision | 2 of 3 predicted are right, so 0.67 |
| Recall | 2 of 2 true matches found, so 1.0 |
| **F0.5** | **0.714** |

### Businesses with no match ("singletons")

These are all-or-nothing:

- predict nothing: score **1**
- predict anything: score **0**

A single wrong guess on such a business costs its full point.

### The final number: macro F0.5

We calculate F0.5 for every S1 business and take the **average**. Every business counts equally,
whether it has 1 match or 10.

---

## 3. Where the scores come from

- We only have answers for the **training** data. The competition's ground-truth file tells us
  which S2/S3 records belong to each training S1.
- The notebook predicts matches for those training S1 and checks them against the answers.
- **No cheating:** the S1 records are split into groups (3 in 03 full, 5 in the run shown here). Each group
  is predicted by a model trained on the others, so no business is ever predicted by a model that saw it during training.
  This makes the score behave like a score on new data.

---

## 4. The main score table

Five ways of picking matches, all scored on the same S1 businesses.

| Method | Cutoff | Score | S1 with matches | Singletons | India | US |
|---|---|---|---|---|---|---|
| Predict nothing | – | 0.061 | 0.000 | 1.000 | 0.062 | 0.059 |
| Cosine rule (notebook 02) | 0.91 | 0.913 | 0.913 | 0.921 | 0.871 | 0.955 |
| LightGBM, probability ≥ cutoff | 0.58 | 0.968 | 0.967 | 0.972 | 0.951 | 0.984 |
| **LightGBM + one owner (final)** | **0.41** | **0.968** | **0.968** | **0.969** | **0.952** | **0.984** |
| Ceiling: a perfect judge | – | 0.982 | 0.981 | 1.000 | 0.967 | 0.998 |

**The columns**

| Column | Meaning |
|---|---|
| Cutoff | the probability (or similarity) a candidate needs to be kept, chosen to maximize the score |
| Score | the competition score over all S1 businesses. **This is the main number.** |
| S1 with matches | the score counting only businesses that have at least one true match |
| Singletons | the score counting only businesses with no match (the share we correctly left empty) |
| India, US | the score for each country's businesses |

### Row by row

**Predict nothing (0.061).**
- It returns an empty list for everyone.
- It still scores above 0 because about 6% of businesses are singletons, and an empty list is
  exactly right for them.
- This is the floor: anything useful must beat it.

**Cosine rule (0.913).** This is the simple rule from notebook 02:
1. Keep every candidate whose text similarity to the S1 is at least 0.91.
2. If one S2/S3 record is kept for several S1s, only the most similar S1 keeps it.

It judges on one number, so it gets fooled when two *different* businesses look alike (same street,
similar names). It is weakest in India (0.871), where names in Hindi or Tamil script and messy or
empty addresses make texts look less alike.

**LightGBM, probability ≥ cutoff (0.968).**
- LightGBM weighs many clues for each (S1, candidate) pair (44 in this early run, 85 in 03 full): name and address similarity, shared house
  numbers and postcodes, how the candidate compares with other S1s, and more.
- It outputs a probability that the pair is the same business. Candidates at or above the cutoff
  are kept.
- It beats the cosine rule everywhere, most of all in India (0.871 to 0.951).

**LightGBM + one owner (0.968). This is the final method.**
- It is the same as the row above, plus one rule from the competition: each S2/S3 record belongs to
  at most one S1. So each record is kept only for the S1 that gives it the highest probability.
- The gain is tiny because the model had already learned this rule: its strongest clue is "is this
  S1 the best fit for this candidate?"
- The cutoff can be lower (0.41 instead of 0.58) because the one-owner rule already removes some
  wrong matches. That lets the model accept less certain candidates and find more true ones.

**Ceiling: a perfect judge (0.982).**
- This is imaginary: a judge that keeps exactly the true matches found in each bucket, and nothing
  else.
- It is the best any judge could do with these buckets. It is not 1.0 because some true matches
  never made it into the bucket: the search step missed them, and nobody can pick a record they
  never see.
- Singletons score 1.0 here, since a perfect judge keeps nothing for them.

---

## 5. Where the points go

```
1.000  a perfect result
  │     −0.018  lost in the search step: true matches that never reached the bucket
0.982  ceiling (perfect judge on our buckets)
  │     −0.014  lost in the judging step: LightGBM's wrong keeps and wrong drops
0.968  our result
```

- To improve the **search** (the bigger loss, mostly in India, whose ceiling is 0.967 vs 0.998 for
  the US): better text for the embeddings, a better embedding model, or extra search methods.
- To improve the **judge**: more or better clues, more training data.

---

## 6. The cutoff table ("How the threshold trades precision for recall")

This table shows the final score at different cutoffs.

| Cutoff | Score | S1 with matches | Singletons |
|---|---|---|---|
| 0.10 | 0.964 | 0.965 | 0.948 |
| 0.40 | 0.968 | 0.968 | 0.969 |
| 0.90 | 0.962 | 0.961 | 0.989 |

- **Low cutoff:** the model keeps more candidates. It finds more true matches, but also makes more
  wrong guesses, which hurts singletons most.
- **High cutoff:** it keeps only very sure candidates. Singletons do great, but some true matches
  are dropped.
- **What to look for:** a flat middle. Here the score stays between 0.966 and 0.968 for any cutoff
  from 0.25 to 0.75, so the exact choice does not matter much. That is good news, because the
  cutoff will have to be re-chosen on the full data.

---

## 7. The country transfer table (a stand-in for France)

France has no training answers, so we cannot score it. Instead we pretend one country is unknown:
train on one country and score the other.

| Trained on | Scored on | Score | Score when trained on both |
|---|---|---|---|
| US | India | 0.938 | 0.952 |
| India | US | 0.980 | 0.984 |

- **What it means:** the model still works well on a country it has never seen, losing 0.004 to
  0.014 points. It learned general rules ("same house number and similar name means a match"), not
  country-specific tricks.
- **What to look for:** a small gap. A large drop (say 0.1 or more) would warn that France results
  will be poor.

---

## 8. Other numbers the notebook prints

**Per training group: "best iteration" and "valid log-loss"**

```
fold 0: ... best iteration 196, valid log-loss 0.0071
```

- **Best iteration:** LightGBM improves in small steps. This is how many steps helped before
  further steps stopped improving the held-out group. It only matters for training time.
- **Log-loss:** how confident and correct the probabilities are. Lower is better; 0 would be
  perfect. It is useful only for comparing two runs of the notebook: if a change raises it, the
  change made things worse.
- All 5 groups should show similar values. One very different group would suggest something odd in
  that part of the data.

**Feature importance: which clues the model relies on most**

| Clue | Share of the model's decisions | In plain words |
|---|---|---|
| `cand_margin` | 56% | Is this S1 the candidate's best fit, and by how much, compared with other S1s? |
| `score` | 18% | How similar the two texts are, according to the embedding search |
| `cand_rank` | 13% | Where this S1 ranks among all S1s that have this candidate in their bucket |
| `addr_token_set` | 4% | How many address words the two records share |
| `raw_addr_token_set` | 2% | The same, on the original address text |
| `numbers_jaccard` | 1% | Do the addresses contain the same numbers (house number, postcode)? |

The model leans most on "which S1 fits this record best", then on overall similarity, then on
address details.

---

## 9. What these scores do NOT tell you

- **They are not the leaderboard score.** The scores come from a sample. In the full data, every S1
  competes with millions of records instead of about 100,000, so matching is harder. Expect the
  leaderboard score to be lower.
- **They are slightly optimistic.** The cutoff was chosen on the same data it is scored on.
- **Only matches inside the sample count.** A business's true matches that were not sampled are
  ignored.
- **France is not tested directly.** The country transfer table is only a stand-in.

---

## 10. Quick checklist for reading a new run

1. **Final row vs cosine rule:** LightGBM should be clearly higher (here 0.968 vs 0.913).
2. **Final row vs ceiling:** the smaller the gap, the better the judge (here 0.014).
3. **Ceiling vs 1.0:** this is what the search step loses (here 0.018). It is usually worst in India.
4. **Singletons column:** it should stay high (here 0.969). If it drops, the model is guessing on
   businesses that have no match.
5. **Cutoff table:** it should be flat around the best cutoff.
6. **Country transfer:** the drop should be small (here at most 0.014).

---

## Glossary

| Term | Meaning |
|---|---|
| S1 | the clean list of businesses we must find matches for |
| S2, S3 | the messy lists the matches come from |
| Bucket | the shortlist of the 20 most similar-looking S2/S3 records for one S1 |
| Singleton | an S1 business with no true match |
| Precision | of what we predicted, the share that is right |
| Recall | of the true matches, the share we found |
| F0.5 | the competition score for one business: combines precision and recall, counting precision more |
| Macro F0.5 | the average F0.5 over all S1 businesses; the final competition score |
| Cutoff / threshold (t) | the minimum probability a candidate needs to be kept |
| One owner | the rule that each S2/S3 record goes to at most one S1 |
| Ceiling | the best score possible with our buckets, if the judge were perfect |
| Cosine | a 0 to 1 measure of how similar two texts are according to the embedding model |
| LightGBM | the model that judges each pair using many clues |
