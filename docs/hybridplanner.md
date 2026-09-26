# Hybrid Planner — anyascii (Latin) + IndicXlit (native scripts) in this pipeline

**Audience:** a coding agent with access to ONLY this folder (`challenge-baseline/`).
**Goal:** route native-script text through the IndicXlit neural transliterator and keep
anyascii for all Latin text (French/English), without changing any existing behavior
for Latin. **Measured basis:** on a 3,200-S1 simulation this hybrid moved macro-F0.5
0.8114 → 0.8246 (+0.013 overall, +0.020 India) vs anyascii-only; word-level wins like
`nmste→namaste`, `srma→sharma`, `mharastr→maharashtra`. Full evidence:
`report.md` / `comparison.md` at the workspace root (not visible to you — this file
is self-contained; trust the numbers and verify everything below yourself).
**First read:** `README.md`, `docs/ARCHITECTURE.md`, `docs/FILES.md` (settings table),
`docs/SUBMISSION_STEPS.md`. Your tunables live in each notebook's first code cell.

**Non-negotiable constraints**
- `country` is NEVER a model feature (France is test-only and must flow through the
  identical code path).
- Outputs (`processed/`, `*.tsv`, `*.executed.ipynb`) are git-ignored — never commit them.
- TSVs: `sep="\t"`, UTF-8, `\n` newlines, exact headers; validate with
  `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids` → must print `PASS`.
- Licenses: IndicXlit is MIT, anyascii is ISC — both shippable. NEVER introduce
  `unidecode` (GPL) or libpostal/IndicXlit-training-data derivatives beyond the
  inference checkpoint named below.
- Do NOT use `fairseq` (un-buildable; archived upstream). Inference below is pure
  `torch` + `indic-nlp-library` (pure Python). Both install with plain `pip install`.

---

## 0. Design (read this once — everything else follows)

```
raw business_name / business_address
  │
  ├─ has native-script char? ──YES──▶ IndicXlit on the RAW word (pre accent-strip)
  │                                     │  Latin output, e.g. नमस्ते → namaste
  │                                     ▼
  │                              rejoin the NORMAL pipeline
  │                              (lower → junk-strip → legal split → …)
  │
  └─ NO (Latin incl. French) ──▶ existing code path, BYTE-IDENTICAL to today
                                 (accent-strip → … → anyascii as now)
```

Why this shape (each point was measured, not guessed):

1. **Only IndicXlit moves before accent-strip; anyascii stays put.** 01's
   accent-stripper deletes Unicode `Mn` marks — which includes Indic vowel signs
   AND viramas (`స్మార్ట్`→`స్మార్ట`). Measured: transliterating post-strip input
   gives `samaarata` instead of `smart`. anyascii, conversely, is position-invariant
   (measured identical outputs raw vs stripped: `nmste` both ways) because its
   failure is tabular (no inherent-`a`, no conjuncts), not positional. Moving
   anyascii buys zero and invalidates all 57 calibrated matcher features.
2. **The rejoin is free.** Xlit output is plain Latin, so embeddings, fuzzy features
   and downstream `anyascii()` calls (identity on ASCII) work unchanged.
3. **Neural model never touches Latin.** It was trained indic→en only; Latin input
   yields garbage (verified). French/English stay 100% on the old path.

---

## 1. Assets to vendor (do this first)

### 1.1 Model checkpoint (official, MIT)

- Download (124,533,637 bytes — verify size after download):
  `https://github.com/AI4Bharat/IndicXlit/releases/download/v1.0/indicxlit-indic-en-v1.0.zip`
- Unzip yields `corpus-bin/dict.<lang>.txt` (24 files) + `transformer/indicxlit.pt`
  (135,825,281 bytes). On Kaggle, attach the unzipped folder as a dataset and point
  `XLIT_DIR` (new first-cell env var, §3) at it — do NOT re-download per run.
- What it is: 6+6-layer transformer, emb 256, 4 heads, FFN 1024, GELU, ~11M params,
  char-level vocab. Joint source vocab = 780 types (all languages share one file
  content — verified byte-identical md5 across `dict.hi/te/ta/ml/mr.txt`); English
  target vocab = 26 lowercase letters + 2 `madeupword` paddings = 28 types.

### 1.2 Python deps (add to install cells next to existing `anyascii` handling)

```
pip install indic-nlp-library   # pure python, for pre-normalization (same call the
                                # official engine makes — NOT optional, see §2.1)
```
`torch` and `rapidfuzz` already exist in this pipeline's environment.

---

## 2. Inference engine to create: `xlit_torch.py` (new file, repo root)

Create this file verbatim (it replicates fairseq v0.12.2 transformer math exactly —
each non-obvious choice cites its source). It has NO fairseq import.

```python
"""Pure-torch inference for IndicXlit indic->en (no fairseq needed)."""
import math
import torch
import torch.nn.functional as F

D, H, HD = 256, 4, 64
SCALE = math.sqrt(D)          # embed_scale: no_scale_embedding=False in checkpoint
EPS = 1e-5                    # fairseq LayerNorm default
PAD, EOS, UNK = 1, 2, 3       # fairseq Dictionary specials: <s>=0,<pad>=1,</s>=2,<unk>=3


def load_dict(path):
    syms = ["<s>", "<pad>", "</s>", "<unk>"]
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                syms.append(line.rsplit(" ", 1)[0])
    return syms


# fairseq multilingual appends __lang__ tokens in SORTED-lang order AFTER the dict
# (source: multilingual_translation.py::prepare — sorted_langs). Base = 780 + 4.
SORTED_LANGS = sorted("as bn brx en gom gu hi kn ks mai ml mni mr ne or pa sa sd si ta te ur".split())
BASE_SRC = 780 + 4


def lang_idx(lang):
    return BASE_SRC + SORTED_LANGS.index(lang)   # e.g. __hi__ -> 790


def sinusoid(n):
    # fairseq SinusoidalPositionalEmbedding, positions START AT 2 (padding_idx+1).
    # Starting at 0 silently degrades every output — verified failure mode.
    half = D // 2
    freqs = torch.exp(torch.arange(half).float() * -(math.log(10000) / (half - 1)))
    pos = torch.arange(2, 2 + n).float().unsqueeze(1)
    e = pos * freqs.unsqueeze(0)
    return torch.cat([torch.sin(e), torch.cos(e)], dim=1)


def _mha(x, kv, p, mask=None):
    nq, nk = x.shape[0], kv.shape[0]
    Q = (x @ p["q"].T + p["qb"]).view(nq, H, HD).permute(1, 0, 2) * (HD ** -0.5)
    K = (kv @ p["k"].T + p["kb"]).view(nk, H, HD).permute(1, 0, 2)
    V = (kv @ p["v"].T + p["vb"]).view(nk, H, HD).permute(1, 0, 2)
    s = Q @ K.transpose(-1, -2)          # (H, nq, nk) — per-head scores, NOT (nq, nk)
    if mask is not None:
        s = s + mask
    o = (torch.softmax(s, dim=-1) @ V).permute(1, 0, 2).reshape(nq, D)
    return o @ p["o"].T + p["ob"]


def gelu_ffn(x, w1, b1, w2, b2):          # exact erf-GELU (fairseq utils.gelu)
    h = x @ w1.T + b1
    return (0.5 * h * (1.0 + torch.erf(h / math.sqrt(2.0)))) @ w2.T + b2


class XlitModel:
    def __init__(self, ckpt_path, src_dict_path, tgt_dict_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
        self.sd = sd
        self.src_syms = load_dict(src_dict_path)
        self.tgt_syms = load_dict(tgt_dict_path)
        self.src_id = {s: i for i, s in enumerate(self.src_syms)}
        assert len(self.src_syms) == 784 and len(self.tgt_syms) == 32
        self.nl = max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.layers.")) + 1

    def _L(self, pre, i, name):
        s = self.sd
        return (lambda x: F.layer_norm(x, (D,), s[f"{pre}.layers.{i}.{name}.weight"],
                                       s[f"{pre}.layers.{i}.{name}.bias"], EPS))

    def _P(self, pre, i, attn):
        s = self.sd
        return {k: s[f"{pre}.layers.{i}.{attn}.{k}{suf}"]
                for k, suf in [("q", ".q_proj.weight"), ("qb", ".q_proj.bias"),
                               ("k", ".k_proj.weight"), ("kb", ".k_proj.bias"),
                               ("v", ".v_proj.weight"), ("vb", ".v_proj.bias"),
                               ("o", ".out_proj.weight"), ("ob", ".out_proj.bias")]}

    @torch.no_grad()
    def encode(self, ids):
        s = self.sd
        x = s["encoder.embed_tokens.weight"][ids] * SCALE + sinusoid(len(ids))
        x = F.layer_norm(x, (D,), s["encoder.layernorm_embedding.weight"],
                         s["encoder.layernorm_embedding.bias"], EPS)
        for i in range(self.nl):
            x = x + _mha(self._L("encoder", i, "self_attn_layer_norm")(x),
                         self._L("encoder", i, "self_attn_layer_norm")(x),
                         self._P("encoder", i, "self_attn"))
            x = x + gelu_ffn(self._L("encoder", i, "final_layer_norm")(x),
                             s[f"encoder.layers.{i}.fc1.weight"], s[f"encoder.layers.{i}.fc1.bias"],
                             s[f"encoder.layers.{i}.fc2.weight"], s[f"encoder.layers.{i}.fc2.bias"])
        # final encoder.layer_norm EXISTS despite pre-norm (fairseq v0.12.2 source
        # applies it under normalize_before — verified in checkpoint keys).
        return F.layer_norm(x, (D,), s["encoder.layer_norm.weight"],
                            s["encoder.layer_norm.bias"], EPS)

    @torch.no_grad()
    def _decode_logits(self, ids, enc):
        s = self.sd
        n = len(ids)
        x = s["decoder.embed_tokens.weight"][ids] * SCALE + sinusoid(n)
        x = F.layer_norm(x, (D,), s["decoder.layernorm_embedding.weight"],
                         s["decoder.layernorm_embedding.bias"], EPS)
        causal = torch.triu(torch.full((n, n), float("-inf")), 1).unsqueeze(0).expand(H, n, n)
        for i in range(self.nl):
            x = x + _mha(self._L("decoder", i, "self_attn_layer_norm")(x),
                         self._L("decoder", i, "self_attn_layer_norm")(x),
                         self._P("decoder", i, "self_attn"), causal)
            x = x + _mha(self._L("decoder", i, "encoder_attn_layer_norm")(x), enc,
                         self._P("decoder", i, "encoder_attn"))
            x = x + gelu_ffn(self._L("decoder", i, "final_layer_norm")(x),
                             s[f"decoder.layers.{i}.fc1.weight"], s[f"decoder.layers.{i}.fc1.bias"],
                             s[f"decoder.layers.{i}.fc2.weight"], s[f"decoder.layers.{i}.fc2.bias"])
        x = F.layer_norm(x, (D,), s["decoder.layer_norm.weight"],
                         s["decoder.layer_norm.bias"], EPS)
        out = x @ s["decoder.output_projection.weight"].T
        if "decoder.output_projection.bias" in s:   # absent in v1.0 — KeyError trap
            out = out + s["decoder.output_projection.bias"]
        return out

    def encode_word(self, word, lang):
        # model input: [__lang__, c1..cn, __lang__] — token at BOTH ends because
        # fairseq TransformEos replaces the trailing </s> with the src langtok,
        # and the shipped pre-processor prepends it. Both verified from sources.
        ids = [lang_idx(lang)] + [self.src_id.get(c, UNK) for c in word] + [lang_idx(lang)]
        return self.encode(torch.tensor(ids))

    @torch.no_grad()
    def greedy(self, word, lang, max_len=60):
        enc = self.encode_word(word, lang)
        out = [EOS]                               # decoder seeded with </s>=2
        for _ in range(max_len):                  # (decoder_langtok=False in ckpt)
            nxt = int(self._decode_logits(torch.tensor(out), enc)[-1].argmax())
            if nxt == EOS:
                break
            out.append(nxt)
        return "".join(self.tgt_syms[i] for i in out[1:])
```

### 2.1 Word-level wrapper (new function, lives next to the engine or in notebook 01)

```python
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
_nf = IndicNormalizerFactory()
_norms = {}
def _norm(lang, w):
    if lang not in _norms:
        try: _norms[lang] = _nf.get_normalizer(lang)
        except Exception: _norms[lang] = _nf.get_normalizer("hi")
    return _norms[lang].normalize(w)

SCRIPT_LANG = [((0x0900,0x097F),"hi"), ((0x0C00,0x0C7F),"te"), ((0x0B80,0x0BFF),"ta"),
  ((0x0D00,0x0D7F),"ml"), ((0x0C80,0x0CFF),"kn"), ((0x0980,0x09FF),"bn"),
  ((0x0A80,0x0AFF),"gu"), ((0x0A00,0x0A7F),"pa"), ((0x0B00,0x0B7F),"or"),
  ((0x0D80,0x0DFF),"si"), ((0x0600,0x06FF),"ur")]
def script_lang(w):
    for (a,b), l in SCRIPT_LANG:
        if any(a <= ord(c) <= b for c in w): return l
    return None   # unknown script -> caller falls back to anyascii path
```

Transliterate per word with a dict cache (`{(word,lang): latin}`); Devanagari-script
words whose language is unknown use `"hi"` (the paper's own zero-shot precedent).

---

## 3. Hook points in the existing notebooks

New first-cell env vars (follow the existing ALL-CAPS pattern):
`XLIT_DIR` (path to unzipped checkpoint; empty/None = hybrid disabled, today's
behavior exactly), `XLIT_LANGS` (default: all 21 — restrict while debugging).

### 3.1 Notebook 01 (`01_eda_preprocessing.ipynb`) — the ONLY preprocessing change

In `normalize_name` and `normalize_address`, at the very top (on the RAW string,
before `base_clean`/accent-strip), insert:

```python
if XLIT_MODEL is not None and _has_native_script(raw):
    raw = _xlit_to_latin(raw)   # native words -> IndicXlit, Latin words passthrough
# ... rest of the function 100% unchanged
```

`_has_native_script` = search for any char outside Latin/Latin-Extended/punctuation
(see §4.1 for the only safe way to build that regex). `_xlit_to_latin` splits on
whitespace, passes Latin words through untouched, transliterates the rest via §2.1
with the unique-word cache. Everything downstream of the branch (lowercasing, junk
regexes, legal-form split, `name_lat`, soundex, address parsing, flags) runs
UNCHANGED on the now-Latin text.

### 3.2 Notebook 02_full (`02_full_e5_buckets.ipynb`) — no logic change needed

`TEXT_MODE=fixed` already transliterates Indian-script originals via anyascii. After
§3.1, 01's cleaned fields for those rows arrive pre-romanized; keep the existing
anyascii fallback as a safety net (it is identity on ASCII). Do NOT remove it.

### 3.3 Notebook 03_full (`03_full_lightgbm_submission.ipynb`) — 2 new features

Additions only (57 → 59); never modify existing features:
- `xlit_exact`: 1 if the hybrid-Latin names are string-equal, else 0.
- `xlit_token_set`: `token_set_ratio` on the hybrid-Latin names.
Both derive from fields 01 now writes (or recompute in 03 from the same helpers —
prefer recompute via shared functions over new parquet columns to avoid schema churn).

---

## 4. Landmine catalog (every item below cost real debugging time — follow exactly)

### 4.1 The regex MUST be built programmatically

NEVER hand-write ranges like `r"^[\x00-\x24F…]*$"`. In Python `re`, `\x` consumes
**exactly 2** hex digits, so `\x24F` parses as `\x24` (`$`) + literal `F`, and
`\x1E00` as `\x1E` + `00`. The resulting class silently matched Indic scripts as
"Latin": nothing got transliterated while Latin words with symbols went through the
Hindi model (an entire simulation round scored 0.7965 vs a true 0.8246 because of
this). Build it like 01's cell does:

```python
_LATIN_RANGES = [(0x00,0x7F),(0xA0,0x24F),(0x1E00,0x1EFF),(0x2000,0x206F),(0x20A0,0x20CF)]
_CLS = "".join(re.escape(chr(a)) + "-" + re.escape(chr(b)) for a,b in _LATIN_RANGES)
RE_NATIVE = re.compile(f"[^{_CLS}]")        # .search(s) -> has native script
RE_LATIN_WORD = re.compile(f"^[{_CLS} ]*$") # .match(w)  -> fully Latin word
```

Acceptance: `smart`→Latin, `café`→Latin, `స్మార్ట్`→native, `नमस्ते`→native.
Probe these four literals in a unit assert before any run.

### 4.2 Feed ORIGINALS, never cleaned columns, to the model

01's stripper deletes viramas/vowel signs (`స్మార్ట్`→`స్మార్ట`; model then emits
`samaarata` instead of `smart`). The §3.1 branch sits before `base_clean` precisely
for this reason. If you add xlit features in 03, recompute from raw
`business_name`/`business_address`, not from `name_norm`.

### 4.3 anyascii stays exactly where it is

Measured identical outputs on raw vs stripped input (`nmste` both ways) — its
failure is tabular (no inherent-`a`, no conjuncts: `nmste/cay/mharastr`), not
positional. Moving it invalidates calibrated features for zero gain.

### 4.4 Engine numerics (each verified against fairseq v0.12.2 sources)

- Positions start at **2**, not 0. Sinusoid formula: half-dim 128,
  `freq = exp(arange(128) * -ln(10000)/127)`, row(p) = `[sin(p·f), cos(p·f)]`.
- `embed_scale = √256 = 16` (`no_scale_embedding=False` in checkpoint args).
- Final `encoder.layer_norm` / `decoder.layer_norm` ARE applied (present in
  checkpoint despite pre-norm — fairseq v0.12.2 applies them under normalize_before).
- GELU is exact erf flavor, LayerNorm eps 1e-5, attention scale `64**-0.5`, no dropout
  at inference, no output-projection bias in v1.0 (guard the key lookup).
- Attention head layout must be `(heads, seq, dim)` for scores — `(seq, heads, dim)`
  silently broadcasts to wrong-shaped garbage without raising (except at reshape).
- Decoder input starts `[2]` (`</s>`), stops at `2`.

### 4.5 Model behavior limits (design around them, don't fight them)

- Greedy quirks: doubled final vowels (`pataa`, `qilaa`, `lakhanauu`), tail repeats
  (`chaayaya`), `ड→dae` (`limitedae` on every लिमिटेड). They are DETERMINISTIC
  (same input → same output), so pairs still match each other — but prefer beam-4 +
  word-prob rescoring in production (needs `word_prob_dicts_en.zip` + beam search;
  optional, ~+6% per upstream docs).
- Weak on 2–4-char tokens (`टेक→techeca` vs anyascii `tek`) and legal abbreviations
  (`प्रा→praaye`, `लि→lilin` vs `pra. li.`). Mitigation: length/abbreviation gate —
  tokens ≤ 4 chars or in a small legal-abbrev map (`प्रा→pvt`, `लि→ltd`, …) bypass
  the model via anyascii/dictionary. Add rows like `टेक/tek` to the acceptance list.
- Devanagari ≠ Hindi: Marathi/Sanskrit/Nepali names get the `hi` token
  (approximation, paper-sanctioned). Log the script→lang decision per row while
  debugging so misrouting is visible.
- The model emits only `[a-z]` (32-type target vocab) — any non-ASCII in its output
  means YOUR post-processing is broken, a free invariant assert.

### 4.6 Scale, perf, Kaggle realities

- Reference CPU cost: ~0.12 s/word greedy (965 words in 116 s). Full data has millions
  of native tokens — but cache per UNIQUE word (vocab-bounded, not row-bounded) plus
  GPU batching makes 02's window feasible; never transliterate row-by-row in a loop
  without the cache.
- `torch.load(..., weights_only=False)` is required (checkpoint holds an argparse
  Namespace) — trusted official release asset, size-checked per §1.1.
- Windows PowerShell 5.1 in this workspace: no `tail`/`&&` (chain with
  `; if ($?) { … }`); Hindi/Devanagari console output needs
  `chcp 65001; $env:PYTHONUTF8='1';` or cp1252 crashes; `pip install` falls back to
  user site-packages (fine).

---

## 5. Build order + acceptance gates (do not skip gates)

1. **Vendor assets** (§1.1–1.2). Gate: zip size = 124,533,637 bytes; `dict.hi.txt` =
   780 lines; checkpoint loads; `encoder.embed_tokens.weight.shape == (806, 256)`.
2. **Create `xlit_torch.py`** (§2). Gate (word list, `hi` token, greedy):
   `नमस्ते→namaste`, `महाराष्ट्र→maharashtra`, `शर्मा→sharma`,
   `One फाइनेंस→One finance`-style mixed input passes Latin through. Any non-ASCII
   in output = fail. Any deviation on these anchors = protocol bug, stop and recheck
   §2 against the cited sources (don't "fix" by retraining/adjusting weights).
3. **Regex + branch unit asserts** (§4.1 + four probe words). Gate: all four route
   correctly; `XLIT_DIR=None` reproduces today's 01 outputs byte-for-byte
   (run 01 in sample mode before/after and diff `processed/`).
4. **01 integration** (§3.1). Gate: Latin rows unchanged (diff), native rows now
   Latin in cleaned fields, spot-check 20 against the §2 anchors' quality.
5. **02 rerun** (needs the buckets rebuilt — embedding text changed). Gate: train
   bucket recall/ceiling ≥ previous run's numbers (India ceiling was ~0.97).
6. **03 features + train/CV.** Gate: OOF macro-F0.5 improves on the test bed
   (regional split per `docs/SUBMISSION_STEPS.md`); France prediction rates stay near
   US/India (~94%, ~3.5/S1) — a spike means lookalike false matches, tighten
   `COUNTRY_THRESHOLDS["France"]` via notebook 04 instead of touching features.
7. **Validate + submit** per `docs/SUBMISSION_STEPS.md` (validator must print PASS
   with `--check-ids`).

**Rollback rule:** if gate 5 or 6 fails, keep the code behind `XLIT_DIR=None` default
and submit the current baseline — the hybrid is additive by construction, so
disabling it is one env var.

## 6. Out of scope (explicitly do NOT do)

- Typo correction in 01 (measured +0.004-class gain — shelved; embeddings + fuzzy
  features already cover it).
- Touching any Latin/French path, adding `country` as a feature, `unidecode`/GPL
  code, geocoding or external data, `fairseq` installation attempts.
- Committing outputs, weights, or TSVs (all git-ignored). Commit code + this doc only.
