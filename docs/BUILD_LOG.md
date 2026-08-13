# SHRUTI — Build Log

A running, honest record of what was built, what was measured, and what was decided.
Latency numbers here are copied from benchmark artifacts, never from memory or a single run.

---

## D1 — 2026-08-13

### Decisions taken before writing code

**D-1 — What "under 200ms" means, and how it is measured.**
The task asks for a retrieval-to-answer path under 200ms, and requires that latency be measured on
the deployed system. These two requirements collide with physics: the free Hugging Face Space CPU
tier is US-hosted, and a browser in Gujarat is ~230–300ms of round-trip from a US host before any
work is done. A client-side stopwatch in India therefore measures the Pacific Ocean, not the
pipeline.

Resolution — three numbers, all published, none of them hidden:

1. **Headline SLO = server-measured pipeline latency on the deployed Space.** Monotonic clock,
   stamped from request-received to Tier-1-answer-serialized. This is what "retrieval-to-answer
   path" denotes, and it is what a latency SLO means in ordinary engineering usage.
2. **Corroborated by an in-region load generator.** `bench/run.py` runs from a US GitHub Actions
   runner against the live Space, so client-observed latency sits within a few ms of the
   server-side figure. Deployed system, real HTTP, no ocean.
3. **India→US client-observed latency published separately**, with the network segment broken out
   as its own block in the waterfall.

Rejected: measuring on localhost (violates "production is the source of truth"); quietly reporting
server-side numbers as if they were end-to-end (dishonest).

**D-2 — Fresh git repo.** This directory sits inside an unrelated parent repo. Initialised a
standalone repo here on `main`; the parent's ignore list excludes this directory so nothing leaks
into an unrelated project's history.

**D-3 — Disk budget.** 16 GB free of 228 GB. See the corpus finding below — this drove a
significant design change for the better.

**D-4 — Python 3.11.** System Python is 3.14, which predates wheels for `onnxruntime`, `usearch`,
and `tokenizers`. Pinned 3.11 locally via `uv`, and the same 3.11 in the Dockerfile, so local and
deployed environments are identical rather than merely similar.

### Verified before use — no assumptions

Every external fact below was checked against the live API rather than taken from memory.

**`ai4bharat/MSMARCO-XI` — schema confirmed.** Public, ungated, parquet. Per-language files:

| Split | Files | Size each |
|---|---|---|
| `train/` | 13 languages (`tel` absent) | ~3.3–4.0 GB |
| `validation/` | 14 languages | ~0.42–0.49 GB |

Row schema (verified against a live sample row):

```
source_lang, target_lang, meta{model_name, temperature, ...}
query_id      int64        # stable across languages — the join key
query         string       # translated query
Answer        string       # translated answer
query_type    string       # DESCRIPTION | NUMERIC | PERSON | LOCATION | ENTITY
Eng_Query     string
Eng_Answer    string
passages      struct of three PARALLEL arrays:
  is_selected          [int64]    # relevance labels — the reason IR metrics are possible
  English_passages     [string]
  Translated_passages  [string]
```

**Finding that changed the plan: use the validation split as the corpus.** Validation files are
~470 MB against ~3.7 GB for train — roughly an eighth the size — and they carry the same
`is_selected` labels. Four language files (hi, gu, bn, ta) total ~1.9 GB downloaded, which fits the
disk budget with room to spare, and yields roughly 10k queries × ~10 passages × 5 languages
(English arrives free inside every file) before dedupe. That is comfortably inside the 150–250k
target passage count without touching a single multi-GB train shard.

`query_id` being stable across languages is what makes this work: English passages repeat
identically in every language file, so content-hash dedupe collapses them to one copy and the
English partition costs nothing extra.

**`minishlab/potion-multilingual-128M`** — confirmed present and ungated. `model_type: model2vec`,
`hidden_dim: 256`, `normalize: true`, tokenizer `BAAI/bge-m3`. Ships `model.safetensors` and
`onnx/model.onnx`. Matches the fast-lane design: static lookup, 256-dim, no transformer forward
pass at query time.

### Still to verify

- Sarvam Saaras v3 streaming WebSocket contract (URL, auth, audio framing, partial/final message
  shapes) — blocks D3, not D1.
- Cerebras free-tier model IDs and token ceiling — blocks D3, not D1.
- Whether Sarvam supports ephemeral client tokens. If it does, the browser can connect directly and
  skip a Pacific double-crossing for audio (browser IN → Space US → Sarvam IN). Worth real effort.

### Data-quality finding: translation degeneration

While validating the corpus builder, one query in a 400-row Hindi sample was 7,783 characters long.
Its English source was `"suit definition"` — fifteen characters. The Hindi translation had
degenerated into a loop, repeating one clause roughly ninety times. This is a known neural-MT
failure mode on short or ambiguous inputs, and MSMARCO-XI is machine-translated throughout.

Rather than guess at a filter, prevalence was measured across the sample using the fraction of
repeated word 5-grams:

| repetition ratio | passages flagged | queries flagged |
|---|---|---|
| > 0.3 | 4.22% | 0.25% |
| > 0.5 | 0.11% | 0.25% |
| > 0.7 | 0.00% | 0.25% |

**0.5 is the chosen threshold.** At 0.3 the filter begins removing legitimate repetitive text —
enumerations, tables, boilerplate — flagging 40× more passages while catching no additional
queries. The 0.7 row shows the degenerate queries are extreme outliers, not a continuum.

This matters more for retrieval than the small percentages suggest. Repeated n-grams inflate BM25
term frequencies and pull the embedding toward the looped phrase, so a degenerate passage retrieves
for far more queries than it deserves — a small fraction of the corpus doing outsized damage.

Degenerate **passages** are dropped. Degenerate **queries** are flagged in a `degenerate` column
rather than dropped: their passages are still valid corpus content and their qrels still describe
real relevance, so removing the row would break joins while fixing nothing.

### First latency reading — local, smoke corpus

Measured on the 8k-passage Hindi+English smoke corpus, 200 queries after a 20-query warmup,
on an M-series Mac. **This is not the published SLO number.** Per D-1, published numbers come from
the deployed Space measured by the benchmark harness; this is an engineering reading taken to find
out where time actually goes.

| stage | hnsw P50 | hnsw P70 | hnsw P100 | exact P50 | exact P70 | exact P100 |
|---|---|---|---|---|---|---|
| guard_safety | 0.007 | 0.009 | 0.052 | 0.006 | 0.008 | 0.026 |
| detect | 0.009 | 0.010 | 0.103 | 0.008 | 0.009 | 0.038 |
| embed | 0.089 | 0.109 | 0.512 | 0.059 | 0.065 | 0.293 |
| dense | 0.178 | 0.203 | 0.631 | 0.162 | 0.189 | 28.331 |
| bm25 | 0.138 | 0.159 | 0.560 | 0.127 | 0.143 | 0.380 |
| fuse | 0.092 | 0.105 | 0.334 | 0.090 | 0.098 | 0.267 |
| guard_scope | 0.002 | 0.002 | 0.039 | 0.002 | 0.003 | 0.023 |
| extract | 1.176 | 1.383 | 2.949 | 0.913 | 1.078 | 2.167 |
| **TOTAL** | **1.759** | **2.091** | **4.566** | **1.443** | **1.619** | **29.853** |

Three things this reading tells us:

1. **`extract` is the dominant stage**, at roughly two-thirds of total time. That is the opposite of
   the naive expectation that vector search dominates, and it is because extraction embeds every
   candidate sentence across the top passages. It is the right place to spend optimisation effort,
   and the wrong place to have guessed.
2. **exact search's 28ms P100 is a page-fault, not a computation.** The embedding matrix is
   memory-mapped, so the first query to touch a cold page pays disk. Its P50 is 0.162ms — three
   orders of magnitude lower. This is precisely what the startup warmup exists to absorb, and it is
   evidence the warmup is load-bearing rather than decorative.
3. **The 200ms budget is not the binding constraint at this scale** — the whole pipeline is under
   5ms. The real question is how this scales to the full ~500k-passage corpus, where exact search
   grows linearly (projected ~10ms) and BM25 grows with posting-list length. Both remain
   comfortably inside budget, which means the engineering effort should go to answer *quality*, not
   speed. That is a useful thing to learn on day one.

### Known quality gap, stated plainly

Retrieval quality is not yet good. Asked `कॉर्पोरेशन क्या है?` ("what is a corporation?"), the
system returned a passage about Givaudan Flavors Corporation's annual sales rather than the
definition — despite the correct definition existing in the corpus. The extractive scorer is
over-weighting lexical overlap on the query term and under-weighting whether the sentence actually
*answers*. Untuned weights, an 8k-passage corpus, and no reranking; this is the exact gap the
chunking lab is built to close on D2, and the lab's metrics will measure it rather than my
impression of it.

### Built

- `app/timing.py` — monotonic span recorder with offsets
- `app/schemas.py` — typed contracts for every stage
- `app/settings.py`, `app/corpus.py` — config, columnar passage store, exact search
- `app/stages/` — embed (2 lanes), lexical (Indic-aware BM25), dense (HNSW + exact), fuse (RRF),
  lang (script detection), answer (extractive Tier 1), guard (4 gates), base (error taxonomy)
- `app/pipeline.py` — orchestrator; `app/main.py` — FastAPI, warmup-gated health
- `scripts/build_corpus.py`, `scripts/embed_corpus.py` — both validated against live data
- `Dockerfile` — 3.13-slim, weights and artifacts baked at build time

---

## D1 (continued) — full corpus, and two performance defects found by measuring

### The validation split is 10x larger than projected

The first full build reported `hi: 97,941 queries, 1,901,146 passages` from a single shard. The
earlier projection of ~10k queries per language was wrong by an order of magnitude — four languages
would have produced ~4.75M passages, ~4.9 GB of float32 embeddings, a multi-gigabyte HNSW graph,
and exact search in the 50–100ms range. It would also have filled the remaining 11 GB of disk.

Killed mid-build and replaced "take the first N rows" with **hash-based subsampling on
`query_id`**. Two properties matter and neither comes free with truncation:

- *Unbiased* — parquet row order may correlate with id, source, or collection date.
- *Identical across languages* — the hash depends only on `query_id`, which is stable across
  shards, so every language contributes translations of the same queries.

Result at 1-in-16: **310,582 passages**, five languages, ~62k each. The **37.9% dedupe rate** is
the proof the second property held — the English passages repeated in all four shards collapsed to
one copy, so each additional language added ~62k rather than ~124k.

### Defect 1: model2vec's multiprocessing is a 35x pessimisation here

Embedding 310k passages did not finish in 13 minutes. Diagnosis: `StaticModel.encode` defaults to
`use_multiprocessing=True` above 10,000 inputs. On macOS the start method is spawn, so each of the
8–10 workers re-imported the module and materialised its own slice. On an 8 GB machine this drove
**swap to 8.8 GB of 10.2 GB**, load average to 28, and left workers at ~16% CPU thrashing on
page-ins.

Single-process, the same job takes **22.2 seconds** (14,022 passages/s). This is a
memory-bandwidth-bound lookup — more processes only multiply the working set. Fixed by passing
`use_multiprocessing=False` and encoding in logged slices, so a long run reports progress instead
of being indistinguishable from a hang.

### Defect 2: memory-mapping the embedding matrix wrecked the tail

First full-corpus measurement, 300 queries after 20 warmup:

| mode | P50 | P70 | P100 |
|---|---|---|---|
| hnsw | 5.03 | 6.45 | **138.4** |
| exact | 8.88 | 9.20 | **1736.1** |

P50 was excellent and P100 was catastrophic — a 260x spread in exact mode. Every outlier was a
**first-touch page fault** against the memory-mapped 318 MB embedding file: `dense` P50 6.6ms
against P100 1,728ms. A 20-query warmup cannot fault in a matrix that size, and page-cache pages
can be evicted afterwards, so the outliers would have kept returning under memory pressure.

Fixed by loading the embedding matrix and the HNSW graph into RAM rather than mapping them
(`np.load` without `mmap_mode`, `usearch` `load()` rather than `view()`). 318 MB resident is
affordable; a 1.7-second P100 is not.

| mode | P50 | P70 | P100 | tail improvement |
|---|---|---|---|---|
| hnsw | 4.94 | 5.83 | **15.29** | 9x |
| exact | 9.63 | 10.26 | **27.24** | 64x |

Both modes now sit far inside the 200ms P100 target, and the distribution is flat rather than
long-tailed. This is the single most valuable measurement taken so far: the naive reading of the
first table would have been "we meet P50, tighten the tail later", when in fact the tail was a
one-line configuration defect, not a tuning problem.

### External providers — verified, not assumed

| provider | result |
|---|---|
| Groq `llama-3.3-70b-versatile` | works, TTFT **477ms** |
| Groq `openai/gpt-oss-20b` | works, TTFT **562ms** |
| Groq `llama-3.1-8b-instant` | works, TTFT 1573ms (cold) |
| Cerebras | **HTTP 402** — free quota unavailable on this account |
| Sarvam Saaras v3 | handshake + auth confirmed live |
| HF Docker Space | **HTTP 402** — Docker Spaces now require PRO; only Static Spaces are free |

Two things fell out of this:

- `llama-3.1-8b` no longer exists on Cerebras at all (their catalogue is now `zai-glm-4.7`,
  `gemma-4-31b`, `gpt-oss-120b`), so the originally specified default model was unavailable
  regardless of billing.
- Groq sits behind Cloudflare, which rejects `urllib`'s default user-agent with a **403 error 1010**
  that reads exactly like an auth failure. `httpx` passes. An explicit User-Agent is now set so this
  cannot silently regress.

**Every measured TTFT is 2.5–8x the entire 200ms budget.** That is the empirical justification for
the two-tier design, and it is worth stating as a finding rather than a design preference:
retrieve-then-generate cannot meet this target by arithmetic, no matter how fast retrieval is.

### Deployment target changed

HF Docker Spaces are no longer free. Moved to **Google Cloud Run** — real vCPUs, US region, free at
this traffic volume, and it runs the same Dockerfile unmodified.

### Bugs found and fixed during validation

- `split_sentences` folded a trailing fragment with `merged[-2] = f"... {merged.pop()}"`. The pop
  shortens the list before the assignment index resolves, so at length 2 it raised `IndexError`.
  Caught by running 200 real queries rather than by reading the code.
- `Corpus.exact_search` originally restricted by language via `emb[rows]`, which materialises a
  fresh copy of the sliced embedding matrix — hundreds of megabytes per query at full scale.
  Replaced with a boolean mask applied to the score vector.
- `Pipeline.ask` embedded the query a second time for the extractive stage instead of reusing the
  vector already computed in `retrieve`.
