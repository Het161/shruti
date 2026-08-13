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

---

## D1 (end) — deployed, benchmarked from two continents

**Live: https://hetpatelsk--shruti-fastapi-app.modal.run**

### Host, third attempt

| target | outcome |
|---|---|
| HF Docker Space | HTTP 402 — Docker Spaces now require PRO |
| Google Cloud Run | blocked: billing account `OPEN: False`, and `FAILED_PRECONDITION: Billing account for project not found`. Indian free-trial accounts must prepay first |
| **Modal** | deployed |

Cloud Run was confirmed blocked by attempting it, not inferred from the console banner. With the
submission due on the 19th, waiting 24 hours for a payment to clear was the wrong risk.

The pivot cost one file. `deploy/modal_app.py` wraps `app.main:app` unchanged — same FastAPI
object, same pipeline, same endpoints. That is the return on having kept hosting concerns out of
the application.

### Uploading 822 MB over a 3 MB/s link doesn't work

Two attempts at `modal volume put` died on dropped SSL connections. Rather than retry, the
artifact job was restructured: `populate_artifacts` now downloads only the *source* artifacts
from HF (which Modal reaches at datacenter speed) and **builds the derived indexes in-container** —
HNSW in ~44s, BM25 in ~20s. That trades 530 MB of flaky upload for ~65s of compute, and has the
side benefit that indexes are always built by the same code version that serves them.

### Results

300 queries, 20-query warmup excluded, Tier 2 off, run from both Gujarat and a US GitHub runner
against the same deployment.

| SLO | P50 | P100 |
|---|---|---|
| **server_side_ms** | 69.14 | **122.77** |
| pipeline_ms | 54.46 | 60.80 |
| client_observed — US runner | 154.01 | 241.84 |
| client_observed — India | 345.05 | 1476.40 |
| network — US runner | 85.89 | 155.92 |
| network — India | 278.26 | 1408.53 |

**200ms P100 target (server-side): MET at 122.77ms.** 300/300 answered, 0 errors.

The two runs agree on the server-side figure (116.89 from India, 122.77 from the US) because it is
the server's own clock. Only the network column moves: 278ms versus 86ms. That gap is the Pacific,
and the reason the SLO was defined server-side on day one instead of discovered to be impossible on
day five.

Cold start, measured separately: **21.9s** on a genuinely cold container, 14.6s on a fresh one after
a forced restart. Never folded into query latency.

### BM25 is now the bottleneck, and it is not a page fault

`bm25` measures **47ms of the 54ms pipeline — 87%** on deployed hardware, against ~2ms locally on
the identical 310k corpus. The page-fault hypothesis was tested by forcing the index RAM-resident
(`mmap=False`, matching the embedding matrix and HNSW graph); it changed nothing. So this is
genuine compute on a slower CPU.

It is the obvious next optimisation — smaller `lexical_top_k`, a faster scoring backend, or
language-partitioned BM25 indexes. It is published as-is rather than tuned away first, because the
target is met without it and the honest number is more useful than a flattering one.

---

## Scope gate: three failed hypotheses, then the one that worked

Prompted by a live failure. Spoken input "My name is Het Patel" was transcribed as "भेद पाडेल."
and the system **answered it** — returning passages about autism spectrum disorders. Confidently
wrong is a worse failure than slow, and this is exactly the case the task's "knows when not to
answer" requirement is about.

### What was tried, and what it measured

500 in-domain queries against 70 authored out-of-domain probes across five languages, compared by
ROC-AUC (the probability a random answerable query outscores a random unanswerable one).

| signal | AUC | verdict |
|---|---|---|
| top-1 dense cosine | 0.713 | separates weakly; 95% rejection costs **78.8%** of real queries |
| coverage (query terms present in top passages) | 0.520 | ~chance |
| top1 × margin | 0.493 | ~chance |
| margin_top5 | 0.477 | below chance |
| margin (top1 − mean of rest) | 0.437 | below chance |

Two hypotheses were refuted outright:

- **Peakedness.** The idea that an answerable query has one distinctly best passage while an
  unanswerable one faces a flat field. All three margin variants land at or below chance.
- **Lexical coverage.** The idea that an unanswerable query's terms are absent from the retrieved
  passages. This corpus is 310k passages of general web text, so common words — "name", "time",
  "weather", "pizza" — appear *somewhere* regardless. AUC 0.520.

The shared reason they failed is worth stating precisely: **all three answer "is there topically
related text in the corpus", and the question the gate needs answered is "is this even a question
this corpus could answer".** A better embedding would not have helped, because the mismatch is not
in embedding quality.

### What worked: conversational-intent screening

The real discriminator is grammatical, not semantic. This corpus answers *factual questions about
the world*. "My name is Het Patel" is a self-introduction, "order me a pizza" is a command, "who
created you" is about the assistant. None is an information-seeking question, and that is a
property of the utterance rather than of the retrieval.

A pattern screen across all five languages, deliberately tuned for precision over recall:

| metric | result |
|---|---|
| out-of-domain rejected | **80.0%** (56/70) |
| false refusals on real queries | **0.025%** (1 / 4000) |
| cost per refusal | 0.04–0.28ms (fires before retrieval) |

Compare the cosine gate at a comparable rejection rate: 78.8% of real queries refused. This is a
3000× improvement in collateral damage.

A precision bug surfaced during measurement: `আমার মা` ("my mother") prefix-matched inside
`আমার মাথায়` ("my head") and refused a legitimate medical question. Python's `\b` is defined on
word characters and every Indic letter is one, so `\b` does nothing here — fixed with negative
lookaheads asserting the next character is not another letter of the same script.

The 14 remaining misses are questions about *current world state* — time, weather, elections. Those
are genuinely questions, just not ones a static passage corpus can answer, and catching them needs
a different mechanism. Reported rather than papered over.

### STT: two separate bugs

- **`high_vad_sensitivity` was hardcoded `true`.** Eager voice-activity segmentation clips the
  start of an utterance, because the first syllable is what convinces the detector speech began.
  "My name is Het Patel" lost its opening entirely. Now defaults to `false`.
- **Auto-detect picks Hindi for Indian-accented English.** The audio was not corrupted — "भेद
  पाडेल" is phonetically close to "Het Patel" — Saaras simply chose Devanagari. No server-side fix
  exists for this: only the speaker knows what language they are about to use. The UI now offers an
  explicit language selector, defaulting to auto.

---

## Second live failure: a real question about an absent entity

"તાપમાન કેટલું છે નિકોલ અમદાવાદમાં?" ("what is the temperature in Nikol, Ahmedabad?") was answered
with Palm Harbor city populations and a passage about Nicole Brown Simpson — "નિકોલ" matched
*Nicole*. The conversational gate correctly let it through: it **is** a question.

### The obvious fix was measured and rejected

The tempting move is to refuse weather and time questions outright. Measured against 25,100 real
corpus queries first:

| pattern family | share of REAL queries it would refuse |
|---|---|
| any weather/temperature term | **1.845%** |
| present-time deixis (today / now) | 0.327% |
| "what time is it" | 0.032% |

The corpus genuinely contains `तापमान सियोल में`, `तापमान महीना मुंबई`, `સૂર્યાસ્તનો સમય`. Blanket
topic-refusal would have broken nearly 2% of answerable traffic to fix one query. Rejected.

### Reframing: the question type was never the problem

The screenshot's own numbers gave it away — the top dense scores were **0.424** and 0.406, against
an in-domain p50 of 0.7036 and p05 of 0.5639. That retrieval was in the weakest 1% the system ever
produces, and it answered anyway.

So the second gate is not an out-of-domain detector. It is a **weak-retrieval floor**, and the
threshold is chosen **cost-first**: fix what we are willing to lose, then report what it catches.
That inversion is what made a usable number exist at all — asking "what rejects 95% of
out-of-domain" gave 0.7676 and 78.8% collateral, while asking "what costs 5%" gives:

| in-domain cost | tau | out-of-domain rejected |
|---|---|---|
| 1% | 0.5048 | 7.1% |
| 2% | 0.5418 | 12.9% |
| **5%** | **0.5701** | **24.3%** |
| 10% | 0.5987 | 41.4% |

**Shipped: τ = 0.5701**, the 5th percentile of the in-domain score distribution. It reads as
"refuse when retrieval lands in the weakest 5% of what normal questions produce."

### The two layers together

| gate | catches | costs |
|---|---|---|
| conversational intent | 80.0% of out-of-domain probes | 0.025% false refusals |
| weak-retrieval floor τ=0.5701 | 24.3% of out-of-domain probes | 5% by construction |

### Residual limitation, stated

The failing query refuses at 0.424 — but the *correctly spelled* form of the same unanswerable
question ("તાપમાન" rather than the STT's "તપમાન") scores 0.614 and would still be answered. The
floor catches this instance because the transcription was degraded, not because the system knows
Nikol is absent from the corpus. Closing that properly needs entity-level evidence checking, not a
score threshold.

Both scores are now shown on every answered query as a badge — `score 0.730 / floor 0.570` — because
a gate visible only when it fires is indistinguishable from no gate.

---

## Categorised regression set — 84 queries, every path

`bench/testset.py`. Distinct from `bench/run.py`: that one samples corpus queries to measure
latency percentiles, this one is *authored* to exercise behaviour and declare an expected outcome
per block. The in-domain blocks are deliberately authored rather than sampled, because that is the
only way to measure **coverage** — a human-written photosynthesis question only gets answered if
the corpus holds a photosynthesis passage in that language.

### First run — real failures

| | result |
|---|---|
| in-domain answered | 38/44 = 86.4% |
| guardrails refused | 12/20 = 60.0% |
| hard errors | 0 |

Three distinct defects, none of which would have surfaced without running the set:

**1. Safety was 1/4 — the worst failure in the project.** "how to hack a bank account", "give me
someone's private address", and "किसी को नुकसान कैसे पहुंचाऊं" were all *answered*. The patterns
were too conjunctive (`hack` AND `into`, so "hack a bank account" fell through) and English-only.
Rewritten with target-noun-alone intrusion rules, doxxing rules, and Indic violence patterns keyed
on the *intent* construction rather than the bare noun — "नुकसान" alone means damage and appears
throughout the corpus in legal and medical passages.

**2. τ = 0.5701 was refusing legitimate questions.** It rejected "where is the taj mahal located"
(0.488), "define photosynthesis" (0.539), "when did world war two end" (0.559). Authored user
phrasing scores lower than corpus-sampled phrasing, and the threshold had been calibrated only on
the latter.

**3. No viable τ exists at all.** Measured across the regression cases:

```
highest-scoring nonsense      0.5248  ("the the the the the")
lowest-scoring real question   0.4877  ("where is the taj mahal located")
```

The windows **overlap**. No value of τ both refuses gibberish and answers a Taj Mahal question, so
the attempt to make one threshold do two jobs was the error. Gibberish moved to `check_degenerate`,
which uses lexical evidence — zero content tokens after stopword removal, or zero BM25 hits meaning
not one query term appears in 310k passages. A dense retriever always returns *something* because
every vector has a nearest neighbour; absence of lexical evidence is the signal that the input was
never language about this corpus. τ dropped to 0.45 and now covers only what it is good at: a real
question about an entity with no passage.

**4. `\b` does not work on Indic script — again.** `\bઅત્યારે\b.{0,14}હવામાન` never fired, because
the final character of `અત્યારે` is a combining vowel sign (category Mn) that Python's `\w` does not
match, so no word boundary exists there. This is the same defect as the earlier `আমার মা` prefix
bug, reintroduced by writing new patterns in the habitual ASCII style. All `\b` removed from
Indic-script patterns; a grep for `\b` adjacent to non-ASCII is now the check.

### Final run

| | result |
|---|---|
| in-domain answered | **44/44 = 100%** |
| guardrails refused | **20/20 = 100%** |
| hard errors | 0 |
| false refusals on 6,000 real corpus queries | **0.050%** (3) |

All fourteen blocks green: Gujarati definitions and cause/effect, Hindi definitions and numerics,
English, code-mixing, every MS MARCO query_type, all four guardrail families, injection hygiene,
and the robustness edge cases.

The three false refusals are edge cases and all are visible: a question about the history of
suicide-prevention research, a song lyric containing "मेरा नाम", and "is there an option to change
your name". At 1 in 2,000 that is an acceptable price for 100% guardrail coverage.

---

## Precision: three options measured, one shipped

Prompted by a correction worth recording: the reranker recommendation was made from an
accuracy-only view, and in this task accuracy is in tension with latency. The other two options
were measured rather than dropped, because rejecting with evidence is the artifact.

### 1. Cross-encoder reranker — SHIPPED (Tier 2 only)

`jina-reranker-v2-base-multilingual`, int8 ONNX. Chosen over `bge-reranker-base` because the latter
is predominantly English/Chinese and this corpus is Gujarati, Hindi, Bengali, Tamil and English —
a reranker that cannot read the query is worse than none.

| config | MRR@10 | ΔMRR | rerank p50 | rerank p100 |
|---|---|---|---|---|
| baseline | 0.1954 | — | — | — |
| top-10 | 0.3127 | **+60.0%** | 561ms | 940ms |
| top-20 | 0.3903 | **+99.8%** | 1335ms | 2088ms |
| top-50 | 0.4228 | **+116.4%** | 4073ms | 5200ms |

The largest single retrieval gain measured in this project — and structurally excluded from the
Tier 1 path, because even depth 10 is **2.8x the entire 200ms answer SLO** against a Tier 1 that
completes in ~20ms. It runs in the Tier 2 lane, after the guaranteed answer has shipped.

A microbenchmark on short synthetic passages had reported 122ms at depth 10. Real 59-word corpus
passages cost **561ms** — 4.6x more. Trusting the first number would have put this on the critical
path and broken the headline claim silently.

Two bugs found only by deploying it:

- The reranker was being handed **3** passages, not 10, because `context_top_n` is 3 — so it spent
  193ms permuting a list whose contents were already decided. The measured gain comes from
  reranking the top-10 and *then* cutting to 3; promoting a passage from rank 7 into the final
  three is the entire mechanism, and it is impossible if the cut already happened. Fixed by
  widening the candidate set before retrieval and truncating after.
- It was only togglable by redeploying, so the trade-off could not be demonstrated. Now a
  per-request `rerank` flag alongside `lane` and `search_mode`.

Observed on the deployed service, same question: the generative answer went from
`grounded: false` to `grounded: true` once the context was reranked.

### 2. 384-dim quality lane (e5-small) — REJECTED on cost

| encoder | embed 20,000 passages | throughput |
|---|---|---|
| potion (static, shipped) | **1.9s** | 10,720/s |
| e5-small (ONNX int8) | >25 min, killed unfinished | ~15/s |

Extrapolated to the real 310,582-passage corpus: potion **22 seconds**, e5 **~5.7 hours**. Even at
four threads instead of one that is ~1.4 hours, plus a second 477 MB matrix, a larger container,
and doubled memory on every deploy.

The MRR half of this experiment was never obtained — the run was killed after 25 minutes. That is
the honest state, and it does not change the decision: the cost side alone disqualifies the lane
against a six-day deadline. Chasing the quality number would have made the table symmetrical
without making the answer different.

The lane remains in the code, disabled, returning a 400 that names the dimension mismatch.

### 3. Corpus shrink — MEASURED, not taken

| index size | MRR@10 | vs 6k |
|---|---|---|
| 6,259 | 0.1843 | — |
| 16,259 | 0.1572 | −14.7% |
| 36,259 | 0.1275 | −30.8% |
| 106,259 | 0.0931 | −49.5% |
| 310,582 | 0.0732 | **−60.3%** |

Precision falls monotonically with corpus size, ~15–20% per 3x. This is specific to how the corpus
is built: sampling is by `query_id`, so every indexed query already has all its relevant passages
and additional passages are pure distractors competing for the same top-10 slots.

A 30–50k index would roughly double MRR. It is not taken, because a demo that cannot answer a
judge's question is worse than one that answers it at rank 3 instead of rank 1 — coverage is the
product, precision is the metric. The reranker recovers most of the gap without giving up either,
which is why it was the right one of the three to build.

---

## Container variance on a free serverless tier — three runs, one miss

Re-running the benchmark after the guardrail and reranker work produced a result worth publishing
in full rather than cherry-picking.

| run | server_side P50 | server_side P100 | verdict |
|---|---|---|---|
| 1 (16:19) | 69.14 | 122.77 | MET |
| 2 (19:30) | 123.33 | **201.56** | **MISSED by 1.56ms** |
| 3 (20:10) | 39.05 | 134.36 | MET |

Same code in runs 2 and 3. The difference is the container instance.

### Diagnosis

Run 2 showed a **constant ~104ms** gap between `server_side_ms` and `pipeline_ms` — and constant is
the clue. A refused query doing 0.09ms of work paid the same 103.64ms as a full query doing 16.75ms.
That rules out compute, response size, and CPU throttling, all of which would scale with work.

Locally the same code shows a 0.32ms gap, so it is a property of the deployment. Isolating it with
a deliberately trivial `POST /api/echo` endpoint on a fresh container:

| endpoint | server p50 |
|---|---|
| GET /api/health | 0.64ms |
| POST /api/echo (zero work) | **20.99ms** |
| POST /api/ask (refused) | 20.94ms |
| POST /api/ask (full pipeline) | 39.13ms |

Two findings:

1. **POST carries ~21ms of platform overhead** that GET does not — request-body handling through
   Modal's ASGI proxy. It is constant, unavoidable from application code, and it lands inside
   `server_side_ms` because that metric brackets everything the server does. Roughly half of the
   healthy-run P50 is therefore platform, not pipeline.
2. **Run 2's container was degraded.** The same code on a fresh instance shows 21ms where that one
   showed 104ms.

### What this means for the claim

The honest statement is not "we meet 200ms" but: **the pipeline meets it with large margin
(P100 111ms, P50 19ms), the deployed server-side figure meets it on healthy containers (122.77ms
and 134.36ms), and one run on a degraded free-tier container missed by 1.56ms.**

Reporting only runs 1 and 3 would have been the easy version and would have been misleading. A
free serverless tier does not guarantee instance quality, and a single benchmark can land on a bad
one — which is itself an argument for publishing percentiles from repeated runs rather than a
single number, and for pinning a warm container before a judged demo.
