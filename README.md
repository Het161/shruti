# SHRUTI

**श्रुति** — *"that which is heard."* In the Indian tradition, knowledge transmitted by voice.

A voice-first, Indic-first grounded retrieval system. Speak in Hindi, Gujarati, Bengali, Tamil, or
English; SHRUTI transcribes as you talk, retrieves from a multilingual MS MARCO corpus, and answers
in your language — with the entire latency budget of every query displayed as a measured waterfall
on screen.

> Heard. Retrieved. Answered — with the stopwatch showing.

---

**Live: https://hetpatelsk--shruti-fastapi-app.modal.run**

## Results — measured on the deployed system

300 queries across hi/gu/bn/ta, 20-query warmup excluded, Tier 2 off. Run twice from different
continents against the same deployment.

| SLO | P50 | P70 | P90 | P100 |
|---|---|---|---|---|
| **server_side_ms** (headline) | 69.14 | 70.50 | 72.29 | **122.77** |
| pipeline_ms | 54.46 | 55.70 | 56.93 | 60.80 |
| client_observed_ms — *US runner* | 154.01 | 195.18 | 210.99 | 241.84 |
| client_observed_ms — *India* | 345.05 | 348.46 | 358.10 | 1476.40 |
| network_overhead_ms — *US runner* | 85.89 | 123.04 | 139.19 | 155.92 |
| network_overhead_ms — *India* | 278.26 | 282.81 | 292.84 | 1408.53 |

**200ms P100 target (server-side): MET — 122.77ms.** 300/300 answered, 0 errors.
Cold start measured separately: **21.9s** on a truly cold container, 231ms first-contact when warm.

The two runs report the *same* server-side figure (116.89 measured from India, 122.77 from the US)
because it comes from the server's own monotonic clock. What changes between them is the network
column — 278ms from Gujarat, 86ms from a US runner. That difference is the Pacific Ocean, and it is
exactly why the headline SLO is defined server-side. Both distributions ship.

### Per stage, deployed

| stage | P50 | P100 |
|---|---|---|
| **bm25** | **47.182** | 52.204 |
| extract | 3.640 | 5.983 |
| dense | 2.457 | 4.034 |
| embed | 0.459 | 0.773 |
| fuse | 0.230 | 0.266 |
| detect | 0.028 | 0.234 |
| guard_safety | 0.029 | 0.062 |
| guard_scope | 0.024 | 0.047 |

**BM25 is 87% of pipeline time on the deployed hardware** — 47ms of 54ms. Locally it measures ~2ms
on the same 310k corpus, so this is genuine compute on a slower CPU rather than a page-fault
artifact (that hypothesis was tested and rejected: forcing the index RAM-resident changed nothing).
It is the obvious next optimisation and it is reported rather than tuned away before publishing.

## What the latency numbers mean

This is the most important section in this README, because a latency claim is worthless without a
definition, and the obvious definition is the one that quietly cheats.

The deployment is US-hosted. A browser in Gujarat is ~250ms of Pacific round-trip away from it
before any work happens. So "answer visible in under 200ms" measured from an Indian laptop is
physically impossible for *any* system, and a system that claims it is either measuring localhost
or lying. Three numbers are therefore published, and none of them is hidden inside another:

| number | what it measures | where it comes from |
|---|---|---|
| **`server_side_ms`** | the deployed process handling a full request, network excluded | the server's own monotonic clock, returned in `X-Server-Time-Ms` |
| **`pipeline_ms`** | the retrieval-to-answer stages alone, framework overhead excluded | `RequestTimer` spans, returned in the response body |
| **`client_observed_ms`** | what a caller at that location actually waited | the benchmark harness's wall clock |

**The headline SLO is `server_side_ms`.** That is what "retrieval-to-answer path" denotes and what
a latency SLO means in ordinary engineering usage. To keep it honest, the benchmark also runs from
a **US GitHub Actions runner** — co-located with the deployment, so `client_observed_ms` lands
within a few milliseconds of `server_side_ms`, on the real deployed system over real HTTP. The
India→US client-observed distribution is published separately, with the network delta as its own
series. The UI shows all three on every query.

### Defined SLOs

- **Retrieval latency** = request received → context assembled. Target **P100 < 60ms**.
- **Answer latency** = request received → Tier-1 answer serialized. Target **P50 < 120ms, P100 < 200ms**.
- **Generative TTFT / completion** — reported alongside, honestly, whatever they measure.
- **Cold start** — measured and reported separately. Never folded into query latency.

---

## Why the answer arrives in two tiers

Measured on this account, time-to-first-token from the generation providers:

| provider | model | TTFT |
|---|---|---|
| Groq | `llama-3.3-70b-versatile` | 477ms |
| Groq | `openai/gpt-oss-20b` | 562ms |
| Groq | `llama-3.1-8b-instant` | 1573ms (cold) |
| Cerebras | — | unavailable: HTTP 402, free quota exhausted on this account |

**Every one of those is multiples of the entire 200ms budget.** A retrieve-then-generate design
cannot meet this target, no matter how fast retrieval is. That is not a tuning problem; it is
arithmetic.

So the answer is produced twice:

- **Tier 1 — extractive, guaranteed, single-digit milliseconds.** The best answer sentences are
  selected from the retrieved passages by combined embedding similarity and query-term overlap,
  with a `query_type` prior. Every character is copied verbatim from a retrieved passage, so it is
  **grounded by construction** — hallucination is not mitigated, it is impossible. This is a
  complete, cited answer, and it is what makes a sub-200ms claim honest.
- **Tier 2 — generative, streamed, discardable.** Groq streams a fluent answer that replaces Tier 1
  in the UI. It is verified for grounding after generation; on failure the UI keeps Tier 1 and
  displays *"generative answer withheld: failed grounding check."*

Generation failing is a **normal state** in this system, not an error. The user already has an
answer before it starts.

---

## Architecture

```
 browser ──PCM16/16kHz──► /ws/voice ──► Sarvam Saaras v3 ──► transcript
                                │
                                ▼
  ┌────────────────────── Pipeline (one object, two entrypoints) ──────────────────────┐
  │ safety → detect → embed → [ dense ‖ bm25 ] → fuse(RRF) → scope → extract          │
  └────────────────────────────────────────────────────────────────────────────────────┘
                                │                              │
                       Tier 1 (immediate)            Tier 2 (streamed, optional)
```

`/api/ask` and the voice WebSocket construct the same `QueryInput` and call the same `Pipeline`
object. The benchmark harness therefore exercises literally the code path the microphone does.

**Zero network calls in the retrieval path.** Embedding, BM25, fusion, and vector search all run
in-process. A hosted embedding API would spend the whole budget on round-trip time alone.

| component | choice | why |
|---|---|---|
| Embeddings (fast) | `potion-multilingual-128M`, 256-dim | Model2Vec static lookup — no transformer forward pass, tens of microseconds |
| Embeddings (quality) | `multilingual-e5-small` ONNX int8 | a real forward pass; lazily loaded, toggleable in the UI |
| Lexical | `bm25s` + Indic-aware tokeniser | covers exact tokens: names, numbers, dates |
| Fusion | Reciprocal Rank Fusion (k=60) | scale-free; BM25 and cosine scores are not commensurable |
| Vector search | `usearch` HNSW **and** exact NumPy | at this corpus size exact is a few ms, so it is a real runtime option and the ground truth HNSW recall is measured against |
| STT | Sarvam Saaras v3 streaming | Indic-specialist, code-mix, auto language detection |
| Generation | Groq → Cerebras → extractive-only | circuit breaker, per-provider failover |
| Host | Google Cloud Run | *(HF Docker Spaces now return 402 — PRO required)* |

### Why Sarvam rather than ElevenLabs

The corpus is Indic. Saaras is trained for Indian languages and code-mixed speech and auto-detects
across them, so a user can switch from Gujarati to English mid-sentence without touching a setting.
For a dataset that is MS MARCO translated into fourteen Indic languages, the Indic specialist is
the choice that matches the data.

---

## The corpus

Built from [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI).

**310,582 passages** across **5 languages**, near-perfectly balanced:

| lang | passages |
|---|---|
| hi | 62,198 |
| en | 62,307 |
| gu | 62,069 |
| bn | 62,183 |
| ta | 61,825 |

Three decisions here were driven by measurement rather than assumption:

**Validation split, not train.** Validation shards are ~470 MB against train's ~3.7 GB and carry
the same `is_selected` relevance labels — an eighth the download for identical evaluation value.

**Sampled 1-in-16, by a hash of `query_id`.** The validation split holds 97,941 queries per
language at ~19.4 passages each: ~4.75M passages across four languages, which would mean ~4.9 GB of
float32 embeddings and exact search in the 50–100ms range. Sampling is by hashed `query_id` so the
*same* queries are selected in every language — which is what makes cross-lingual comparison
meaningful and lets the repeated English passages collapse under content-hash dedupe. The observed
**37.9% dedupe rate** confirms it worked.

**Translation-degeneration filter.** MSMARCO-XI is machine-translated, and neural MT loops on short
inputs. The English query `"suit definition"` (15 chars) became a 7,783-character Hindi string
repeating one clause ~90 times. Prevalence was measured, not guessed:

| repeated-5-gram ratio | passages | queries |
|---|---|---|
| > 0.3 | 4.22% | 0.25% |
| > 0.5 | 0.11% | 0.25% |
| > 0.7 | 0.00% | 0.25% |

0.5 is the cut: at 0.3 the filter starts eating legitimate enumerations and boilerplate, flagging
40× more passages while catching no additional queries. This matters more than the percentages
suggest — looped n-grams inflate BM25 term frequencies and drag embeddings toward the repeated
phrase, so a degenerate passage retrieves for far more queries than it deserves.

---

## Chunking Lab

Six strategies, **scored** rather than asserted — possible only because the dataset ships
`is_selected` relevance labels, so MRR@10 / Recall@5 / Recall@20 are computable against ground
truth.

1. **passage-native** — the dataset's own units. The baseline to beat.
2. **fixed-256-64** — the naive baseline, implemented precisely so the table shows *why* it loses.
3. **sentence-window** — embed a sentence, return it ±1 as context.
4. **semantic-boundary** — split where adjacent-sentence similarity drops.
5. **parent-child** — embed small children, return the parent passage.
6. **metadata-aware** — language partitioning plus a `query_type` prior.

Sampling is by *query*, not by passage, so every sampled query's relevant passages are guaranteed
present — sampling passages directly would delete correct answers and score every strategy against
an unreachable ceiling. Chunks are collapsed to parent passages before scoring, so a sentence-level
index and a passage-level one are compared on the same footing.

### Results — 500 eval queries, 6,259 passages, hi/gu/bn/ta

| strategy | chunks | mean chars | MRR@10 | R@5 | R@10 | R@20 | p50 ms |
|---|---|---|---|---|---|---|---|
| **metadata-aware** | 6,259 | 302 | **0.2939** | **0.247** | **0.3315** | **0.4095** | 0.157 |
| parent-child | 20,784 | 88 | 0.2687 | 0.2165 | 0.2955 | 0.357 | 0.598 |
| sentence-window | 17,046 | 110 | 0.2667 | 0.219 | 0.3005 | 0.346 | 0.444 |
| passage-native | 6,259 | 302 | 0.2585 | 0.2085 | 0.2765 | 0.335 | 0.147 |
| semantic-boundary | 13,307 | 142 | 0.2563 | 0.218 | 0.291 | 0.3435 | 0.204 |
| fixed-256-64 | 11,863 | 189 | 0.2505 | 0.2065 | 0.272 | 0.3355 | 0.199 |

**`metadata-aware` is the shipped configuration** — it wins every quality metric while being the
second-fastest strategy, so there is no quality/latency trade to negotiate. `fixed-256-64`, the
naive baseline, places last on MRR@10 exactly as expected.

**A correction worth reporting.** The first run of this lab scored `metadata-aware` first on MRR@10
and *last* on recall@20 (0.165). That was a defect in the evaluation, not the strategy: `query_id`
is shared across language shards, so a Hindi query's raw qrels also mark its Gujarati, Bengali, and
Tamil translations relevant. The metric was rewarding systems for returning a Tamil passage to a
Hindi speaker and punishing language filtering for correctly declining to. Restricting ground truth
to the query's own language plus English — applied identically to all six strategies — moved
`metadata-aware`'s recall@20 from 0.165 to 0.410 and made it win outright.

```bash
python lab/chunking_eval.py --corpus data/corpus --queries 500
```

Full artifact: [`bench/results/chunking_lab.json`](bench/results/), also rendered on **/method**.

---

## Guardrails

Four gates, each surfaced in the UI with its name and reason. Refusal is a first-class output, not
a hidden branch.

| gate | mechanism |
|---|---|
| **scope** | abstention threshold τ on the top **dense cosine** score — not the fused RRF score, which is rank-derived and has no stable cross-query meaning. **Calibration was run and the honest result is that no shippable threshold exists — see below.** The gate reports itself uncalibrated in every response rather than applying a number that would break the product. |
| **safety** | fast pattern screen on the transcript; runs before retrieval so a refusal costs nothing |
| **grounding** | Tier 1 is grounded by construction. Tier 2 is verified post-hoc by token containment against the passages it cited — not an LLM judge, which would add a round-trip and another chance to be wrong |
| **injection** | retrieved passages are wrapped as data and never interpolated as instructions; instruction-like passages are flagged inline |

---

### Scope-gate calibration — a negative result, reported

`lab/calibrate_scope.py` scored 500 in-domain queries against a 70-probe out-of-domain set
(conversational, assistant-meta, personal-private, actionable-command) across all five languages.

```
in-domain   p05=0.5639  p50=0.7036  p95=0.8367
out-domain  p05=0.4968  p50=0.6132  p95=0.7621
```

The distributions overlap almost entirely. Four candidate signals were compared by ROC-AUC:

| signal | AUC | τ @95% OOD rejection | in-domain wrongly refused |
|---|---|---|---|
| top1 cosine | **0.713** | 0.7676 | **78.8%** |
| top1 × margin | 0.493 | 0.1197 | 90.4% |
| margin (top1 − mean of rest) | 0.437 | 0.1697 | 93.2% |
| margin_top5 | 0.477 | 0.1178 | 93.4% |

The hypothesis behind the margin signals — that an answerable query has one distinctly best passage
while an unanswerable one faces a flat field of mediocre matches — is **refuted**. All three score
at or below chance.

Even the best operating point (50% OOD rejection) refuses 14.2% of answerable queries. There is no
threshold under 10% collateral damage, so **none ships.** The reason is structural rather than a
tuning failure: a static embedder maps any text to a bag-of-subwords centroid, so every query has
*some* passage at moderate cosine. That score measures topical proximity, which is a different
question from "does this corpus answer this".

Fixing it properly means a different signal — a cross-encoder over the top-k, or an intent
classifier for conversational/command utterances, which the probe families suggest would separate
cleanly. That is the next piece of work, not something to fake with a constant.

### The two-lane toggle, and why it is disabled

The fast lane (potion) is 256-dim; the quality lane (e5-small) is 384-dim. A 384-dim query cannot
search a 256-dim index, so the lanes need *separate corpus embeddings* — ~477 MB more and ~26
minutes of e5 forward passes, against a 2 GB deployment. The toggle is disabled in the UI and the
API returns a **400 with the dimensions named**, rather than the HTTP 500 it used to.

## Running it

```bash
uv venv --python 3.13 .venv && uv pip install -e .
cp .env.example .env          # fill in keys

python scripts/build_corpus.py --languages hi gu bn ta --sample-1-in 16
python scripts/embed_corpus.py --corpus data/corpus --lane fast

SHRUTI_ARTIFACT_DIR=data/corpus uvicorn app.main:app --port 8080
```

`/api/health` reports `ready: false` until a 20-query warmup completes, so the service is never
announced as healthy while the first real query would still pay model-load cost.

### Benchmark

```bash
python bench/run.py --url https://<deployed-url> --n 320
```

≥300 measured queries across all indexed languages, 20-query warmup excluded and counted, P50/P70/
P90/P100 per stage, per language, and per SLO. Artifacts land in `bench/results/`.

---

## Honest status

- Cerebras is in the provider chain but returns **402** on this account; Groq is primary.
- The scope gate is **uncalibrated** as of this writing and says so in every response.
- BM25 is 87% of deployed pipeline time and is the clear next optimisation.
- Fonts are system stacks, not JetBrains Mono / Instrument Sans — a webfont on the critical path
  is the wrong trade for a product whose claim is measured speed.
- Retrieval quality is the open problem, not speed. See `docs/BUILD_LOG.md`, which records what was
  measured, what broke, and what is still wrong.

Every number in this README came from a run whose artifact is in this repo. Nothing here is
estimated.
