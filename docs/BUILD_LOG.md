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

### Built

_(in progress — updated as commits land)_
