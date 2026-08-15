# SHRUTI on Render's free tier — 512 MB RAM, 0.1 CPU, no credit card.
#
# This is the fourth deployment target. The previous three each died on a paywall:
# HF Docker Spaces now require PRO, Cloud Run requires a prepayment on Indian free-trial accounts,
# and Modal's spend limit was exhausted. Render's free web service needs no card at all, which is
# the binding constraint here.
#
# Two hard limits shape everything below.
#
# **512 MB RAM.** The x86 deployment measured 555 MB resident with the full 310k corpus — 43 MB
# over. Two changes fix it with room to spare:
#   - int8 embeddings: 318 MB -> 80 MB. Quantisation error is 0.001244 mean absolute on vectors
#     that are L2-normalised, so every component already lay in [-1, 1] and a single global scale
#     loses almost nothing.
#   - no HNSW graph: the usearch index is 347 MB and would blow the budget on its own. Exact
#     search replaces it, which at this corpus size is a single chunked matmul and is also the
#     ground truth the approximate index was always measured against.
# Projected resident: ~316 MB.
#
# **0.1 CPU.** Nothing here can fix that. The x86 pipeline ran 18.89 ms P50; this will be
# materially slower and the deployed numbers must be re-measured and re-published rather than
# carried over. See deploy/render/README.md.
#
# Nothing is built at image-build time. Render's free builder is as slow as its runtime, and
# constructing the BM25 index over 310k passages there would time out — so every artifact,
# including the derived indexes, is downloaded pre-built.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    SHRUTI_ARTIFACT_DIR=/app/artifacts \
    SHRUTI_SEARCH_MODE=exact \
    SHRUTI_EMBED_LANE=fast \
    SHRUTI_WARMUP_QUERIES=10

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
# onnxruntime and tokenizers are omitted deliberately: they are ~200 MB and only the quality lane
# and the reranker use them. Both are off on this tier — there is no memory budget for either.
RUN pip install --upgrade pip && pip install \
    "fastapi>=0.115" "uvicorn[standard]>=0.32" "pydantic>=2.9" "pydantic-settings>=2.6" \
    "orjson>=3.10" "model2vec>=0.4" "bm25s>=0.2" "PyStemmer>=2.2" "numpy>=1.26,<3" \
    "pyarrow>=17" "huggingface-hub>=0.26" "httpx>=0.27" "websockets>=13"

# Model weights baked in, so a cold start is a disk read rather than a 1 GB download. Render free
# services sleep after 15 minutes, so cold starts are frequent and worth optimising.
RUN python -c "\
from model2vec import StaticModel; \
StaticModel.from_pretrained('minishlab/potion-multilingual-128M')"

# Artifacts baked in for the same reason. int8 embeddings and the prebuilt BM25 index only —
# hnsw.usearch is deliberately excluded.
ARG ARTIFACT_REPO=Het0812/shruti-artifacts
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${ARTIFACT_REPO}', repo_type='dataset', local_dir='/app/artifacts', \
  allow_patterns=['passages.parquet','queries.parquet','qrels.parquet', \
                  'embeddings_int8.npy','manifest_int8.json','bm25/*'])" \
 && cd /app/artifacts \
 && mv embeddings_int8.npy embeddings.npy \
 && mv manifest_int8.json manifest.json \
 && python -c "\
import json,pathlib; p=pathlib.Path('/app/artifacts/manifest.json'); m=json.loads(p.read_text()); \
m['embeddings']=m.pop('embeddings_int8'); p.write_text(json.dumps(m))" \
 && du -sh /app/artifacts

COPY app/ ./app/
COPY web/ ./web/
COPY bench/results/ ./bench/results/

EXPOSE 10000
# Render injects $PORT. Single worker: the corpus is per-process and a second worker would double
# resident memory on a tier that has none to spare.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1
