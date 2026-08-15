# SHRUTI — container image for Google Cloud Run.
#
# Originally targeted a Hugging Face Docker Space. That target died on contact with reality:
# HF now returns 402 for Docker Spaces on cpu-basic ("requires a PRO subscription"), so only
# Static Spaces remain free. Cloud Run replaces it — real vCPUs, US region, free at this traffic
# volume, and it runs this same Dockerfile unmodified.
#
# Two properties this image is built for:
#
#   1. Everything expensive happens at BUILD time. Model weights and corpus artifacts are baked
#      into the image, so container start is a memory-map and a warmup rather than a download.
#      A cold start that pulls 500 MB from the Hub would put a minute of latency in front of
#      whoever wakes the Space.
#   2. Layers are ordered by change frequency. Dependencies change rarely, artifacts occasionally,
#      application code constantly — so a code edit rebuilds one small layer.
#
# Pinned to 3.13 to match the local development interpreter exactly. "Works on my machine" is not
# a debugging strategy two days before a deadline.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Spaces run the container as uid 1000; HF libraries need a writable home for their caches.
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    SHRUTI_ARTIFACT_DIR=/app/artifacts

RUN useradd -m -u 1000 user

# --- system deps ------------------------------------------------------------------------
# curl is used by the container healthcheck; build-essential is not installed because every
# dependency ships a manylinux wheel for cp313 (verified — see docs/BUILD_LOG.md).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- python deps ------------------------------------------------------------------------
# Copied alone, before any source, so dependency layers are cached across code changes.
COPY --chown=user pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install .

# --- model weights ----------------------------------------------------------------------
# Baked in at build time with an explicit allowlist. The repo also ships an ONNX export of the
# same weights (~320 MB) which the fast lane never touches; `allow_patterns` keeps it out rather
# than downloading it and letting it sit in the image.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('minishlab/potion-multilingual-128M', \
    allow_patterns=['*.json','*.safetensors','*.txt'])" \
    && chown -R user:user /home/user/.cache

# --- corpus artifacts -------------------------------------------------------------------
# Produced offline by scripts/build_corpus.py + scripts/embed_corpus.py and committed to a
# separate HF dataset repo — never to git, where half a gigabyte of embeddings does not belong.
# Build with:  --build-arg ARTIFACT_REPO=<user>/shruti-artifacts
# Defaulted rather than required. `gcloud run deploy --source` builds through Cloud Build, where
# passing a Docker ARG means threading substitutions through a build config — avoidable complexity
# for a value that only ever has one setting. The repo is public, so no token is needed either.
ARG ARTIFACT_REPO="Het0812/shruti-artifacts"
ARG HF_TOKEN=""
ENV R=${ARTIFACT_REPO} T=${HF_TOKEN}

# Only the float32 source artifacts are fetched. The int8 variants on the same repo exist for the
# memory-constrained Render experiment and would otherwise be pulled for nothing.
RUN if [ -n "$ARTIFACT_REPO" ]; then \
        python -c "\
import os; from huggingface_hub import snapshot_download; \
snapshot_download(os.environ['R'], repo_type='dataset', local_dir='/app/artifacts', \
    token=os.environ.get('T') or None, \
    allow_patterns=['passages.parquet','queries.parquet','qrels.parquet','embeddings.npy','manifest.json','bm25/*'])" ; \
    else echo 'no ARTIFACT_REPO given; artifacts must be mounted at /app/artifacts'; fi

# --- application ------------------------------------------------------------------------
COPY --chown=user app/ ./app/
COPY --chown=user web/ ./web/
COPY --chown=user bench/results/ ./bench/results/

# Build the HNSW graph at image-build time if the artifact repo did not supply one.
#
# It is ~347 MB of derived data — too slow to upload from a 3 MB/s link, and it must not be built
# at container *start* because Cloud Run counts that against the startup probe and would kill the
# revision before it ever serves. Building it here costs ~90s once per image and makes cold starts
# a pure memory-map. Without this the app silently falls back to exact search: correct, but ~7ms
# per query instead of ~1.5ms, and "silently slower" is the kind of regression nobody notices.
RUN if [ -f /app/artifacts/embeddings.npy ] && [ ! -f /app/artifacts/hnsw.usearch ]; then \
      python -c "\
import numpy as np, time; from app.stages.dense import DenseIndex; \
t=time.perf_counter(); \
DenseIndex.build(np.load('/app/artifacts/embeddings.npy')).save('/app/artifacts/hnsw.usearch'); \
print(f'HNSW built in {time.perf_counter()-t:.1f}s')" ; \
    fi

RUN mkdir -p /app/artifacts && chown -R user:user /app

USER user

# Cloud Run injects $PORT and expects the container to listen on it. Defaulted so the image also
# runs unchanged with a plain `docker run -p 8080:8080`.
ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=240s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/api/health" | grep -q '"ready":true' || exit 1

# Single worker deliberately: the embedding matrix and indexes are per-process, so a second worker
# would double resident memory for concurrency this service does not have. Cloud Run scales by
# adding containers, not by adding workers inside one.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
