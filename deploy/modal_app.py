"""Modal deployment — the fallback host.

Why this file exists
--------------------
The deployment target has moved twice, both times because a "free tier" turned out not to be:

1. **Hugging Face Docker Space** — now returns 402; Docker Spaces require PRO.
2. **Google Cloud Run** — free at our volume, but Indian free-trial accounts must make a
   prepayment before billable APIs can be enabled, which may gate it.

Modal is the fallback: free monthly compute credits with no prepayment, a US region, real CPUs, and
Volumes sized for the ~650 MB of corpus artifacts. It also wraps an existing ASGI app directly, so
`app.main:app` deploys unchanged — the same FastAPI object, the same pipeline, the same
`/api/health` and `/api/ask`. Nothing about the system is Modal-shaped; only this file is.

Artifacts live on a Volume rather than inside the image. A 650 MB image layer rebuilds on every
code change; a Volume is written once and mounted read-only thereafter, so a one-line edit to the
answerer redeploys in seconds instead of re-uploading the corpus.

Usage
-----
    pip install modal && modal setup

    # one-time: populate the artifact volume from the HF dataset repo
    modal run deploy/modal_app.py::populate_artifacts

    # deploy
    modal deploy deploy/modal_app.py
"""

from __future__ import annotations

import os

import modal

APP_NAME = "shruti"
ARTIFACT_REPO = "Het0812/shruti-artifacts"
ARTIFACT_DIR = "/artifacts"

volume = modal.Volume.from_name("shruti-artifacts", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "fastapi>=0.115",
        "uvicorn[standard]>=0.32",
        "pydantic>=2.9",
        "pydantic-settings>=2.6",
        "orjson>=3.10",
        "model2vec>=0.4",
        "bm25s>=0.2",
        "PyStemmer>=2.2",
        "usearch>=2.16",
        "numpy>=1.26,<3",
        "pyarrow>=17",
        "huggingface-hub>=0.26",
        "httpx>=0.27",
        "websockets>=13",
    )
    # Model weights are baked into the image, not the Volume: they never change, so they belong in
    # a cached layer where they cost nothing on redeploy.
    .run_commands(
        "python -c \"from model2vec import StaticModel; "
        "StaticModel.from_pretrained('minishlab/potion-multilingual-128M')\""
    )
    .add_local_dir("app", remote_path="/root/app")
    .add_local_dir("web", remote_path="/root/web")
    .add_local_dir("bench/results", remote_path="/root/bench/results")
)

app = modal.App(APP_NAME, image=image)


@app.function(volumes={ARTIFACT_DIR: volume}, timeout=3600, memory=8192, cpu=4.0)
def populate_artifacts(rebuild_indexes: bool = True) -> dict[str, object]:
    """Fetch the corpus from HF and BUILD the derived indexes here. Run once.

    Only the *source* artifacts travel over the network — passages, embeddings, qrels, queries.
    The HNSW graph and the BM25 index are derived, and they are rebuilt inside this container
    rather than uploaded.

    That is not premature cleverness; it is a response to measurement. Uploading the full 822 MB
    artifact set from the development machine failed twice on a link that averages ~3 MB/s and
    drops SSL connections mid-transfer. The derived indexes are ~530 MB of that, and they cost 44s
    (HNSW) and 20s (BM25) of CPU to regenerate. Moving a minute of compute into the datacenter to
    avoid half a gigabyte of flaky upload is straightforwardly the better trade — and it means the
    indexes are always built by the same code version that serves them.
    """
    import time
    from pathlib import Path

    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    t0 = time.perf_counter()
    snapshot_download(
        ARTIFACT_REPO,
        repo_type="dataset",
        local_dir=ARTIFACT_DIR,
        allow_patterns=["*.parquet", "*.npy", "*.json"],
    )
    download_s = time.perf_counter() - t0
    out: dict[str, object] = {"download_s": round(download_s, 1)}

    d = Path(ARTIFACT_DIR)
    if rebuild_indexes:
        import sys

        sys.path.insert(0, "/root")
        from app.stages.dense import DenseIndex
        from app.stages.lexical import LexicalIndex

        vectors = np.load(d / "embeddings.npy")
        t0 = time.perf_counter()
        DenseIndex.build(vectors).save(d / "hnsw.usearch")
        out["hnsw_build_s"] = round(time.perf_counter() - t0, 1)

        texts = pq.read_table(d / "passages.parquet", columns=["text"]).column("text").to_pylist()
        t0 = time.perf_counter()
        LexicalIndex.build(texts).save(str(d / "bm25"))
        out["bm25_build_s"] = round(time.perf_counter() - t0, 1)

    volume.commit()
    out["files"] = sorted(p.name for p in d.iterdir())
    return out


@app.function(
    volumes={ARTIFACT_DIR: volume},
    # Sized from measurement, not guesswork — see scripts/deploy_cloudrun.sh for the itemisation.
    # The embedding matrix and HNSW graph are RAM-resident on purpose; memory-mapping them produced
    # a 1,728ms P100 from first-touch page faults.
    memory=2048,
    cpu=2.0,
    # Scales to zero by default. The arithmetic, since "keep it warm" is not free:
    #
    #   a 2-core / 2 GB container costs roughly $0.11/hour on Modal
    #   pinned 24/7           ~$79/month   — most of the free monthly credit, spent on idling
    #   pinned for 6 days     ~$16         — affordable, and the submission window IS ~6 days
    #   scale-to-zero          $0 idle     — but a judge can land on a cold start
    #
    # A keepalive ping only helps if the scaledown window outlasts the ping interval, and at that
    # point the container never scales down and costs exactly the same as pinning it. There is no
    # clever middle: you either pay to stay warm or you accept cold starts. Pretending otherwise
    # with a 30-minute cron and a 35-minute window would have been self-deception.
    #
    # Default: zero. Before the demo and the judging window, pin one deliberately:
    #     MIN_CONTAINERS=1 modal deploy deploy/modal_app.py
    #
    # Cold start is measured and published separately either way — never folded into query latency.
    min_containers=int(os.environ.get("MIN_CONTAINERS", "0")),
    scaledown_window=600,
    secrets=[modal.Secret.from_name("shruti-secrets")],
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def fastapi_app():
    """Serve the unmodified FastAPI application."""
    import os

    os.environ.setdefault("SHRUTI_ARTIFACT_DIR", ARTIFACT_DIR)
    os.environ.setdefault("SHRUTI_EMBED_LANE", "fast")
    os.environ.setdefault("SHRUTI_SEARCH_MODE", "hnsw")
    os.environ.setdefault("SHRUTI_WARMUP_QUERIES", "20")

    from app.main import app as fastapi

    return fastapi
