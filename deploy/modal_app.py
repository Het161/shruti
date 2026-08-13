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


@app.function(volumes={ARTIFACT_DIR: volume}, timeout=3600)
def populate_artifacts() -> list[str]:
    """Download corpus artifacts from the HF dataset repo onto the Volume. Run once."""
    from pathlib import Path

    from huggingface_hub import snapshot_download

    snapshot_download(
        ARTIFACT_REPO,
        repo_type="dataset",
        local_dir=ARTIFACT_DIR,
        allow_patterns=["*.parquet", "*.npy", "*.usearch", "*.json", "bm25/*"],
    )
    volume.commit()
    return sorted(p.name for p in Path(ARTIFACT_DIR).iterdir())


@app.function(
    volumes={ARTIFACT_DIR: volume},
    # Sized from measurement, not guesswork — see scripts/deploy_cloudrun.sh for the itemisation.
    # The embedding matrix and HNSW graph are RAM-resident on purpose; memory-mapping them produced
    # a 1,728ms P100 from first-touch page faults.
    memory=2048,
    cpu=2.0,
    # Keeps one container warm. Cold start means container boot plus a ~318 MB load plus a
    # 20-query warmup, and a judge opening the link should not be the one to pay it. Cold start is
    # still measured and reported separately rather than hidden.
    min_containers=1,
    scaledown_window=900,
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
