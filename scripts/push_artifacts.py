#!/usr/bin/env python3
"""Upload corpus artifacts to a Hugging Face dataset repo.

Artifacts do not belong in git. `embeddings.npy` alone is ~318 MB, the HNSW graph is comparable,
and git stores binaries badly — every rebuild would add another full copy to history and the clone
would become unusable within days. They live in a dataset repo instead, versioned by the Hub,
pulled at Docker build time into the image.

`manifest.json` is uploaded last, deliberately. It is what the server reads to learn the corpus
shape, so publishing it only after every binary has landed means a partially-uploaded artifact set
is never advertised as complete.

Usage
-----
    python scripts/push_artifacts.py --corpus data/corpus --repo Het0812/shruti-artifacts
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

log = logging.getLogger("push_artifacts")

# Ordered: bulk binaries first, manifest last.
ARTIFACTS = [
    "passages.parquet",
    "queries.parquet",
    "qrels.parquet",
    "embeddings.npy",
    "hnsw.usearch",
    "manifest.json",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--repo", default=os.environ.get("SHRUTI_ARTIFACT_REPO"))
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    if not args.repo:
        log.error("no --repo and SHRUTI_ARTIFACT_REPO unset")
        return 2
    if not args.token:
        log.error("no --token and HF_TOKEN unset")
        return 2

    api = HfApi(token=args.token)
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    log.info("target: https://huggingface.co/datasets/%s", args.repo)

    total = 0
    for name in ARTIFACTS:
        path = args.corpus / name
        if not path.exists():
            log.warning("missing %s, skipping", name)
            continue
        size_mb = path.stat().st_size / 1e6
        log.info("uploading %s (%.1f MB)…", name, size_mb)
        t0 = time.perf_counter()
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=args.repo,
            repo_type="dataset",
        )
        total += size_mb
        log.info("  done in %.1fs", time.perf_counter() - t0)

    # The BM25 index is a directory of several files.
    bm25 = args.corpus / "bm25"
    if bm25.is_dir():
        log.info("uploading bm25/ …")
        api.upload_folder(
            folder_path=str(bm25), path_in_repo="bm25", repo_id=args.repo, repo_type="dataset"
        )

    log.info("uploaded ~%.0f MB to %s", total, args.repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
