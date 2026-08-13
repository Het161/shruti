#!/usr/bin/env python3
"""Embed the corpus offline and build every index the server loads at startup.

Everything expensive happens here so that nothing expensive happens per request. The server's job
at startup is to memory-map what this script produced; its job during a request is arithmetic over
resident memory.

Outputs, all written to the artifact directory:

    embeddings.npy    (N, dim) float32 or int8, row-aligned with passages.parquet
    hnsw.usearch      approximate index over those vectors
    bm25/             lexical index
    manifest.json     updated in place with embedding + index provenance

On float32 vs int8
------------------
float32 is the default. At 256 dimensions the matrix is ~1 KB per passage, so even a 500k-passage
corpus is ~0.5 GB — affordable, and it keeps exact search a single BLAS call. `--quantize int8`
cuts that 4× for when artifact size genuinely binds; the loader handles either, and the quantised
path costs a chunked widen-and-multiply on the exact path. Quantisation error is recorded in the
manifest rather than assumed negligible.

Usage
-----
    python scripts/embed_corpus.py --corpus data/corpus --lane fast
    python scripts/embed_corpus.py --corpus data/corpus --quantize int8 --skip-bm25
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.stages.dense import DenseIndex
from app.stages.embed import get_embedder
from app.stages.lexical import LexicalIndex

log = logging.getLogger("embed_corpus")


def quantize_int8(mat: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Symmetric int8 quantisation with a single global scale.

    A global scale is safe here specifically because the vectors are L2-normalised, so every
    component already lies in [-1, 1] and no single row can blow out the range. Per-row scales
    would buy almost nothing and would cost a multiply per row on every search.

    Returns the quantised matrix, the dequantisation scale, and the measured mean absolute error
    so the manifest can record what the compression actually cost.
    """
    peak = float(np.abs(mat).max())
    scale = peak / 127.0 if peak > 0 else 1.0
    q = np.clip(np.round(mat / scale), -127, 127).astype(np.int8)
    err = float(np.abs(q.astype(np.float32) * scale - mat).mean())
    return q, scale, err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--out", type=Path, default=None, help="Defaults to --corpus")
    ap.add_argument("--lane", choices=["fast", "quality"], default="fast")
    ap.add_argument("--quantize", choices=["none", "int8"], default="none")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--skip-hnsw", action="store_true")
    ap.add_argument("--skip-bm25", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )

    out_dir = args.out or args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    passages_path = args.corpus / "passages.parquet"
    if not passages_path.exists():
        log.error("no passages at %s — run scripts/build_corpus.py first", passages_path)
        return 2

    table = pq.read_table(passages_path)
    texts = table.column("text").to_pylist()
    log.info("loaded %d passages", len(texts))

    # --- embed ------------------------------------------------------------------------
    embedder = get_embedder(args.lane)
    log.info("embedding with lane=%s dim=%d", embedder.lane, embedder.dim)
    t0 = time.perf_counter()

    # Encoded in slices with progress logging rather than one opaque call. Two reasons, both
    # learned the hard way on an 8 GB machine: peak memory stays bounded by one slice's
    # intermediates instead of the whole corpus, and a long run reports progress instead of
    # looking identical to a hang for thirteen minutes.
    vectors = np.empty((len(texts), embedder.dim), dtype=np.float32)
    slice_size = max(args.batch_size, 20_000)
    for start in range(0, len(texts), slice_size):
        chunk = texts[start : start + slice_size]
        vectors[start : start + len(chunk)] = embedder.encode_batch(
            chunk, batch_size=args.batch_size
        )
        done = start + len(chunk)
        rate = done / max(1e-9, time.perf_counter() - t0)
        log.info("  embedded %d/%d (%.0f/s)", done, len(texts), rate)

    embed_s = time.perf_counter() - t0
    log.info(
        "embedded %d passages in %.1fs (%.0f passages/s)", len(texts), embed_s, len(texts) / embed_s
    )

    quant_scale = 1.0
    quant_err = 0.0
    if args.quantize == "int8":
        vectors_out, quant_scale, quant_err = quantize_int8(vectors)
        log.info("quantised to int8: scale=%.6g mean_abs_err=%.6g", quant_scale, quant_err)
    else:
        vectors_out = vectors.astype(np.float32)

    emb_path = out_dir / "embeddings.npy"
    np.save(emb_path, vectors_out)
    log.info("wrote %s (%.1f MB)", emb_path, emb_path.stat().st_size / 1e6)

    # --- indexes ----------------------------------------------------------------------
    hnsw_s = None
    if not args.skip_hnsw:
        t0 = time.perf_counter()
        # HNSW is always built over float32 vectors: quantising before graph construction degrades
        # the neighbour lists themselves, which is a permanent recall cost rather than a
        # recoverable one.
        index = DenseIndex.build(vectors)
        index.save(out_dir / "hnsw.usearch")
        hnsw_s = time.perf_counter() - t0
        log.info("HNSW built + saved in %.1fs", hnsw_s)

    bm25_s = None
    if not args.skip_bm25:
        t0 = time.perf_counter()
        lex = LexicalIndex.build(texts)
        bm25_dir = out_dir / "bm25"
        if bm25_dir.exists():
            shutil.rmtree(bm25_dir)
        lex.save(str(bm25_dir))
        bm25_s = time.perf_counter() - t0
        log.info("BM25 built + saved in %.1fs", bm25_s)

    # --- manifest ---------------------------------------------------------------------
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["embeddings"] = {
        "lane": embedder.lane,
        "model": "minishlab/potion-multilingual-128M" if args.lane == "fast" else "intfloat/multilingual-e5-small",
        "dim": int(vectors.shape[1]),
        "n_vectors": int(vectors.shape[0]),
        "dtype": str(vectors_out.dtype),
        "quant_scale": quant_scale,
        "quant_mean_abs_err": quant_err,
        "embed_seconds": round(embed_s, 2),
    }
    manifest["indexes"] = {
        "hnsw": None if args.skip_hnsw else {"build_seconds": round(hnsw_s or 0, 2)},
        "bm25": None if args.skip_bm25 else {"build_seconds": round(bm25_s or 0, 2)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log.info("manifest updated -> %s", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
