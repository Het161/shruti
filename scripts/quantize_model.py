#!/usr/bin/env python3
"""Quantise the Model2Vec lookup table to int8 and verify it still retrieves.

512 MB -> 128 MB. This is what makes the free tier possible: measured, the model is the entire
memory cost of the service, and the corpus is a rounding error beside it.

The script does not just convert — it checks. Cosine agreement against the float32 model is
measured on real corpus text, because a quantisation that silently degrades embeddings would
produce a service that loads fine and retrieves badly.

Usage
-----
    python scripts/quantize_model.py --out models/potion-int8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("quantize_model")

MODEL_ID = "minishlab/potion-multilingual-128M"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--out", type=Path, default=Path("models/potion-int8"))
    ap.add_argument("--sample", type=int, default=2000, help="Texts used to verify agreement")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    from huggingface_hub import hf_hub_download
    from model2vec import StaticModel

    log.info("loading %s (float32)", args.model)
    model = StaticModel.from_pretrained(args.model)
    table = np.asarray(model.embedding, dtype=np.float32)
    log.info("table %s float32 = %.0f MB", table.shape, table.nbytes / 1e6)

    # One global symmetric scale. Per-row scales would buy little here — the rows are a distilled
    # embedding space with comparable magnitudes — and would cost a multiply per token at query
    # time plus another 2 MB of scales to carry.
    peak = float(np.abs(table).max())
    scale = peak / 127.0
    q = np.clip(np.round(table / scale), -127, 127).astype(np.int8)
    err = float(np.abs(q.astype(np.float32) * scale - table).mean())
    log.info("int8 %.0f MB  scale=%.8g  mean_abs_err=%.6f  (table std %.6f)",
             q.nbytes / 1e6, scale, err, float(table.std()))

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "embedding_int8.npy", q)

    cfg = json.loads(Path(hf_hub_download(args.model, "config.json")).read_text())
    (args.out / "int8_meta.json").write_text(json.dumps({
        "source_model": args.model,
        "scale": scale,
        "dim": int(table.shape[1]),
        "vocab": int(table.shape[0]),
        "normalize": bool(cfg.get("normalize", True)),
        "apply_zipf": cfg.get("apply_zipf"),
        "sif_coefficient": cfg.get("sif_coefficient"),
        "quant_mean_abs_err": err,
    }, indent=2))
    tok = Path(hf_hub_download(args.model, "tokenizer.json"))
    (args.out / "tokenizer.json").write_bytes(tok.read_bytes())
    log.info("wrote %s", args.out)

    # --- verify -----------------------------------------------------------------------
    # Agreement is measured on real corpus passages, not synthetic strings: quantisation error
    # depends on which rows a text touches, and corpus text touches the rows that matter.
    corpus = Path("data/corpus-small/passages.parquet")
    if not corpus.exists():
        corpus = Path("data/corpus/passages.parquet")
    if corpus.exists():
        import pyarrow.parquet as pq

        from app.stages.embed_int8 import Int8StaticEmbedder

        texts = pq.read_table(corpus, columns=["text"]).column("text").to_pylist()[: args.sample]
        ref = np.asarray(model.encode(texts, use_multiprocessing=False), dtype=np.float32)
        ref /= np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12)
        got = Int8StaticEmbedder(args.out).encode_batch(texts)
        cos = (ref * got).sum(axis=1)
        log.info("cosine agreement vs float32 on %d passages:", len(texts))
        log.info("  mean %.6f   p05 %.6f   min %.6f", cos.mean(), np.percentile(cos, 5), cos.min())
        if cos.mean() < 0.99:
            log.warning("agreement below 0.99 — check before shipping")
    else:
        log.warning("no corpus found; skipped verification")
    return 0


if __name__ == "__main__":
    sys.exit(main())
