"""
Pre-build full-corpus chunk embeddings for cosine search.

Splits each document into ~512-token chunks preserving sentence boundaries,
then encodes each chunk with the embedding model. The resulting cache maps
each chunk back to its source doc_id, enabling max-pooling per document at
search time.

Usage:
    python -m TRACE.util.build_embeddings
    python -m TRACE.util.build_embeddings --model Qwen/Qwen3-Embedding-8B
    python -m TRACE.util.build_embeddings --chunk-tokens 512
    python -m TRACE.util.build_embeddings --resume
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from ..src.types import CORPUS_DIR, CHUNK_MAX_TOKENS
from ..src.data import build_full_corpus, build_chunked_corpus
from .config import get_settings

DEFAULT_CACHE_DIR = CORPUS_DIR

CHECKPOINT_INTERVAL = 500


def _save_checkpoint(
    embeddings: np.ndarray,
    doc_ids: list[str],
    encoded_count: int,
    out_dir: Path,
) -> None:
    """Save encoding progress so we can resume after OOM."""
    ckpt_dir = out_dir / "_embedding_checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(ckpt_dir / "embeddings.npy"), embeddings[:encoded_count])
    (ckpt_dir / "doc_ids.json").write_text(
        json.dumps(doc_ids[:encoded_count], ensure_ascii=False), encoding="utf-8",
    )
    (ckpt_dir / "progress.json").write_text(
        json.dumps({"encoded": encoded_count}), encoding="utf-8",
    )


def _load_checkpoint(out_dir: Path) -> tuple[np.ndarray | None, list[str], int]:
    """Load previous checkpoint if it exists."""
    ckpt_dir = out_dir / "_embedding_checkpoint"
    progress_path = ckpt_dir / "progress.json"
    if not progress_path.exists():
        return None, [], 0
    try:
        progress = json.loads(progress_path.read_text("utf-8"))
        encoded = progress["encoded"]
        emb = np.load(str(ckpt_dir / "embeddings.npy"))
        ids = json.loads((ckpt_dir / "doc_ids.json").read_text("utf-8"))
        return emb, ids, encoded
    except Exception:
        return None, [], 0


def _cleanup_checkpoint(out_dir: Path) -> None:
    """Remove checkpoint files after successful completion."""
    ckpt_dir = out_dir / "_embedding_checkpoint"
    if ckpt_dir.exists():
        for f in ckpt_dir.iterdir():
            f.unlink()
        ckpt_dir.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunk-based embedding cache")
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_CACHE_DIR,
        help="Output directory for cache files",
    )
    parser.add_argument(
        "--model", type=str, default=get_settings().embedding.model,
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--chunk-tokens", type=int, default=CHUNK_MAX_TOKENS,
        help="Target chunk size in tokens (default: 512)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint (after OOM or interruption)",
    )
    args = parser.parse_args()

    t0 = time.monotonic()

    print("Loading corpus...")
    full_docs = build_full_corpus()
    print(f"  {len(full_docs)} documents loaded")

    print(f"Chunking documents (~{args.chunk_tokens} tokens per chunk)...")
    chunked = build_chunked_corpus(full_docs, max_tokens=args.chunk_tokens)

    chunk_keys = list(chunked.keys())
    chunk_doc_ids = [chunked[k].doc_id for k in chunk_keys]
    chunk_texts = [chunked[k].text_original for k in chunk_keys]

    n_docs = len(full_docs)
    n_chunks = len(chunk_texts)
    print(f"  {n_docs} docs → {n_chunks} chunks (avg {n_chunks / n_docs:.1f} chunks/doc)")

    del full_docs, chunked
    gc.collect()

    start_idx = 0
    cached_emb = None
    if args.resume:
        cached_emb, cached_ids, start_idx = _load_checkpoint(args.out_dir)
        if start_idx > 0:
            if cached_ids == chunk_doc_ids[:start_idx]:
                print(f"  Resuming from checkpoint: {start_idx}/{n_chunks} already encoded")
            else:
                print("  Checkpoint mismatch (corpus changed?), starting fresh")
                start_idx = 0
                cached_emb = None

    print(f"Loading embedding model: {args.model}")
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            args.model,
            model_kwargs={"torch_dtype": torch.float16},
        )
    except Exception as exc:
        raise SystemExit(f"Failed to load embedding model: {exc}") from exc

    device = str(model.device)
    model.max_seq_length = min(args.chunk_tokens + 64, model.max_seq_length)
    print(f"  Model on {device}, max_seq_length={model.max_seq_length}")

    dim = model.get_sentence_embedding_dimension()
    all_embeddings = np.zeros((n_chunks, dim), dtype=np.float32)

    if cached_emb is not None and start_idx > 0:
        all_embeddings[:start_idx] = cached_emb[:start_idx]
        del cached_emb
        gc.collect()

    import torch

    remaining = n_chunks - start_idx
    print(f"Encoding {remaining} chunks (batch_size=1, checkpoint every {CHECKPOINT_INTERVAL})...")

    encoded = start_idx
    last_checkpoint = start_idx

    for i in range(start_idx, n_chunks):
        text = chunk_texts[i]

        with torch.no_grad():
            emb = model.encode(
                [text],
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=1,
            )
        all_embeddings[i] = np.asarray(emb[0], dtype=np.float32)
        encoded += 1

        if encoded % 50 == 0:
            del emb
            gc.collect()
            torch.cuda.empty_cache()

        if encoded % 200 == 0 or encoded == n_chunks:
            elapsed = time.monotonic() - t0
            rate = encoded / elapsed if elapsed > 0 else 0
            eta = (n_chunks - encoded) / rate if rate > 0 else 0
            print(f"  {encoded}/{n_chunks} ({rate:.1f} chunks/s, ETA {eta:.0f}s)")

        if encoded - last_checkpoint >= CHECKPOINT_INTERVAL:
            _save_checkpoint(all_embeddings, chunk_doc_ids, encoded, args.out_dir)
            last_checkpoint = encoded
            print(f"  [checkpoint saved at {encoded}]")

    print(f"  Embedding matrix: {all_embeddings.shape}")

    del chunk_texts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = args.out_dir / "embeddings_cache.npy"
    ids_path = args.out_dir / "embeddings_doc_ids.json"

    np.save(str(npy_path), all_embeddings)
    ids_path.write_text(
        json.dumps(chunk_doc_ids, ensure_ascii=False),
        encoding="utf-8",
    )

    _cleanup_checkpoint(args.out_dir)

    elapsed = time.monotonic() - t0
    print(f"\nSaved {n_chunks} chunk embeddings ({n_docs} docs) to:")
    print(f"  {npy_path} ({npy_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  {ids_path}")
    print(f"Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
