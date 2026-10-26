"""QSA decode indexer: per-row 1-token sequences vs the native ragged MTP batch.

Decode/verify steps with MTP K=3 hold 4 rows per request (lengths L..L+3) over
one compressed-key history. Today every row is its own 1-token sequence
(cu = arange(rows + 1), one query per block), so each row streams the whole
history. The ragged form passes cu = request starts (4 rows each) as
chunked-prefill sequences with a 4- or 8-query block, so each KV block loads
once per request. Prints IDXR lines (ms per call, per-row top-k set equality).
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp.indexer_components import qsa_indexing  # noqa: E402

HEADS, DIM, RATIO, TOPK, GROUPS_PER_PAGE, ROWS = 8, 128, 4, 512, 32, 4
ITERS = 20


def _call(q, cache, lens, pages, cu, dist, nq):
    return streamindex_topk(
        q,
        jnp.full(q.shape[:2], q.shape[-1] ** -0.5, jnp.float32),
        cache,
        lens,
        pages,
        cu,
        dist,
        k=TOPK,
        compression_ratio=RATIO,
        num_kv_pages_per_block=(256 // GROUPS_PER_PAGE,) * 3,
        num_queries_per_block=nq,
        decode_req_batch_size=8,
        enable_early_exit=True,
        components=qsa_indexing(),
    )


def _timed(f, *args):
    compiled = jax.jit(f).lower(*args).compile()
    for _ in range(3):
        jax.block_until_ready(compiled(*args))
    samples = []
    for _ in range(ITERS):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples), np.asarray(compiled(*args))


def test_ragged_vs_per_row():
    rng = np.random.default_rng(5)
    for context in (32_000, 96_000):
        for batch in (8, 32):
            pages_per_seq = -(-(context + ROWS) // (RATIO * GROUPS_PER_PAGE)) + 1
            total_pages = batch * pages_per_seq
            compressed = jnp.asarray(
                rng.normal(size=(total_pages, GROUPS_PER_PAGE, 1, DIM)), jnp.bfloat16
            )
            cache = compressed.reshape(total_pages, GROUPS_PER_PAGE // 2, 2, -1)
            table = rng.permutation(total_pages).reshape(batch, pages_per_seq).astype(np.int32)
            base = rng.integers(context - 4096, context, size=batch).astype(np.int32)
            q = jnp.asarray(rng.normal(size=(batch * ROWS, HEADS, DIM)), jnp.bfloat16)
            req = np.repeat(np.arange(batch), ROWS)
            row_lens = (base[req] + np.tile(np.arange(1, ROWS + 1), batch)).astype(np.int32)
            n = batch * ROWS
            per_row = dict(
                lens=jnp.asarray(row_lens),
                pages=jnp.asarray(table[req].reshape(-1)),
                cu=jnp.arange(n + 1, dtype=jnp.int32),
                dist=jnp.full((3,), n, jnp.int32),
            )
            ragged = dict(
                lens=jnp.asarray(base + ROWS),
                pages=jnp.asarray(table.reshape(-1)),
                cu=jnp.arange(batch + 1, dtype=jnp.int32) * ROWS,
                dist=jnp.asarray([0, batch, batch], jnp.int32),
            )
            t_row, ids_row = _timed(
                lambda q, c, l, p, u, d: _call(q, c, l, p, u, d, 1),
                q, cache, *per_row.values(),
            )
            for bq in (4, 8):
                t_rag, ids_rag = _timed(
                    lambda q, c, l, p, u, d, bq=bq: _call(q, c, l, p, u, d, (1, bq, bq)),
                    q, cache, *ragged.values(),
                )
                same = all(
                    np.array_equal(np.sort(ids_row[r]), np.sort(ids_rag[r])) for r in range(n)
                )
                print("IDXR " + json.dumps(dict(
                    context=context, requests=batch, bq=bq, per_row_ms=t_row,
                    ragged_ms=t_rag, speedup=t_row / t_rag, topk_sets_equal=same)), flush=True)
