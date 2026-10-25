"""QSA sparse prefill attention: the fixed cost per call and per query block.

Same per-rank geometry as bench_qsa_prefill_attn.py (3 query heads, one KV
head, FP8 pool of 512-token pages, ratio 4, top-512 groups). Every query row of
a request selects groups inside the first ``k`` KV tiles of its context, so
each query block visits exactly ``k`` tiles. Timing ``k`` = 1..32 at 512, 2048
and 8192 rows separates a + b*k into the per-call/per-block fixed part ``a``
and the per-tile part ``b``.
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (  # noqa: E402
    block_sparse_gqa_attention,
)

HEADS, HEAD_DIM, RATIO, TOPK = 3, 256, 4, 512
PAGE, CONTEXT, KV_TILE, QUERY_BLOCK, QUERY_TILE = 512, 94000, 1024, 512, 256
ITERS = 20


def make_step(rng, tokens, tiles):
    columns = (CONTEXT + tokens + PAGE - 1) // PAGE
    pages = columns + 1
    table = np.broadcast_to(rng.permutation(np.arange(1, pages)), (tokens, columns)).copy()
    lengths = CONTEXT + np.arange(1, tokens + 1, dtype=np.int32)
    span = tiles * KV_TILE // RATIO  # groups covered by the first `tiles` tiles
    blocks = np.full((tokens, TOPK), -1, np.int32)
    count = min(TOPK, span)
    for i in range(tokens):
        blocks[i, :count] = np.sort(rng.choice(span, count, replace=False))
    query = rng.normal(0, 0.3, (tokens, HEADS, HEAD_DIM)).astype(jnp.bfloat16)
    gate = rng.uniform(-3, 3, query.shape).astype(jnp.bfloat16)
    cache = rng.normal(0, 0.3, (pages, PAGE, 1, 4, HEAD_DIM)).astype(jnp.float8_e4m3fn)
    return [jnp.asarray(x) for x in (query, gate, blocks, lengths, table, cache)] + [
        jnp.array([0, tokens], jnp.int32)
    ]


def fn(query, gate, blocks, lengths, table, cache, starts):
    return block_sparse_gqa_attention(
        query, gate, blocks, lengths, table, cache, starts,
        kv_heads=1, ratio=RATIO, query_block=QUERY_BLOCK, kv_tile=KV_TILE,
        query_tile=QUERY_TILE,
    )


def test_fixed_cost():
    rng = np.random.default_rng(7)
    rows = []
    for tokens in (512, 2048, 8192):
        for tiles in (1, 2, 4, 8, 16, 32):
            inputs = make_step(rng, tokens, tiles)
            compiled = jax.jit(fn).lower(*inputs).compile()
            for _ in range(3):
                jax.block_until_ready(compiled(*inputs))
            samples = []
            for _ in range(ITERS):
                start = time.perf_counter()
                jax.block_until_ready(compiled(*inputs))
                samples.append((time.perf_counter() - start) * 1000)
            row = dict(tokens=tokens, tiles_per_block=tiles,
                       blocks=-(-tokens // QUERY_BLOCK), ms=statistics.median(samples))
            rows.append(row)
            print("PFIX " + json.dumps(row), flush=True)
    trace_dir = os.environ.get("CDK_OUTPUT_DIR", "/tmp") + "/trace"
    inputs = make_step(rng, 2048, 2)
    compiled = jax.jit(fn).lower(*inputs).compile()
    jax.block_until_ready(compiled(*inputs))
    with jax.profiler.trace(trace_dir):
        for _ in range(3):
            jax.block_until_ready(compiled(*inputs))
    for tokens in (512, 2048, 8192):
        pts = [(r["tiles_per_block"], r["ms"]) for r in rows if r["tokens"] == tokens]
        k = np.array([p[0] for p in pts], float)
        t = np.array([p[1] for p in pts], float)
        b, a = np.polyfit(k, t, 1)
        print("PFIT " + json.dumps(dict(tokens=tokens, fixed_ms=a, per_tile_per_block_ms=b)), flush=True)
    assert rows
