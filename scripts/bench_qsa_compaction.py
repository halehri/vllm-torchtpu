"""QSA sparse prefill attention: does gather compaction pay at real union sizes?

Per-rank SP geometry: 1024 query rows (two 512-row query blocks), 3 query
heads, one KV head, FP8 pool of 512-token pages, ratio 4, top-512 groups per
row. Per context bucket, the union of groups a query block selects and its
spread over KV tiles come from the u04 real-traffic stats (groups/block and
tile_tokens/union_tokens). For each bucket:

  spread     every row picks 512 groups from a block pool of G groups spread
             over ceil(G * 4 * ratio / 1024) tiles (today's served layout)
  compacted  the same pool packed into the first ceil(G / 256) tiles (what the
             kernel would visit after compaction)
  gather     an XLA gather of each block's G * 4 union tokens out of the paged
             pool into a contiguous buffer (the compaction's extra traffic)

net = spread - (compacted + gather), reported per call.
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

from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (  # noqa: E402
    block_sparse_gqa_attention,
)

HEADS, HEAD_DIM, RATIO, TOPK = 3, 256, 4, 512
PAGE, KV_TILE, QUERY_BLOCK, QUERY_TILE = 512, 1024, 512, 256
TOKENS, ITERS = 1024, 20
# (context, groups per block, tile_tokens / union_tokens) from u04 (j-d09da003).
BUCKETS = [(24_000, 5369, 1.13), (48_000, 9210, 1.30), (96_000, 14432, 1.61),
           (160_000, 19852, 2.33)]


def make_inputs(rng, context, pool_span, groups):
    columns = (context + TOKENS + PAGE - 1) // PAGE
    pages = columns + 1
    table = np.broadcast_to(rng.permutation(np.arange(1, pages)), (TOKENS, columns)).copy()
    lengths = context + np.arange(1, TOKENS + 1, dtype=np.int32)
    blocks = np.full((TOKENS, TOPK), -1, np.int32)
    for b in range(TOKENS // QUERY_BLOCK):
        pool = np.sort(rng.choice(pool_span, groups, replace=False))
        for i in range(b * QUERY_BLOCK, (b + 1) * QUERY_BLOCK):
            blocks[i] = np.sort(rng.choice(pool, TOPK, replace=False))
    query = rng.normal(0, 0.3, (TOKENS, HEADS, HEAD_DIM)).astype(jnp.bfloat16)
    gate = rng.uniform(-3, 3, query.shape).astype(jnp.bfloat16)
    cache = rng.normal(0, 0.3, (pages, PAGE, 1, 4, HEAD_DIM)).astype(jnp.float8_e4m3fn)
    return [jnp.asarray(x) for x in (query, gate, blocks, lengths, table, cache)] + [
        jnp.array([0, TOKENS], jnp.int32)
    ]


def attend(query, gate, blocks, lengths, table, cache, starts):
    return block_sparse_gqa_attention(
        query, gate, blocks, lengths, table, cache, starts,
        kv_heads=1, ratio=RATIO, query_block=QUERY_BLOCK, kv_tile=KV_TILE,
        query_tile=QUERY_TILE,
    )


def gather(cache, table, token_ids):
    """Each block's union tokens [blocks, n] out of the paged pool."""
    page = jnp.take(table[0], token_ids // PAGE, axis=0)
    return cache[page, token_ids % PAGE]


def timed(f, *args):
    compiled = jax.jit(f).lower(*args).compile()
    for _ in range(3):
        jax.block_until_ready(compiled(*args))
    samples = []
    for _ in range(ITERS):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def test_compaction_net():
    rng = np.random.default_rng(11)
    for context, groups, ratio in BUCKETS:
        spread_tiles = min(-(-int(groups * RATIO * ratio) // KV_TILE), context // KV_TILE)
        compact_tiles = -(-groups // (KV_TILE // RATIO))
        spread = make_inputs(rng, context, spread_tiles * KV_TILE // RATIO, groups)
        compact = make_inputs(rng, context, compact_tiles * KV_TILE // RATIO, groups)
        t_spread = timed(attend, *spread)
        t_compact = timed(attend, *compact)
        n = groups * RATIO
        ids = jnp.asarray(np.stack([np.sort(rng.choice(context, n, replace=False))
                                    for _ in range(TOKENS // QUERY_BLOCK)]).astype(np.int32))
        t_gather = timed(gather, spread[5], spread[4], ids)
        row = dict(context=context, groups=groups, spread_tiles=spread_tiles,
                   compact_tiles=compact_tiles, spread_ms=t_spread, compact_ms=t_compact,
                   gather_ms=t_gather, net_ms=t_spread - t_compact - t_gather,
                   net_frac=(t_spread - t_compact - t_gather) / t_spread)
        print("PCMP " + json.dumps(row), flush=True)
