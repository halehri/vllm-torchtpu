"""QSA sparse prefill attention at the served SP shapes: tiles, union and bytes.

One TP rank of Qwen3.8-Flash (3 query heads x 256, one KV head of the FP8
unified pool with 512-token pages, ratio 4, top-512 groups per query row).
Steps of 2048 and 8192 tokens: one long prefill request, or four prefill
requests with different contexts. Selection patterns per query row:

- recent: the last 512 groups (best case for KV reuse);
- uniform: 512 groups uniformly over the context (worst case);
- hot: 128 recent groups plus 384 drawn from a per-request hot set of 1024
  groups, so neighbouring rows overlap like a served indexer does.

Every (query_block, kv_tile, query_tile) arm, with and without the BF16 MXU
path, is timed, checked against the served tiles, and reported with the KV
tiles each query block visits (the union of its rows' selections), the bytes
that fetch moves, the bytes the step actually needs (unique selected groups)
and the dense score work of the visited tiles.
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

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (  # noqa: E402
    block_sparse_gqa_attention,
)

HEADS, HEAD_DIM, RATIO, TOPK_GROUPS = 3, 256, 4, 512
PAGE = 512
TOKEN_BYTES = 4 * HEAD_DIM  # one pool row: [1, 4, 256] FP8
ITERS = 10
# (query_block, kv_tile, query_tile); the first is the served SHARD_PREFILL_TILES.
ARMS = [
    (512, 1024, 256),
    (256, 1024, 256),
    (512, 512, 256),
    (256, 512, 256),
    (256, 256, 256),
    (128, 512, 128),
    (128, 256, 128),
    (64, 256, 64),
]


def make_step(rng, tokens, layout, pattern):
    if layout == "one":
        query_lens, contexts = [tokens], [94000]
    else:
        share = tokens // 4
        query_lens, contexts = [share] * 4, [12000, 40000, 70000, 94000]
    starts = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)
    columns = (max(c + q for c, q in zip(contexts, query_lens)) + PAGE - 1) // PAGE
    pages = len(query_lens) * columns + 1
    table = np.zeros((tokens, columns), np.int32)
    lengths = np.zeros(tokens, np.int32)
    blocks = np.full((tokens, TOPK_GROUPS), -1, np.int32)
    free = rng.permutation(np.arange(1, pages))
    for r, (q, c) in enumerate(zip(query_lens, contexts)):
        rows = slice(starts[r], starts[r + 1])
        table[rows] = free[r * columns : (r + 1) * columns]
        lengths[rows] = c + np.arange(1, q + 1)
        hot = rng.choice(c // RATIO, 1024, replace=False)
        for i in range(starts[r], starts[r + 1]):
            complete = lengths[i] // RATIO
            count = min(complete, TOPK_GROUPS)
            if pattern == "recent":
                chosen = np.arange(complete - count, complete)
            elif pattern == "uniform":
                chosen = np.unique(rng.integers(0, complete, 2 * count))[:count]
            else:
                near = np.arange(complete - 128, complete)
                far = rng.choice(hot, count - 128, replace=False)
                chosen = np.unique(np.concatenate([near, far]))[:count]
            blocks[i, : len(chosen)] = np.sort(chosen)
    query = rng.normal(0, 0.3, (tokens, HEADS, HEAD_DIM)).astype(jnp.bfloat16)
    gate = rng.uniform(-3, 3, query.shape).astype(jnp.bfloat16)
    cache = rng.normal(0, 0.3, (pages, PAGE, 1, 4, HEAD_DIM)).astype(jnp.float8_e4m3fn)
    return dict(
        query=jnp.asarray(query),
        gate=jnp.asarray(gate),
        blocks=jnp.asarray(blocks),
        lengths=jnp.asarray(lengths),
        table=jnp.asarray(table),
        cache=jnp.asarray(cache),
        starts=jnp.asarray(starts),
        np_blocks=blocks,
        np_starts=starts,
    )


def union_stats(step, query_block, kv_tile):
    """KV tiles visited per query block (a block never spans two requests)."""
    blocks, starts = step["np_blocks"], step["np_starts"]
    visited, unique_groups = 0, 0
    for r in range(len(starts) - 1):
        rows = blocks[starts[r] : starts[r + 1]]
        unique_groups += len(np.unique(rows[rows >= 0]))
        for b in range(0, len(rows), query_block):
            part = rows[b : b + query_block]
            tiles = np.unique(part[part >= 0] * RATIO // kv_tile)
            visited += len(tiles)
    fetched = visited * kv_tile * TOKEN_BYTES
    needed = unique_groups * RATIO * TOKEN_BYTES
    padded_rows = sum(
        -(-(starts[r + 1] - starts[r]) // query_block) * query_block
        for r in range(len(starts) - 1)
    )
    dense_flop = visited * query_block * HEADS * kv_tile * HEAD_DIM * 4
    return dict(
        visited_tiles=int(visited),
        fetched_mb=fetched / 2**20,
        needed_mb=needed / 2**20,
        padded_rows=int(padded_rows),
        dense_gflop=dense_flop / 1e9,
    )


def run_arm(step, arm, bf16_mxu):
    query_block, kv_tile, query_tile = arm
    envs.VLLM_TPU_QSA_PREFILL_BF16_MXU = bf16_mxu

    def fn(query, gate, blocks, lengths, table, cache, starts):
        return block_sparse_gqa_attention(
            query, gate, blocks, lengths, table, cache, starts,
            kv_heads=1, ratio=RATIO, query_block=query_block, kv_tile=kv_tile,
            query_tile=query_tile,
        )

    inputs = [step[k] for k in ("query", "gate", "blocks", "lengths", "table", "cache", "starts")]
    compiled = jax.jit(fn).lower(*inputs).compile()
    out = None
    for _ in range(3):
        out = jax.block_until_ready(compiled(*inputs))
    samples = []
    for _ in range(ITERS):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*inputs))
        samples.append((time.perf_counter() - start) * 1000)
    first = out[0] if isinstance(out, (tuple, list)) else out
    return statistics.median(samples), np.asarray(first, np.float32)


def test_prefill_attention():
    rng = np.random.default_rng(1964)
    results = []
    for tokens in (2048, 8192):
        for layout in ("one", "four"):
            for pattern in ("recent", "hot", "uniform"):
                step = make_step(rng, tokens, layout, pattern)
                reference = None
                for arm in ARMS:
                    for bf16_mxu in (False, True):
                        try:
                            ms, out = run_arm(step, arm, bf16_mxu)
                        except Exception as e:  # unsupported tile combination
                            row = dict(tokens=tokens, layout=layout, pattern=pattern,
                                       arm=arm, bf16_mxu=bf16_mxu, error=str(e)[:200])
                            print("PATTN " + json.dumps(row), flush=True)
                            continue
                        if reference is None:
                            reference = out
                        rel = float(np.linalg.norm(out - reference) / np.linalg.norm(reference))
                        row = dict(tokens=tokens, layout=layout, pattern=pattern, arm=arm,
                                   bf16_mxu=bf16_mxu, ms=ms, rel_l2=rel,
                                   **union_stats(step, arm[0], arm[1]))
                        results.append(row)
                        print("PATTN " + json.dumps(row), flush=True)
    out_dir = Path(os.environ.get("CDK_OUTPUT_DIR", "/tmp"))
    (out_dir / "prefill_attn.json").write_text(json.dumps(results, indent=1))
    assert results
