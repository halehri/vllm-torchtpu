# SPDX-License-Identifier: Apache-2.0
"""Debug-only QSA prefill selection statistics (`VLLM_TPU_QSA_UNION_STATS`).

For each 512-row query block of a step (the served prefill attention's
block), the 1024-token KV tiles the block's rows select at least one token
from and the unique compressed groups they select, accumulated by context
bucket into a small int32 device buffer. No host callback: the buffer is a
module buffer the caller reads outside the compiled graph.
"""

import functools

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

BLOCK, TILE, RATIO = 512, 1024, 4
CTX_EDGES = (16000, 32000, 64000, 128000)
BUCKETS = len(CTX_EDGES) + 1
MAX_TILES = 128
MAX_GROUPS = 128 * TILE // RATIO
STATS = 3 * BUCKETS  # blocks, visited tiles, unique groups per bucket


def union_stats_step(selected: jax.Array, stats: jax.Array) -> jax.Array:
    """``stats`` plus this step's per-block sums; ``selected`` is the op's
    [T, k] selected token positions (-1 for none)."""
    blocks = selected.shape[0] // BLOCK
    if blocks == 0:
        return stats
    sel = selected[: blocks * BLOCK].reshape(blocks, -1)
    rows = jnp.arange(blocks, dtype=jnp.int32)[:, None]
    valid = sel >= 0
    tiles = jnp.where(valid, jnp.minimum(sel // TILE, MAX_TILES - 1), MAX_TILES)
    tile_hit = jnp.zeros((blocks, MAX_TILES + 1), jnp.bool_).at[rows, tiles].set(True)
    groups = jnp.where(valid, jnp.minimum(sel // RATIO, MAX_GROUPS - 1), MAX_GROUPS)
    group_hit = jnp.zeros((blocks, MAX_GROUPS + 1), jnp.bool_).at[rows, groups].set(True)
    n_tiles = jnp.sum(tile_hit[:, :MAX_TILES], axis=1, dtype=jnp.int32)
    n_groups = jnp.sum(group_hit[:, :MAX_GROUPS], axis=1, dtype=jnp.int32)
    context = jnp.max(sel, axis=1)
    bucket = sum((context >= edge).astype(jnp.int32) for edge in CTX_EDGES)
    member = (bucket[:, None] == jnp.arange(BUCKETS)[None, :]) & (context[:, None] >= 0)
    member = member.astype(jnp.int32)
    add = jnp.concatenate(
        [
            jnp.sum(member, axis=0),
            jnp.sum(member * n_tiles[:, None], axis=0),
            jnp.sum(member * n_groups[:, None], axis=0),
        ]
    )
    return stats + add


@functools.cache
def build_union_stats_op():
    def core(selected: jax.Array, stats: jax.Array) -> jax.Array:
        return union_stats_step(selected, stats)

    op = pallas.jax_op("qwen4_exp::qsa_union_stats_v1", core)

    @op.register_fake
    def fake(selected, stats):
        return torch.empty_like(stats)

    return op


def format_stats(values) -> str:
    names = [f"<{e // 1000}K" for e in CTX_EDGES] + [f">={CTX_EDGES[-1] // 1000}K"]
    parts = []
    for b, name in enumerate(names):
        n = int(values[b])
        if not n:
            continue
        tiles = int(values[BUCKETS + b]) / n
        groups = int(values[2 * BUCKETS + b]) / n
        parts.append(
            f"ctx{name}: blocks={n} tiles/block={tiles:.1f} groups/block={groups:.1f} "
            f"tile_tokens/union_tokens={tiles * TILE / max(groups * RATIO, 1):.1f}"
        )
    return "; ".join(parts)
