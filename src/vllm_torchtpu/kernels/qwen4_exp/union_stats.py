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
MAX_CONTEXT = 256 * 1024
# KV granularities whose touched-unit counts are reported per block: the
# served 1024-token tile, two finer candidate tiles, and the compressed group.
UNITS = (TILE, 256, 128, RATIO)
STATS = (1 + len(UNITS)) * BUCKETS  # blocks, then touched units per granularity


def union_stats_step(selected: jax.Array, stats: jax.Array) -> jax.Array:
    """``stats`` plus this step's per-block sums; ``selected`` is the op's
    [T, k] selected token positions (-1 for none)."""
    blocks = selected.shape[0] // BLOCK
    if blocks == 0:
        return stats
    sel = selected[: blocks * BLOCK].reshape(blocks, -1)
    rows = jnp.arange(blocks, dtype=jnp.int32)[:, None]
    valid = sel >= 0
    counts = []
    for unit in UNITS:
        cap = MAX_CONTEXT // unit
        idx = jnp.where(valid, jnp.minimum(sel // unit, cap - 1), cap)
        hit = jnp.zeros((blocks, cap + 1), jnp.bool_).at[rows, idx].set(True)
        counts.append(jnp.sum(hit[:, :cap], axis=1, dtype=jnp.int32))
    context = jnp.max(sel, axis=1)
    bucket = sum((context >= edge).astype(jnp.int32) for edge in CTX_EDGES)
    member = (bucket[:, None] == jnp.arange(BUCKETS)[None, :]) & (context[:, None] >= 0)
    member = member.astype(jnp.int32)
    add = jnp.concatenate(
        [jnp.sum(member, axis=0)]
        + [jnp.sum(member * c[:, None], axis=0) for c in counts]
    )
    return stats + add


@functools.cache
def build_union_stats_op():
    def core(selected: jax.Array, stats: jax.Array) -> jax.Array:
        return union_stats_step(selected, stats)

    op = pallas.jax_op("qwen4_exp::qsa_union_stats_v2", core)

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
        per = [int(values[(1 + u) * BUCKETS + b]) / n for u in range(len(UNITS))]
        union = per[-1] * RATIO
        ratios = " ".join(
            f"t{unit}={per[u] * unit / max(union, 1):.2f}" for u, unit in enumerate(UNITS[:-1])
        )
        parts.append(
            f"ctx{name}: blocks={n} tiles/block={per[0]:.1f} groups/block={per[-1]:.1f} "
            f"tile_tokens/union_tokens {ratios}"
        )
    return "; ".join(parts)
