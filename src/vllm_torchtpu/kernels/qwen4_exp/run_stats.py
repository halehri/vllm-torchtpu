# SPDX-License-Identifier: Apache-2.0
"""Debug-only QSA decode selection run lengths (`VLLM_TPU_QSA_RUN_STATS`).

For decode / verify steps (at most 256 rows), per row: the selected compressed
groups, the runs of consecutive groups inside one pool page, and how many DMA
copies a coalescing reader would issue with runs capped at 4 or 8 groups, for
32- and 128-group pages. Accumulated into a small int32 device buffer the
caller reads outside the compiled graph.
"""

import functools

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

RATIO = 4
MAX_ROWS = 256
PAGES = (32, 128)  # groups per pool page (128- and 512-token pages)
CAPS = (4, 8)
FIELDS = ("rows", "groups") + tuple(
    f"{k}_p{p}" for p in PAGES for k in ("runs",) + tuple(f"dma_c{c}" for c in CAPS)
)
STATS = len(FIELDS)


def run_stats_step(selected: jax.Array, stats: jax.Array) -> jax.Array:
    """``stats`` plus this step's sums; ``selected``: [T, k] token positions (-1: none)."""
    if selected.shape[0] > MAX_ROWS:
        return stats
    big = jnp.iinfo(jnp.int32).max
    g = jnp.where(selected >= 0, selected // RATIO, big)
    g = jnp.sort(g, axis=1)
    valid = g != big
    prev = jnp.concatenate([jnp.full_like(g[:, :1], -2), g[:, :-1]], axis=1)
    dup = valid & (g == prev)
    valid = valid & ~dup
    # Position among the row's distinct groups (duplicates take no slot).
    idx = jnp.cumsum(valid.astype(jnp.int32), axis=1) - 1
    rows = jnp.sum(jnp.any(valid, axis=1).astype(jnp.int32))
    out = [rows, jnp.sum(valid.astype(jnp.int32))]
    for p in PAGES:
        new = valid & ((g != prev + 1) | (g // p != prev // p))
        start = jax.lax.cummax(jnp.where(new, idx, -1), axis=1)
        out.append(jnp.sum(new.astype(jnp.int32)))
        for c in CAPS:
            out.append(jnp.sum((valid & ((idx - start) % c == 0)).astype(jnp.int32)))
    return stats + jnp.stack(out).astype(jnp.int32)


@functools.cache
def build_run_stats_op():
    def core(selected: jax.Array, stats: jax.Array) -> jax.Array:
        return run_stats_step(selected, stats)

    op = pallas.jax_op("qwen4_exp::qsa_run_stats_v1", core)

    @op.register_fake
    def fake(selected, stats):
        return torch.empty_like(stats)

    return op


def format_run_stats(values) -> str:
    v = dict(zip(FIELDS, (int(x) for x in values)))
    rows, groups = max(v["rows"], 1), max(v["groups"], 1)
    parts = [f"rows={v['rows']} groups/row={groups / rows:.1f}"]
    for p in PAGES:
        parts.append(
            f"p{p}: groups/run={groups / max(v[f'runs_p{p}'], 1):.2f} "
            + " ".join(f"dma_c{c}/groups={v[f'dma_c{c}_p{p}'] / groups:.3f}" for c in CAPS)
        )
    return "; ".join(parts)
