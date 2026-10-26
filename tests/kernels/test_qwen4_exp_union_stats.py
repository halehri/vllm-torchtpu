# SPDX-License-Identifier: Apache-2.0
"""Debug QSA prefill selection-union statistics against a NumPy count."""

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.qwen4_exp.union_stats import (
    BUCKETS,
    STATS,
    build_union_stats_op,
    format_stats,
    union_stats_step,
)


def test_union_stats_counts_tiles_and_groups():
    rng = np.random.default_rng(0)
    tokens, k = 1024 + 100, 64
    sel = np.full((tokens, k), -1, np.int32)
    for i in range(tokens):
        sel[i, : k - 4] = rng.choice(40000, k - 4, replace=False)  # ctx bucket <64K
    stats = np.asarray(union_stats_step(jnp.asarray(sel), jnp.zeros(STATS, jnp.int32)))
    blocks = sel[:1024].reshape(2, -1)
    bucket = 2  # 32K <= max < 64K
    assert stats[bucket] == 2
    for u, unit in enumerate((1024, 256, 128, 4)):
        want = sum(len(np.unique(b[b >= 0] // unit)) for b in blocks)
        assert stats[(1 + u) * BUCKETS + bucket] == want
    assert "ctx<64K" in format_stats(stats)


def test_union_stats_skips_decode_steps():
    stats = union_stats_step(jnp.full((64, 8), 5, jnp.int32), jnp.ones(STATS, jnp.int32))
    assert np.asarray(stats).tolist() == [1] * STATS


def test_union_stats_op_registers():
    assert build_union_stats_op() is not None
