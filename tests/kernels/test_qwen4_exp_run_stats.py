# SPDX-License-Identifier: Apache-2.0
"""Debug QSA decode selection run lengths against a NumPy count."""

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.qwen4_exp.run_stats import (
    FIELDS,
    STATS,
    format_run_stats,
    run_stats_step,
)


def _ref(sel):
    out = dict.fromkeys(FIELDS, 0)
    for row in sel:
        g = sorted({int(x) // 4 for x in row if x >= 0})
        if not g:
            continue
        out["rows"] += 1
        out["groups"] += len(g)
        for p in (32, 128):
            starts = [i for i, x in enumerate(g) if i == 0 or x != g[i - 1] + 1 or x // p != g[i - 1] // p]
            out[f"runs_p{p}"] += len(starts)
            for c in (4, 8):
                n, s = 0, 0
                for i in range(len(g)):
                    if i in starts:
                        s = i
                    n += (i - s) % c == 0
                out[f"dma_c{c}_p{p}"] += n
    return [out[f] for f in FIELDS]


def test_run_stats_counts_runs_and_coalesced_copies():
    rng = np.random.default_rng(1)
    sel = np.full((12, 64), -1, np.int32)
    for r in range(12):
        base = rng.integers(0, 3000)
        runs = [base + 4 * k for k in range(rng.integers(1, 12))]  # one contiguous run
        rest = rng.choice(20000, 30, replace=False) * 4 + rng.integers(0, 4)
        toks = np.array(runs + list(rest) + [runs[0] + 1])  # a same-group duplicate
        sel[r, : len(toks)] = toks
    sel[11] = -1
    got = np.asarray(run_stats_step(jnp.asarray(sel), jnp.zeros(STATS, jnp.int32))).tolist()
    assert got == _ref(sel)
    assert "groups/run" in format_run_stats(got)


def test_run_stats_skips_prefill_sized_steps():
    stats = run_stats_step(jnp.zeros((512, 8), jnp.int32), jnp.ones(STATS, jnp.int32))
    assert np.asarray(stats).tolist() == [1] * STATS
