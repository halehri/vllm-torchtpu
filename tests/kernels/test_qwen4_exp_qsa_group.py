# SPDX-License-Identifier: Apache-2.0
"""QSA KV-group split (VLLM_TPU_QSA_GROUP_SPLIT): split vs unsplit on CPU."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from test_qwen4_exp_paired_pages import pack_rows

from vllm_torchtpu.kernels.qwen4_exp import qsa_group, qsa_rpa

GROUP, HEADS, DIM, HIDDEN, RATIO = 4, 3, 256, 128, 4
EPS, ROTARY = 1e-6, 64


def _trig():
    angle = jnp.arange(4096)[:, None] * jnp.arange(ROTARY // 2)[None, :] / 8192
    return jnp.pad(
        jnp.concatenate([jnp.cos(angle), jnp.sin(angle)], 1), ((0, 0), (0, 64))
    ).astype(jnp.bfloat16)


@pytest.fixture
def interpret(monkeypatch):
    original = pl.pallas_call
    monkeypatch.setattr(
        pl,
        "pallas_call",
        lambda *args, **kwargs: original(*args, **{**kwargs, "interpret": True}),
    )
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)

    # Prefill-only steps (every request longer than a verify step) have an
    # empty decode prefix; its RPA call only threads the cache through.
    def no_decode(query, gate, blocks, lengths, table, cache, **_):
        return jnp.zeros(query.shape, jnp.float32), cache

    monkeypatch.setattr(qsa_rpa, "decode_attention", no_decode)
    monkeypatch.setattr(qsa_group, "decode_attention", no_decode)


def _step(rng, tokens, starts, contexts, page_size=128):
    """Inputs of one step: two-plus requests with prefixes, per-token tables."""
    requests = len(starts) - 1
    pages = int(-(-(max(contexts) + tokens) // page_size)) + 2
    request = np.minimum(
        np.searchsorted(starts[1:], np.arange(tokens), side="right"), requests - 1
    )
    live = np.arange(tokens) < starts[-1]
    lengths = np.where(
        live, contexts[request] + np.arange(tokens) - starts[request] + 1, 0
    ).astype(np.int32)
    owners = np.stack([rng.permutation(pages) for _ in range(requests)])
    table = owners[request].astype(np.int32)
    blocks = np.full((tokens, 64), -1, np.int32)
    for row, length in enumerate(lengths):
        if length:
            chosen = rng.permutation(length // RATIO)[:64]
            blocks[row, : len(chosen)] = chosen
    key = rng.normal(0, 0.3, (pages * page_size, 2, DIM)).astype(np.float32)
    value = rng.normal(0, 0.3, key.shape).astype(np.float32)
    cache = pack_rows(key, value, fp8=True, k_scale=0.375, v_scale=2.5).reshape(
        pages, page_size, 1, 4, DIM
    )
    hidden = jnp.asarray(rng.normal(0, 1.0, (tokens, HIDDEN)), jnp.bfloat16)
    q12_w = jnp.asarray(
        rng.normal(0, 0.1, (GROUP * HEADS * 2 * DIM, HIDDEN)), jnp.bfloat16
    )
    o12 = jnp.asarray(
        rng.normal(0, 0.05, (GROUP * HEADS * DIM, HIDDEN)), jnp.bfloat16
    )
    return dict(
        hidden=hidden,
        q12_w=q12_w,
        nq=jnp.asarray(rng.normal(0, 0.1, (DIM,)), jnp.bfloat16),
        positions=jnp.asarray(np.maximum(lengths - 1, 0), jnp.int32),
        trig=_trig(),
        blocks=jnp.asarray(blocks),
        lengths=jnp.asarray(lengths),
        table=jnp.asarray(table),
        cache=cache,
        starts=jnp.asarray(starts, jnp.int32),
        o12=o12,
    )


def _run(inp, s, **kw):
    out, _ = qsa_group.group_attention(
        inp["hidden"],
        inp["q12_w"],
        inp["nq"],
        inp["positions"],
        inp["trig"],
        inp["blocks"],
        inp["lengths"],
        inp["table"],
        inp["cache"],
        inp["starts"],
        jnp.asarray([s], jnp.int32),
        inp["o12"],
        ratio=RATIO,
        norm_eps=EPS,
        rotary_dim=ROTARY,
        k_scale=0.375,
        v_scale=2.5,
        max_query_len=4,
        **kw,
    )
    return np.asarray(out, np.float32)


def test_split_matches_unsplit_over_the_group(interpret):
    rng = np.random.default_rng(0)
    tokens = 1024
    # Three prefill chunks with prefixes; the bucket's tail is empty, so the
    # quarters split the live rows, not the padded bucket.
    starts = np.array([0, 420, 700, 900], np.int32)
    contexts = np.array([600, 100, 1500], np.int32)
    inp = _step(rng, tokens, starts, contexts)
    split = sum(_run(inp, s, min_tokens=256) for s in range(GROUP))
    unsplit = sum(_run(inp, s, min_tokens=1 << 30) for s in range(GROUP))
    live = np.asarray(inp["lengths"]) > 0
    assert float(np.abs(unsplit[live]).mean()) > 1e-3
    assert np.all(split[~live] == 0) and np.all(unsplit[~live] == 0)
    scale = float(np.abs(unsplit[live]).mean())
    assert float(np.abs(split - unsplit).max()) < 3e-2 * scale


@pytest.mark.parametrize("tokens", [4, 8, 16, 32, 64, 512])
def test_below_threshold_is_the_per_rank_path(interpret, tokens):
    """MK buckets (4..64 rows) and every step under the default 1024 tokens
    keep the rank's own 3 heads: the same attention as without the split."""
    rng = np.random.default_rng(tokens)
    starts = np.array([0, tokens // 2, tokens], np.int32)
    contexts = np.array([300, 50], np.int32)
    inp = _step(rng, tokens, starts, contexts)
    for s in range(GROUP):
        got = _run(inp, s)  # default min_tokens=1024
        q3_w = qsa_group.rank_columns(inp["q12_w"], jnp.asarray([s]), HEADS * 2 * DIM)
        q3, gate3 = qsa_group.project_query(
            inp["hidden"], q3_w, inp["nq"], inp["positions"], inp["trig"], EPS, ROTARY
        )
        attn, _ = qsa_rpa.qsa_attention(
            q3, gate3, inp["blocks"], inp["lengths"], inp["table"], inp["cache"],
            inp["starts"], kv_heads=1, ratio=RATIO, k_scale=0.375, v_scale=2.5,
            max_query_len=4,
        )
        o3 = qsa_group.rank_columns(inp["o12"], jnp.asarray([s]), HEADS * DIM)
        want = jnp.dot(
            attn.reshape(tokens, -1), o3, preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)
        np.testing.assert_array_equal(got, np.asarray(want, np.float32))


@pytest.mark.parametrize("prefix,live", [(0, 1024), (40, 900), (128, 131), (0, 3)])
def test_quarters_cover_live_prefill_rows_once(prefix, live):
    tokens = 1024
    starts = jnp.asarray([0, prefix, live] if prefix else [0, live], jnp.int32)
    seen = np.zeros(tokens, np.int32)
    for s in range(GROUP):
        ids, valid, sub = qsa_group.quarter_rows(
            starts, jnp.int32(prefix), jnp.asarray([s]), tokens // GROUP
        )
        ids, valid, sub = np.asarray(ids), np.asarray(valid), np.asarray(sub)
        np.add.at(seen, ids[valid], 1)
        assert np.all(np.diff(ids[valid]) == 1)  # contiguous
        assert np.all(np.diff(sub) >= 0) and sub[-1] == valid.sum()
    assert np.all(seen[prefix:live] == 1)
    assert np.all(seen[:prefix] == 0) and np.all(seen[live:] == 0)
