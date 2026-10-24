# SPDX-License-Identifier: Apache-2.0
"""QSA KV-group split: the attention stage (VLLM_TPU_QSA_GROUP_SPLIT).

At TP8 the 24 query heads sit 3 per rank and the 2 KV heads 1 per group of 4
ranks; the indexer is replicated, so the 4 ranks of a group read the same
selected KV rows, each for only 3 query heads. With the split, each rank of a
group takes one contiguous quarter of the step's live prefill rows and runs
the group's 12 query heads on it (12 heads on T/4 rows costs about 0.65x of 3
heads on T rows on v7, measured with the serving tiles). It applies the
group's 12-head ``o_proj`` columns to its rows and places them in an
otherwise-zero ``[T, hidden]`` partial (selects and gathers, no scatter), so
the layer's existing TP all-reduce sums exactly two contributions per row,
one per group: the full output, with no new collective. Queries are
projected in the stage, only for the rows that use them.

Steps below ``min_tokens`` and the decode prefix of mixed steps keep the
per-rank 3-head path (the rank's heads are a slice of the group's 12), whose
attention is the same computation as without the split.
"""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.qwen4_exp.qsa import gemma_norm, main_rope
from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import (
    _prefill_attention,
    decode_attention,
    qsa_attention,
)
from vllm_torchtpu.kernels.qwen4_exp.request_types import short_request_prefix_tokens

#: Ranks of one KV group (TP8 with 2 KV heads).
GROUP = 4


def rank_heads(x, group_index, heads_per_rank):
    """This rank's slice of a ``[T, group_heads, ...]`` group-head array."""
    start = group_index.reshape(()) * heads_per_rank
    return jax.lax.dynamic_slice_in_dim(x, start, heads_per_rank, axis=1)


def rank_columns(o_proj, group_index, width):
    """This rank's rows of a ``[group_heads * D, hidden]`` group ``o_proj``."""
    return jax.lax.dynamic_slice_in_dim(o_proj, group_index.reshape(()) * width, width, 0)


def quarter_rows(starts, prefix, group_index, rows):
    """Contiguous quarter of the live prefill rows for ``group_index``.

    Returns ``(ids [rows], valid [rows], sub_starts)``: token ids of the
    quarter (clipped, masked by ``valid``) and the request starts re-based to
    the quarter, so every request keeps its own contiguous rows.
    """
    s = group_index.reshape(()).astype(jnp.int32)
    live = starts[-1] - prefix
    per = (live + GROUP - 1) // GROUP
    base = prefix + s * per
    count = jnp.clip(starts[-1] - base, 0, per)
    local = jnp.arange(rows, dtype=jnp.int32)
    ids = jnp.minimum(base + local, rows * GROUP - 1)
    valid = local < count
    sub_starts = jnp.clip(starts - base, 0, count).astype(starts.dtype)
    return ids, valid, sub_starts


def project_query(hidden, weight, nq, positions, trig, norm_eps, rotary_dim):
    """``(q, gate)`` ``[rows, heads, D]`` of a ``q_proj`` weight slice
    (``[heads * 2D, hidden]``, torch layout), as ``qsa.produce`` derives them."""
    dim = nq.shape[0]
    qgate = jnp.dot(hidden, weight.T, preferred_element_type=jnp.float32)
    qgate = qgate.astype(jnp.bfloat16).reshape(len(hidden), -1, 2 * dim)
    q = main_rope(gemma_norm(qgate[..., :dim], nq, norm_eps), positions, trig, rotary_dim)
    return q, qgate[..., dim:]


def group_attention(
    hidden,
    q12_w,
    nq,
    positions,
    trig,
    blocks,
    lengths,
    table,
    cache,
    starts,
    group_index,
    o12,
    *,
    ratio,
    norm_eps,
    rotary_dim,
    k_scale=1.0,
    v_scale=1.0,
    kv_head_offset=0,
    max_query_len=1,
    min_tokens=1024,
):
    """``[T, hidden]`` partial of this rank (all-reduce it over TP) and the cache.

    ``hidden`` ``[T, H]`` is the layer's attention input and ``q12_w``
    (``[12 * 2D, H]``, torch layout) the KV group's ``q_proj`` rows: queries
    are projected here, and only for the rows that attend with them (the
    rank's 3 heads on small steps and the decode prefix, the group's 12 on
    the quarter). ``o12``: ``[12 * D, hidden]`` group ``o_proj`` (input
    major); ``group_index``: int32 ``[1]``, the rank's index in its group (an
    operand, so every rank traces the same program). ``blocks`` /
    ``lengths`` / ``table`` are the all-token selection, causal lengths and
    page-table rows, as ``qsa_attention`` takes them.
    """
    tokens = len(hidden)
    dim = nq.shape[0]
    heads = q12_w.shape[0] // (2 * dim) // GROUP
    q3_w = rank_columns(q12_w, group_index, heads * 2 * dim)
    o3 = rank_columns(o12, group_index, heads * dim)
    options = dict(kv_heads=1, ratio=ratio, k_scale=k_scale, v_scale=v_scale)
    if kv_head_offset:
        options["kv_head_offset"] = kv_head_offset
    query = dict(nq=nq, trig=trig, norm_eps=norm_eps, rotary_dim=rotary_dim)

    def project(x, w):
        return jnp.dot(x.reshape(len(x), -1), w, preferred_element_type=jnp.float32)

    requests = len(starts) - 1
    decode_only = tokens <= requests * max_query_len
    if tokens < min_tokens or decode_only or tokens % GROUP:
        q3, gate3 = project_query(hidden, q3_w, positions=positions, **query)
        out, cache = qsa_attention(
            q3, gate3, blocks, lengths, table, cache, starts,
            **options, max_query_len=max_query_len,
        )
        return project(out, o3).astype(jnp.bfloat16), cache

    rows_out = jnp.arange(tokens)
    # Decode prefix (one row per short request, times the verify tokens): the
    # rank's 3 heads on RPA, as qsa_attention routes it.
    capacity = min(tokens, requests * max_query_len)
    prefix = short_request_prefix_tokens(starts, capacity)
    first = jnp.arange(capacity)
    q3, gate3 = project_query(
        hidden[:capacity], q3_w, positions=positions[:capacity], **query
    )
    decoded, cache = decode_attention(
        q3,
        gate3,
        blocks[:capacity],
        jnp.where(first < prefix, lengths[:capacity], 0),
        table[:capacity],
        cache,
        **options,
    )
    decoded = project(decoded, o3)

    # Prefill rows: this rank's contiguous quarter, the group's 12 heads.
    rows = tokens // GROUP
    ids, valid, sub_starts = quarter_rows(starts, prefix, group_index, rows)
    sub_lengths = jnp.where(valid, lengths[ids], 0)
    q12, gate12 = project_query(hidden[ids], q12_w, positions=positions[ids], **query)
    prefilled, cache = _prefill_attention(
        q12,
        gate12,
        blocks[ids],
        sub_lengths,
        table[ids],
        cache,
        sub_starts,
        **options,
    )
    prefilled = project(prefilled, o12)

    # Assemble the partial with selects and gathers (no scatter): the quarter
    # is contiguous from ids[0], the decode prefix is the first rows.
    local = rows_out - ids[0]
    in_quarter = (local >= 0) & (local < rows)
    pick = jnp.clip(local, 0, rows - 1)
    take_prefill = in_quarter & (sub_lengths[pick] > 0)
    take_decode = (rows_out < prefix) & (lengths > 0)
    partial = jnp.where(take_prefill[:, None], prefilled[pick], 0.0)
    partial = partial + jnp.where(
        take_decode[:, None], decoded[jnp.minimum(rows_out, capacity - 1)], 0.0
    )
    return partial.astype(jnp.bfloat16), cache
