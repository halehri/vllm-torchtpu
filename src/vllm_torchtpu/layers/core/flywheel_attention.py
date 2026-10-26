# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The FlyWheel attention kernels (flywheel_tpu) behind
--attention-backend flywheel.

The host picks one kernel per step from the request lengths (decode, extend,
or dense or ragged prefill at a block span) and builds the table of page
pieces this step's K/V rows go to. On the device every layer runs the KV
writer, then the decode kernel on the leading decode rows, then a `lax.switch`
over the step's other kernels, so one executable per token bucket serves every
step. Ported from tpu-inference's flywheel_attention.py.
"""

import enum
import functools
from collections.abc import Sequence

import flywheel_tpu
import jax
import jax.numpy as jnp
import numpy as np
from flywheel_tpu import flash_attn_func, flash_attn_with_kvcache
from flywheel_tpu.flash_attn_interface import PAGE_SIZE_MULTIPLE
from flywheel_tpu.pallas.block_sizes import FWD_BLOCKS, NUM_LANES
from flywheel_tpu.pallas.kv_cache_write import write_kv_cache_pages
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

__all__ = [
    "DECODE_HEAD_DIMS",
    "PAGE_SIZE_MULTIPLE",
    "AttentionPhase",
    "StepKernel",
    "build_kv_write_slices",
    "flywheel_attention",
    "kv_write_slice_bound",
    "plan_step_kernel",
    "select_attention_phase",
    "step_kernels",
]

# Note: at a padded head_dim the decode kernel's rows come out near zero
# (tpu-inference#3837), so the backend serves only head_dims needing no padding.
DECODE_HEAD_DIMS = (128, 256)


class AttentionPhase(enum.Enum):
    """A step's attention phase. PREFILL is fresh prefill requests alone;
    EXTEND is a mixed step, or a request whose keys are partly in the pages."""

    DECODE = "decode"
    EXTEND = "extend"
    PREFILL = "prefill"


class StepKernel(enum.Enum):
    """A kernel for the rows after a step's decode rows. A ragged prefill
    kernel is its block span instead, an int."""

    DECODE = "decode"  # no rows after the decode rows
    EXTEND = "extend"  # the paged varlen kernel over cached prefixes
    DENSE = "dense"  # the dense kernel on one (1, bucket) prefill row


def kv_write_slice_bound(num_tokens: int, max_num_seqs: int, page_size: int) -> int:
    """Upper bound on the page pieces one step writes, which sizes the slice
    table. A request writing q_len tokens touches at most
    q_len // page_size + 2 pages, so the bound depends only on the token bucket
    and the request slots, never on the batch."""
    return num_tokens // page_size + 2 * max_num_seqs


def build_kv_write_slices(
    cu_q_lens: np.ndarray,  # i32[num_reqs + 1]
    kv_lens: np.ndarray,  # i32[num_reqs], includes this step's tokens
    block_table: np.ndarray,  # i32[num_reqs, pages_per_seq]
    *,
    page_size: int,
    out: np.ndarray,  # i32[3, bound], overwritten
) -> int:
    """Fill out with the slices that copy this step's new K/V rows into the
    paged cache and return their count.

    Column j < count is (cache_row_start, new_row_start, length) in rows of
    the flat (num_pages * page_size, ...) cache view, each piece inside one
    page; the remaining columns are zero.
    """
    num_reqs = kv_lens.shape[0]
    q_lens = cu_q_lens[1 : num_reqs + 1] - cu_q_lens[:num_reqs]
    kv_start = kv_lens - q_lens
    first_page = kv_start // page_size
    last_page = (kv_lens - 1) // page_size
    counts = np.where(q_lens > 0, last_page - first_page + 1, 0)
    total = int(counts.sum())
    req_idx = np.repeat(np.arange(num_reqs), counts)
    piece_start = np.repeat(np.cumsum(counts) - counts, counts)
    page_slot = first_page[req_idx] + (np.arange(total) - piece_start)
    page_start_token = page_slot * page_size
    write_start = np.maximum(kv_start[req_idx], page_start_token)
    write_end = np.minimum(kv_lens[req_idx], page_start_token + page_size)
    out[:] = 0
    out[0, :total] = block_table[req_idx, page_slot] * page_size + (
        write_start - page_start_token
    )
    out[1, :total] = cu_q_lens[:num_reqs][req_idx] + (write_start - kv_start[req_idx])
    out[2, :total] = write_end - write_start
    return total


def select_attention_phase(
    num_decode: int, cu_q_lens: np.ndarray, kv_lens: np.ndarray
) -> AttentionPhase:
    """A step's attention phase, from its host lengths: the num_reqs + 1
    query starts and num_reqs kv lengths of its requests, decode rows first."""
    q_lens = np.diff(cu_q_lens)
    if np.any(q_lens[:num_decode] != 1):
        # The decode kernel takes the leading num_decode rows as one token
        # each, so a batch that stops ordering decode requests first fails
        # here instead of attending the wrong rows.
        raise ValueError(
            "A batch must put its decode requests first, one query each; got "
            f"q_lens {q_lens[:num_decode].tolist()} for the leading {num_decode}."
        )
    if num_decode == kv_lens.size:
        return AttentionPhase.DECODE
    if num_decode > 0 or np.any(kv_lens != q_lens):
        # A mixed step is EXTEND even when its prefill requests are fresh: the
        # ragged kernel would need the decode rows merged into its output, an
        # extra [T, N, D] copy per layer.
        return AttentionPhase.EXTEND
    return AttentionPhase.PREFILL


# flash_attn_func pads the sequence axis to a multiple of NUM_LANES, then runs
# one block up to the largest of FWD_BLOCKS, else whole blocks of the largest
# FWD_BLOCKS size that divides the axis. The dense cost model below describes
# an unpadded bucket of whole min(bucket, largest) blocks only.
_DENSE_MAX_BLOCK = max(FWD_BLOCKS)

# The varlen kernel caps its q/kv block at max_seqlen_q/k, picking from
# FWD_BLOCKS, so a span is one of those block sizes. The smallest is left out:
# it saves a kernel per bucket and rarely won (tpu-inference#3837).
_VARLEN_BLOCK_SPANS = tuple(sorted(b for b in FWD_BLOCKS if b > min(FWD_BLOCKS)))

# Fixed cost of one ragged kernel step, written as the side of a square tile
# of equal cost; fitted on v6e, and it picks the fastest span on nearly every
# sampled v7x batch too (tpu-inference#3837).
_VARLEN_STEP_OVERHEAD_TOKENS = 704


def is_dense_tileable(bucket: int) -> bool:
    """Whether the dense kernel can tile one (1, bucket) row."""
    return bucket % NUM_LANES == 0 and (
        bucket <= _DENSE_MAX_BLOCK or bucket % _DENSE_MAX_BLOCK == 0
    )


def varlen_block_spans(bucket: int) -> tuple[int, ...]:
    """Every block span the ragged prefill kernel can take at this token
    bucket, the set of ragged kernels in the bucket's executable."""
    spans = tuple(s for s in _VARLEN_BLOCK_SPANS if s <= bucket)
    return spans or (bucket,)


def ragged_step_cost(scheduled_tokens: Sequence[int], bucket: int, span: int) -> int:
    """Modelled relative cost of one ragged prefill step at this block span.

    A sequence of t = ceil(length / span) blocks takes t * (t + 1) / 2 tile
    steps on or below the causal diagonal; the bucket's unfilled tail runs as
    one more sequence. A step costs its span x span tile plus a fixed
    overhead. For one 3000-token request in a 4096-token bucket:
    (21 + 6) * (512**2 + 704**2) = 20.5M at span 512, and
    (6 + 3) * (1024**2 + 704**2) = 13.9M at span 1024.
    """
    tail = bucket - sum(scheduled_tokens)
    num_blocks = [-(-n // span) for n in [*scheduled_tokens, tail] if n > 0]
    num_steps = sum(t * (t + 1) // 2 for t in num_blocks)
    return num_steps * (span * span + _VARLEN_STEP_OVERHEAD_TOKENS**2)


def varlen_block_span(scheduled_tokens: Sequence[int], bucket: int) -> int:
    """The varlen_block_spans entry with the lowest ragged_step_cost. min keeps
    the first of equal costs and the spans go widest first, so a tie goes to
    the wider span."""
    return min(
        reversed(varlen_block_spans(bucket)),
        key=lambda span: ragged_step_cost(scheduled_tokens, bucket, span),
    )


def use_dense_prefill(
    scheduled_tokens: Sequence[int], bucket: int, ragged_block_span: int
) -> bool:
    """Whether a prefill step of one request runs the dense kernel instead of
    the ragged one. The runner packs requests end to end and pads the tail, so
    one request is already a (1, bucket) row: the causal mask keeps every real
    query off the padded keys, and the KV writer copies real rows only."""
    if len(scheduled_tokens) != 1 or not is_dense_tileable(bucket):
        return False
    block = min(bucket, _DENSE_MAX_BLOCK)
    num_blocks = bucket // block
    dense_cost = num_blocks * (num_blocks + 1) // 2 * block * block
    return dense_cost <= ragged_step_cost(scheduled_tokens, bucket, ragged_block_span)


def step_kernels(bucket: int) -> tuple[StepKernel | int, ...]:
    """The kernels one executable holds at this token bucket for a step's rows
    after its decode rows, in `lax.switch` order."""
    kernels: list[StepKernel | int] = [StepKernel.DECODE, StepKernel.EXTEND]
    if is_dense_tileable(bucket):
        kernels.append(StepKernel.DENSE)
    kernels.extend(varlen_block_spans(bucket))
    return tuple(kernels)


def plan_step_kernel(
    num_decode: int, cu_q_lens: np.ndarray, kv_lens: np.ndarray, bucket: int
) -> int:
    """The step's index into step_kernels(bucket), from its host lengths."""
    phase = select_attention_phase(num_decode, cu_q_lens, kv_lens)
    kernel: StepKernel | int
    if phase is AttentionPhase.DECODE:
        kernel = StepKernel.DECODE
    elif phase is AttentionPhase.EXTEND:
        kernel = StepKernel.EXTEND
    else:
        scheduled_tokens = np.diff(cu_q_lens).tolist()
        span = varlen_block_span(scheduled_tokens, bucket)
        kernel = (
            StepKernel.DENSE
            if use_dense_prefill(scheduled_tokens, bucket, span)
            else span
        )
    return step_kernels(bucket).index(kernel)


# Note: jit here so that every layer calls one shared traced kernel, as
# flash_attn_with_kvcache and the KV writer already are; inlined, every layer
# would lower its own copy.
_flash_attn_varlen_func = jax.jit(
    flywheel_tpu.flash_attn_varlen_func,
    static_argnames=(
        "max_seqlen_q",
        "max_seqlen_k",
        "softmax_scale",
        "causal",
        "interpret",
        "token_major",
    ),
)


def _decode_rows(
    q: jax.Array,  # [T, N, D]
    k: jax.Array,  # [T, H, D]
    v: jax.Array,  # [T, H, D]
    kv_cache: jax.Array,  # [num_pages, page_size, 2 * H, D]
    kv_lens: jax.Array,  # i32[max_num_seqs], includes this step's token
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    num_decode: jax.Array,
    *,
    sm_scale: float,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    """The decode kernel: append and attend one query token per row, for the
    leading min(max_num_seqs, T) rows; returns those rows (not padded back to
    T) and the cache."""
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    batch = min(max_num_seqs, q.shape[0])
    block_table = page_indices[: batch * pages_per_seq].reshape(batch, pages_per_seq)
    # kv_lens count this step's token, so the in-kernel append takes the
    # length before it; num_active stops the kernel at the real requests.
    output, cache = flash_attn_with_kvcache(
        q[:batch, None],
        kv_cache,
        None,
        k[:batch, None],
        v[:batch, None],
        cache_seqlens=kv_lens[:batch] - 1,
        block_table=block_table,
        num_active=num_decode,
        softmax_scale=sm_scale,
        interpret=interpret,
    )
    return output[:, 0], cache


def _extend_rows(
    q: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_decode: jax.Array,
    num_active: jax.Array,
    *,
    sm_scale: float,
    interpret: bool,
) -> jax.Array:
    """The paged varlen kernel over the requests after the leading decode
    rows, reading their cached prefixes through the block table. Rows it does
    not own are unspecified; the caller merges them."""
    page_size = kv_cache.shape[1]
    max_num_seqs = kv_lens.shape[0]
    block_table = page_indices.reshape(max_num_seqs, -1)
    pages_per_seq = block_table.shape[1]
    # The window slides past the decode requests: a request clamped to zero
    # tokens in place still costs a kernel step. Each array is padded by its
    # own length to keep dynamic_slice's start in range.
    cu_q_window = lax.dynamic_slice(
        jnp.pad(cu_q_lens, (0, max_num_seqs)), (num_decode,), (max_num_seqs + 1,)
    )
    kv_lens_window = lax.dynamic_slice(
        jnp.pad(kv_lens, (0, max_num_seqs)), (num_decode,), (max_num_seqs,)
    )
    block_table_window = lax.dynamic_slice(
        jnp.pad(block_table, ((0, max_num_seqs), (0, 0))),
        (num_decode, 0),
        (max_num_seqs, pages_per_seq),
    )
    # max_seqlen_q/k are static block-size bounds (the token bucket and the
    # table's capacity), so one executable serves every kv_len of a bucket.
    return _flash_attn_varlen_func(
        q.transpose(1, 0, 2),
        kv_cache,
        None,
        cu_q_window,
        None,
        max_seqlen_q=q.shape[0],
        max_seqlen_k=pages_per_seq * page_size,
        softmax_scale=sm_scale,
        causal=True,
        token_major=False,
        block_table=block_table_window,
        seqused_k=kv_lens_window,
        num_active=num_active - num_decode,
        interpret=interpret,
    ).transpose(1, 0, 2)


def _dense_prefill_rows(
    q: jax.Array, k: jax.Array, v: jax.Array, *, sm_scale: float, interpret: bool
) -> jax.Array:
    """The dense kernel on the step's one (1, T) row of this step's K/V."""
    num_tokens, num_heads, head_dim = q.shape

    def heads_outer(x: jax.Array) -> jax.Array:
        # [T, H, D] -> (H, 1, T, D). XLA lays the projections out
        # heads-outermost, so this transpose compiles to a bitcast.
        return x.transpose(1, 0, 2)[:, None]

    output = flash_attn_func(
        heads_outer(q),
        heads_outer(k),
        heads_outer(v),
        causal=True,
        softmax_scale=sm_scale,
        token_major=False,
        heads_outer_fold=True,
        interpret=interpret,
    )
    # A tuple only comes back when the softmax statistics are requested.
    if isinstance(output, tuple):
        raise TypeError("flash_attn_func returned softmax statistics")
    return output.reshape(num_heads, num_tokens, head_dim).transpose(1, 0, 2)


def _ragged_prefill_rows(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cu_q_lens: jax.Array,
    *,
    span: int,
    sm_scale: float,
    interpret: bool,
) -> jax.Array:
    """The ragged prefill kernel at block span `span` on this step's K/V."""
    return _flash_attn_varlen_func(
        q.transpose(1, 0, 2),
        k.transpose(1, 0, 2),
        v.transpose(1, 0, 2),
        cu_q_lens,
        cu_q_lens,
        max_seqlen_q=span,
        max_seqlen_k=span,
        causal=True,
        softmax_scale=sm_scale,
        token_major=False,
        interpret=interpret,
    ).transpose(1, 0, 2)


def _step_kernel_rows(
    kernel: StepKernel | int,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    num_decode: jax.Array,
    num_active: jax.Array,
    *,
    sm_scale: float,
    interpret: bool,
) -> jax.Array:
    """The rows after the decode rows, by one step_kernels entry. `kernel` is
    a Python value, so each `lax.switch` branch traces only its own kernel."""
    if kernel is StepKernel.DECODE:
        return jnp.zeros_like(q)
    if kernel is StepKernel.EXTEND:
        return _extend_rows(
            q,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            num_decode,
            num_active,
            sm_scale=sm_scale,
            interpret=interpret,
        )
    if kernel is StepKernel.DENSE:
        return _dense_prefill_rows(q, k, v, sm_scale=sm_scale, interpret=interpret)
    return _ragged_prefill_rows(
        q, k, v, cu_q_lens, span=kernel, sm_scale=sm_scale, interpret=interpret
    )


def _flywheel_attention_shard(
    q: jax.Array,  # [T, N, D]
    k: jax.Array,  # [T, H, D]
    v: jax.Array,  # [T, H, D]
    kv_cache: jax.Array,  # [num_pages, page_size, H, 2, D]: RPA v3's bf16 pages
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    request_distribution: jax.Array,  # i32[3]
    kv_write_slices: jax.Array,  # i32[3 * bound]
    kv_write_num_slices: jax.Array,  # i32[1]
    kernel_index: jax.Array,  # i32[1]: into step_kernels(T)
    *,
    sm_scale: float,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    """One shard's attention for any step, in one executable per bucket.

    The DMA writer copies the K/V of the rows after the decode rows into the
    pages and the decode kernel appends and attends the decode rows; then
    `lax.switch` runs the step's kernel on the other rows. No switch branch
    writes the cache: a branch that did would make XLA copy it per layer.

    request_distribution is RPA's (decode_end, prefill_end, mixed_end); the
    runner puts no request in the prefill-only group, so only entries 0 and 2
    are read.
    """
    pool_shape = kv_cache.shape
    if (
        kv_cache.ndim != 5
        or kv_cache.dtype != jnp.bfloat16
        or pool_shape[3:]
        != (
            2,
            q.shape[-1],
        )
    ):
        raise NotImplementedError(
            "The flywheel attention backend runs on the layer's own bf16 pages "
            "[pages, page_size, kv_heads, 2, head_dim], not a shared KV pool; "
            f"got a {pool_shape} {kv_cache.dtype} KV cache for head_dim "
            f"{q.shape[-1]}."
        )
    num_pages, page_size, num_kv_heads, kv_packing, head_dim = pool_shape
    # RPA v3 interleaves each KV head's K and V rows, [k0, v0, k1, v1, ...], in
    # (head, 2) pairs: the same bytes as FlyWheel's merged (2 * H) row axis.
    cache = kv_cache.reshape(num_pages, page_size, num_kv_heads * kv_packing, head_dim)
    num_decode = request_distribution[0]
    num_active = request_distribution[2]
    cache = write_kv_cache_pages(
        k,
        v,
        cache,
        kv_write_slices.reshape(3, -1),
        kv_write_num_slices,
        interpret=interpret,
    )
    decode_output, cache = _decode_rows(
        q,
        k,
        v,
        cache,
        kv_lens,
        page_indices,
        num_decode,
        sm_scale=sm_scale,
        interpret=interpret,
    )
    branches = [
        functools.partial(
            _step_kernel_rows, kernel, sm_scale=sm_scale, interpret=interpret
        )
        for kernel in step_kernels(q.shape[0])
    ]
    output = lax.switch(
        kernel_index[0],
        branches,
        q,
        k,
        v,
        cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_decode,
        num_active,
    )
    # The leading rows take the decode kernel's output below num_decode and
    # zeros past the scheduled tokens: a branch need not own the rows below
    # its first request.
    num_leading_rows = min(-(-decode_output.shape[0] // 8) * 8, q.shape[0])
    padded_decode_output = jnp.pad(
        decode_output,
        ((0, num_leading_rows - decode_output.shape[0]), (0, 0), (0, 0)),
    )
    row_index = jnp.arange(num_leading_rows)[:, None, None]
    num_scheduled_tokens = cu_q_lens[num_active]
    leading_output = jnp.where(
        row_index < num_decode,
        padded_decode_output,
        jnp.where(row_index < num_scheduled_tokens, output[:num_leading_rows], 0),
    )
    output = output.at[:num_leading_rows].set(leading_output)
    return cache.reshape(pool_shape), output


def flywheel_attention(
    kv_cache: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    request_distribution: jax.Array,
    kv_write_slices: jax.Array,
    kv_write_num_slices: jax.Array,
    kernel_index: jax.Array,
    *,
    mesh: Mesh,
    sm_scale: float,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """(kv_cache, output) of one attention layer, sharded along the KV heads
    over the mesh's one axis, as sharded_ragged_paged_attention shards RPA."""
    (heads_axis,) = mesh.axis_names
    num_shards = mesh.shape[heads_axis]
    if k.shape[1] % num_shards:
        raise NotImplementedError(
            f"The flywheel attention backend shards {k.shape[1]} KV heads over "
            f"{num_shards} devices; use a tensor parallel size dividing them."
        )
    qkv_spec = P(None, heads_axis, None)
    kv_cache_spec = P(None, None, heads_axis, None, None)
    in_specs = (qkv_spec, qkv_spec, qkv_spec, kv_cache_spec) + (P(None),) * 7
    shard = functools.partial(
        _flywheel_attention_shard, sm_scale=sm_scale, interpret=interpret
    )
    new_kv_cache, output = jax.shard_map(
        shard,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=(kv_cache_spec, qkv_spec),
        check_vma=False,
    )(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        request_distribution,
        kv_write_slices,
        kv_write_num_slices,
        kernel_index,
    )
    return new_kv_cache, output
