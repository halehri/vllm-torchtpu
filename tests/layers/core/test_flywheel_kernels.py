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
"""flywheel_attention: the host's per-step kernel choice and KV write table,
and the sharded kernel call against a numpy reference in Pallas interpret
mode (CPU)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.core import flywheel_attention as fa


def naive_kv_write_slices(cu_q_lens, kv_lens, block_table, page_size):
    """Token by token: (cache_row, new_row, length), merging runs that stay in
    one page and are contiguous on both sides."""
    pieces = []
    for req, kv_len in enumerate(kv_lens):
        q_len = cu_q_lens[req + 1] - cu_q_lens[req]
        for t in range(q_len):
            pos = kv_len - q_len + t
            cache_row = block_table[req, pos // page_size] * page_size + pos % page_size
            new_row = cu_q_lens[req] + t
            if (
                pieces
                and t > 0
                and pos % page_size != 0
                and pieces[-1][0] + pieces[-1][2] == cache_row
                and pieces[-1][1] + pieces[-1][2] == new_row
            ):
                pieces[-1][2] += 1
            else:
                pieces.append([cache_row, new_row, 1])
    return pieces


@pytest.mark.parametrize("page_size", [128, 256])
@pytest.mark.parametrize("seed", range(5))
def test_kv_write_slices_match_a_token_by_token_reference(page_size, seed):
    rng = np.random.default_rng(seed)
    num_reqs, pages_per_seq = 6, 8
    q_lens = rng.integers(0, 3 * page_size, num_reqs)
    q_lens[:2] = 1
    kv_lens = q_lens + rng.integers(0, 4 * page_size, num_reqs)
    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    block_table = (
        rng.permutation(num_reqs * pages_per_seq)
        .reshape(num_reqs, pages_per_seq)
        .astype(np.int32)
    )
    bound = fa.kv_write_slice_bound(int(cu_q_lens[-1]), num_reqs, page_size)
    out = np.full((3, bound), -1, np.int32)

    count = fa.build_kv_write_slices(
        cu_q_lens, kv_lens.astype(np.int32), block_table, page_size=page_size, out=out
    )

    expected = naive_kv_write_slices(
        cu_q_lens.tolist(), kv_lens.tolist(), block_table, page_size
    )
    assert count == len(expected) <= bound
    np.testing.assert_array_equal(out[:, :count].T, np.array(expected))
    np.testing.assert_array_equal(out[:, count:], 0)


@pytest.mark.parametrize(
    ("num_decode", "seq_lens", "phase"),
    [
        (2, [(1, 300), (1, 129)], fa.AttentionPhase.DECODE),
        (0, [(300, 300), (129, 129)], fa.AttentionPhase.PREFILL),
        (1, [(1, 300), (129, 129)], fa.AttentionPhase.EXTEND),
        (0, [(200, 457)], fa.AttentionPhase.EXTEND),
    ],
)
def test_attention_phase(num_decode, seq_lens, phase):
    q_lens = [q_len for q_len, _ in seq_lens]
    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)])
    kv_lens = np.array([kv_len for _, kv_len in seq_lens])

    assert fa.select_attention_phase(num_decode, cu_q_lens, kv_lens) is phase


def test_attention_phase_needs_decode_rows_first():
    with pytest.raises(ValueError, match="decode requests first"):
        fa.select_attention_phase(1, np.array([0, 5, 6]), np.array([5, 9]))


@pytest.mark.parametrize(
    ("scheduled_tokens", "bucket", "dense"),
    [
        ([1000], 1024, True),
        ([100], 1024, True),
        ([500, 500], 1024, False),
        # The dense kernel cannot tile these buckets.
        ([64], 64, False),
        ([3000], 3072, False),
    ],
)
def test_dense_prefill_for_one_request(scheduled_tokens, bucket, dense):
    span = fa.varlen_block_span(scheduled_tokens, bucket)

    assert fa.use_dense_prefill(scheduled_tokens, bucket, span) is dense


@pytest.mark.parametrize(
    ("scheduled_tokens", "bucket", "span"),
    [([128] * 16, 2048, 256), ([2048], 2048, 1024), ([100], 128, 128)],
)
def test_varlen_block_span(scheduled_tokens, bucket, span):
    assert fa.varlen_block_span(scheduled_tokens, bucket) == span
    assert span in fa.varlen_block_spans(bucket)


def test_varlen_block_span_tie_goes_to_the_wider_span(monkeypatch):
    monkeypatch.setattr(fa, "ragged_step_cost", lambda *args: 1)

    assert fa.varlen_block_span([1000], 2048) == 2048


@pytest.mark.parametrize(
    ("num_decode", "q_lens", "kv_lens", "bucket", "kernel"),
    [
        (2, [1, 1], [10, 300], 128, fa.StepKernel.DECODE),
        (1, [1, 40], [10, 300], 128, fa.StepKernel.EXTEND),
        (0, [200], [200], 256, fa.StepKernel.DENSE),
        (0, [100, 100], [100, 100], 256, 256),
    ],
)
def test_plan_step_kernel(num_decode, q_lens, kv_lens, bucket, kernel):
    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)

    index = fa.plan_step_kernel(
        num_decode, cu_q_lens, np.array(kv_lens, np.int32), bucket
    )

    assert fa.step_kernels(bucket)[index] == kernel


# --------------------------------------------------------------------------
# The sharded kernel call in Pallas interpret mode, against numpy.
# --------------------------------------------------------------------------

PAGE_SIZE = 128
HEAD_DIM = 128
NUM_HEADS, NUM_KV_HEADS = 4, 2
MAX_NUM_SEQS, PAGES_PER_SEQ = 4, 4


def reference_attention(q, keys, values, q_len, sm_scale):
    """Causal attention of a request's q_len last positions over all its
    keys, with KV head n // (N / H) per query head n."""
    group = NUM_HEADS // NUM_KV_HEADS
    k = np.repeat(keys, group, axis=1)  # [kv_len, N, D]
    v = np.repeat(values, group, axis=1)
    kv_len = keys.shape[0]
    scores = np.einsum("tnd,snd->nts", q, k) * sm_scale
    q_pos = np.arange(kv_len - q_len, kv_len)[:, None]
    scores = np.where(np.arange(kv_len)[None, :] <= q_pos, scores, -np.inf)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    return np.einsum("nts,snd->tnd", probs, v)


def _bf16(x):
    return np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32))


def _call(
    cache, q, k, v, kv_lens, block_table, cu_q_lens, num_decode, num_reqs, *, interpret
):
    """One flywheel_attention call, its host metadata planned as the runner's
    prepare_step plans it. Returns (new_cache, output) as float32 numpy."""
    bucket = q.shape[0]
    host_cu, host_kv = cu_q_lens[: num_reqs + 1], kv_lens[:num_reqs]
    kernel_index = fa.plan_step_kernel(num_decode, host_cu, host_kv, bucket)
    bound = fa.kv_write_slice_bound(bucket, MAX_NUM_SEQS, PAGE_SIZE)
    slices = np.zeros((3, bound), np.int32)
    num_slices = fa.build_kv_write_slices(
        host_cu[num_decode:],
        host_kv[num_decode:],
        block_table[num_decode:num_reqs],
        page_size=PAGE_SIZE,
        out=slices,
    )
    # Another axis name than the runner's, so nothing depends on it.
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("heads",))
    new_cache, output = fa.flywheel_attention(
        jnp.asarray(cache, jnp.bfloat16),
        jnp.asarray(q, jnp.bfloat16),
        jnp.asarray(k, jnp.bfloat16),
        jnp.asarray(v, jnp.bfloat16),
        jnp.asarray(kv_lens),
        jnp.asarray(block_table.reshape(-1)),
        jnp.asarray(cu_q_lens),
        jnp.asarray([num_decode, num_decode, num_reqs], jnp.int32),
        jnp.asarray(slices.reshape(-1)),
        jnp.asarray([num_slices], jnp.int32),
        jnp.asarray([kernel_index], jnp.int32),
        mesh=mesh,
        sm_scale=HEAD_DIM**-0.5,
        interpret=interpret,
    )
    return (
        np.asarray(new_cache.astype(jnp.float32)),
        np.asarray(output.astype(jnp.float32)),
    )


class _Requests:
    """Random bf16 K/V/Q streams of a few requests on their own pages."""

    def __init__(self, kv_lens, seed=0):
        rng = np.random.default_rng(seed)
        self.num_reqs = len(kv_lens)
        self.block_table = np.zeros((MAX_NUM_SEQS, PAGES_PER_SEQ), np.int32)
        self.block_table[: self.num_reqs] = 1 + rng.permutation(
            self.num_reqs * PAGES_PER_SEQ
        ).reshape(self.num_reqs, PAGES_PER_SEQ)
        shape = (MAX_NUM_SEQS * PAGES_PER_SEQ + 1, PAGE_SIZE, NUM_KV_HEADS, 2, HEAD_DIM)
        self.empty_cache = np.zeros(shape, np.float32)  # RPA v3's bf16 pages
        self.keys, self.values, self.queries = [], [], []
        for kv_len in kv_lens:
            self.keys.append(
                _bf16(rng.standard_normal((kv_len, NUM_KV_HEADS, HEAD_DIM)))
            )
            self.values.append(
                _bf16(rng.standard_normal((kv_len, NUM_KV_HEADS, HEAD_DIM)))
            )
            self.queries.append(
                _bf16(rng.standard_normal((kv_len, NUM_HEADS, HEAD_DIM)))
            )

    def cache_with(self, lengths):
        """The cache holding each request's first lengths[req] tokens."""
        cache = self.empty_cache.copy()
        for req, n in enumerate(lengths):
            for pos in range(n):
                page = self.block_table[req, pos // PAGE_SIZE]
                cache[page, pos % PAGE_SIZE, :, 0] = self.keys[req][pos]
                cache[page, pos % PAGE_SIZE, :, 1] = self.values[req][pos]
        return cache

    def step(self, steps, bucket):
        """The (q, k, v, kv_lens, cu_q_lens, reference) of steps = [(q_len,
        kv_len)], one per request in order."""
        q = np.zeros((bucket, NUM_HEADS, HEAD_DIM), np.float32)
        k = np.zeros((bucket, NUM_KV_HEADS, HEAD_DIM), np.float32)
        v = np.zeros_like(k)
        cu_q_lens = np.zeros(MAX_NUM_SEQS + 1, np.int32)
        kv_lens = np.zeros(MAX_NUM_SEQS, np.int32)
        reference = np.zeros_like(q)
        for req, (q_len, kv_len) in enumerate(steps):
            start = cu_q_lens[req]
            cu_q_lens[req + 1 :] = start + q_len
            kv_lens[req] = kv_len
            new = slice(kv_len - q_len, kv_len)
            q[start : start + q_len] = self.queries[req][new]
            k[start : start + q_len] = self.keys[req][new]
            v[start : start + q_len] = self.values[req][new]
            reference[start : start + q_len] = reference_attention(
                self.queries[req][new],
                self.keys[req][:kv_len],
                self.values[req][:kv_len],
                q_len,
                HEAD_DIM**-0.5,
            )
        return q, k, v, kv_lens, cu_q_lens, reference


def run_step(steps, num_decode, bucket, *, interpret=True):
    """One step of steps = [(q_len, kv_len)], decode requests first, on a
    cache already holding each request's first kv_len - q_len tokens. Returns
    (output, reference output, the cache, the expected cache), float32."""
    reqs = _Requests([kv_len for _, kv_len in steps])
    q, k, v, kv_lens, cu_q_lens, reference = reqs.step(steps, bucket)
    cache = reqs.cache_with([kv_len - q_len for q_len, kv_len in steps])
    new_cache, output = _call(
        cache,
        q,
        k,
        v,
        kv_lens,
        reqs.block_table,
        cu_q_lens,
        num_decode,
        reqs.num_reqs,
        interpret=interpret,
    )
    num_tokens = int(cu_q_lens[reqs.num_reqs])
    expected_cache = reqs.cache_with([kv_len for _, kv_len in steps])
    return output[:num_tokens], reference[:num_tokens], new_cache, expected_cache


_STEPS = [
    pytest.param([(1, 1), (1, 130), (1, 257)], 3, 128, id="decode"),
    pytest.param([(200, 200)], 0, 256, id="dense_prefill"),
    pytest.param([(100, 100), (60, 60)], 0, 256, id="ragged_prefill"),
    pytest.param([(1, 140), (70, 300), (5, 5)], 1, 128, id="extend"),
]


def run_prefill_then_decode(*, interpret, num_decode_steps=3):
    """Fresh prefills, then decode steps, each on the cache the previous step
    returned: the hand-off a serving engine depends on."""
    prompt_lens = [150, 90]
    reqs = _Requests([n + num_decode_steps for n in prompt_lens])
    q, k, v, kv_lens, cu_q_lens, reference = reqs.step(
        [(n, n) for n in prompt_lens], 256
    )
    cache, output = _call(
        reqs.empty_cache,
        q,
        k,
        v,
        kv_lens,
        reqs.block_table,
        cu_q_lens,
        0,
        reqs.num_reqs,
        interpret=interpret,
    )
    outputs = [(output[: sum(prompt_lens)], reference[: sum(prompt_lens)])]
    for i in range(1, num_decode_steps + 1):
        q, k, v, kv_lens, cu_q_lens, reference = reqs.step(
            [(1, n + i) for n in prompt_lens], 16
        )
        cache, output = _call(
            cache,
            q,
            k,
            v,
            kv_lens,
            reqs.block_table,
            cu_q_lens,
            reqs.num_reqs,
            reqs.num_reqs,
            interpret=interpret,
        )
        outputs.append((output[: reqs.num_reqs], reference[: reqs.num_reqs]))
    expected_cache = reqs.cache_with([n + num_decode_steps for n in prompt_lens])
    return outputs, cache, expected_cache


@pytest.fixture
def cpu():
    # Pallas interpret mode does not lower on TPU-placed arrays.
    if jax.default_backend() != "cpu":
        pytest.skip("interpret mode runs on CPU")


@pytest.mark.parametrize(("steps", "num_decode", "bucket"), _STEPS)
def test_flywheel_attention_matches_reference(cpu, steps, num_decode, bucket):
    output, reference, cache, expected_cache = run_step(steps, num_decode, bucket)

    np.testing.assert_allclose(output, reference, atol=3e-2, rtol=3e-2)
    np.testing.assert_array_equal(cache, expected_cache)


def test_decode_reads_what_the_prefill_wrote(cpu):
    outputs, cache, expected_cache = run_prefill_then_decode(interpret=True)

    for i, (output, reference) in enumerate(outputs):
        np.testing.assert_allclose(
            output, reference, atol=3e-2, rtol=3e-2, err_msg=f"step {i}"
        )
    np.testing.assert_array_equal(cache, expected_cache)


@pytest.fixture
def tpu():
    # In a fixture, not at import: collection must not claim the TPU.
    if jax.default_backend() != "tpu":
        pytest.skip("needs a TPU")


@pytest.mark.tpu_test
@pytest.mark.parametrize(("steps", "num_decode", "bucket"), _STEPS)
def test_flywheel_attention_matches_reference_on_tpu(tpu, steps, num_decode, bucket):
    output, reference, cache, expected_cache = run_step(
        steps, num_decode, bucket, interpret=False
    )

    np.testing.assert_allclose(output, reference, atol=3e-2, rtol=3e-2)
    np.testing.assert_array_equal(cache, expected_cache)


@pytest.mark.tpu_test
def test_decode_reads_what_the_prefill_wrote_on_tpu(tpu):
    outputs, cache, expected_cache = run_prefill_then_decode(interpret=False)

    for i, (output, reference) in enumerate(outputs):
        np.testing.assert_allclose(
            output, reference, atol=3e-2, rtol=3e-2, err_msg=f"step {i}"
        )
    np.testing.assert_array_equal(cache, expected_cache)


# The first steps a Qwen3-0.6B engine (16 query / 8 KV heads) served at
# max_num_batched_tokens=256, max_num_seqs=8: (num_tokens, num_decode,
# [(request, q_len, kv_len)]), with request r on pages ENGINE_PAGES[r].
ENGINE_PAGES = {
    0: [1],
    1: [2],
    2: [3],
    3: [4],
    4: [5],
    5: [6, 7],
    6: [8, 9, 10, 11],
    7: [12],
}
ENGINE_STEPS = [
    (
        256,
        0,
        [
            (0, 5, 5),
            (1, 8, 8),
            (2, 10, 10),
            (3, 11, 11),
            (4, 4, 4),
            (5, 210, 210),
            (6, 8, 8),
        ],
    ),
    (
        256,
        6,
        [
            (0, 1, 6),
            (1, 1, 9),
            (2, 1, 11),
            (3, 1, 12),
            (4, 1, 5),
            (5, 1, 211),
            (6, 250, 258),
        ],
    ),
    (
        256,
        6,
        [
            (0, 1, 7),
            (1, 1, 10),
            (2, 1, 12),
            (3, 1, 13),
            (4, 1, 6),
            (5, 1, 212),
            (6, 202, 460),
            (7, 14, 14),
        ],
    ),
    (
        16,
        8,
        [
            (0, 1, 8),
            (1, 1, 11),
            (2, 1, 13),
            (3, 1, 14),
            (4, 1, 7),
            (5, 1, 213),
            (6, 1, 461),
            (7, 1, 15),
        ],
    ),
    (
        16,
        8,
        [
            (0, 1, 9),
            (1, 1, 12),
            (2, 1, 14),
            (3, 1, 15),
            (4, 1, 8),
            (5, 1, 214),
            (6, 1, 462),
            (7, 1, 16),
        ],
    ),
]


def replay_engine_steps(*, interpret):
    """ENGINE_STEPS at the engine's geometry, each on the previous step's
    cache. Returns [(step, output rows, reference rows)]."""
    num_heads, num_kv_heads, max_num_seqs, pages_per_seq = 16, 8, 8, 16
    rng = np.random.default_rng(7)
    streams = {}
    for r in ENGINE_PAGES:
        n = len(ENGINE_PAGES[r]) * PAGE_SIZE
        streams[r] = tuple(
            _bf16(rng.standard_normal((n, heads, HEAD_DIM)))
            for heads in (num_kv_heads, num_kv_heads, num_heads)
        )
    cache = np.zeros((32, PAGE_SIZE, num_kv_heads, 2, HEAD_DIM), np.float32)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("heads",))
    group = num_heads // num_kv_heads
    results = []
    for i, (bucket, num_decode, rows) in enumerate(ENGINE_STEPS):
        q = np.zeros((bucket, num_heads, HEAD_DIM), np.float32)
        k = np.zeros((bucket, num_kv_heads, HEAD_DIM), np.float32)
        v = np.zeros_like(k)
        reference = np.zeros_like(q)
        cu_q_lens = np.zeros(max_num_seqs + 1, np.int32)
        kv_lens = np.zeros(max_num_seqs, np.int32)
        block_table = np.zeros((max_num_seqs, pages_per_seq), np.int32)
        for row, (r, q_len, kv_len) in enumerate(rows):
            start = cu_q_lens[row]
            cu_q_lens[row + 1 :] = start + q_len
            kv_lens[row] = kv_len
            block_table[row, : len(ENGINE_PAGES[r])] = ENGINE_PAGES[r]
            keys, values, queries = streams[r]
            new = slice(kv_len - q_len, kv_len)
            q[start : start + q_len] = queries[new]
            k[start : start + q_len] = keys[new]
            v[start : start + q_len] = values[new]
            kk = np.repeat(keys[:kv_len], group, axis=1)
            vv = np.repeat(values[:kv_len], group, axis=1)
            scores = np.einsum("tnd,snd->nts", queries[new], kk) * HEAD_DIM**-0.5
            q_pos = np.arange(kv_len - q_len, kv_len)[:, None]
            scores = np.where(np.arange(kv_len)[None, :] <= q_pos, scores, -np.inf)
            probs = np.exp(scores - scores.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            reference[start : start + q_len] = np.einsum("nts,snd->tnd", probs, vv)
        num_reqs = len(rows)
        host_cu, host_kv = cu_q_lens[: num_reqs + 1], kv_lens[:num_reqs]
        kernel_index = fa.plan_step_kernel(num_decode, host_cu, host_kv, bucket)
        slices = np.zeros(
            (3, fa.kv_write_slice_bound(bucket, max_num_seqs, PAGE_SIZE)), np.int32
        )
        num_slices = fa.build_kv_write_slices(
            host_cu[num_decode:],
            host_kv[num_decode:],
            block_table[num_decode:num_reqs],
            page_size=PAGE_SIZE,
            out=slices,
        )
        new_cache, output = fa.flywheel_attention(
            jnp.asarray(cache, jnp.bfloat16),
            jnp.asarray(q, jnp.bfloat16),
            jnp.asarray(k, jnp.bfloat16),
            jnp.asarray(v, jnp.bfloat16),
            jnp.asarray(kv_lens),
            jnp.asarray(block_table.reshape(-1)),
            jnp.asarray(cu_q_lens),
            jnp.asarray([num_decode, num_decode, num_reqs], jnp.int32),
            jnp.asarray(slices.reshape(-1)),
            jnp.asarray([num_slices], jnp.int32),
            jnp.asarray([kernel_index], jnp.int32),
            mesh=mesh,
            sm_scale=HEAD_DIM**-0.5,
            interpret=interpret,
        )
        cache = np.asarray(new_cache.astype(jnp.float32))
        num_tokens = int(cu_q_lens[num_reqs])
        results.append(
            (
                i,
                np.asarray(output.astype(jnp.float32))[:num_tokens],
                reference[:num_tokens],
            )
        )
    return results


@pytest.mark.tpu_test
def test_engine_steps_on_tpu(tpu):
    for i, output, reference in replay_engine_steps(interpret=False):
        bad = np.nonzero(
            ~np.isclose(output, reference, atol=3e-2, rtol=3e-2).all(axis=(1, 2))
        )[0]
        assert not bad.size, f"step {i}: wrong rows {bad.tolist()}"
