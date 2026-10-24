# SPDX-License-Identifier: Apache-2.0
"""Per-sequence GDN metadata inputs rolled once per step are bitwise neutral.

`VLLM_TPU_GDN_HOIST_PER_SEQ_ROLLS` rolls query_start_loc / seq_lens /
state_indices / read_offsets (and the checkpoint indices) to the first prefill
sequence in the step graph; the GDN v3 kernel must produce the same outputs
and states as when its per-sequence metadata rolls them itself. CPU interpret
mode.
"""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import wrapper
from vllm_torchtpu.layers.adapter.custom_ops import gdn_attention_op

N_KQ, N_V, D_K, D_V, KERNEL = 1, 2, 128, 128, 4
DIM = 2 * N_KQ * D_K + N_V * D_V
NUM_DECODE, NUM_SEQS = 4, 6


def _inputs():
    rngs = iter(jax.random.split(jax.random.key(0), 9))
    return dict(
        qkv=jax.random.normal(next(rngs), (NUM_SEQS, DIM), dtype=jnp.bfloat16),
        b=jax.random.normal(next(rngs), (NUM_SEQS, N_V), dtype=jnp.bfloat16),
        a=jax.random.normal(next(rngs), (NUM_SEQS, N_V), dtype=jnp.bfloat16),
        conv_weight=jax.random.normal(next(rngs), (DIM, 1, KERNEL), dtype=jnp.bfloat16),
        conv_bias=jax.random.normal(next(rngs), (DIM,), dtype=jnp.bfloat16),
        a_log=jax.random.normal(next(rngs), (N_V,), dtype=jnp.float32),
        dt_bias=jax.random.normal(next(rngs), (N_V,), dtype=jnp.float32),
        conv_state=jax.random.normal(
            next(rngs), (NUM_SEQS, KERNEL - 1, DIM), dtype=jnp.bfloat16
        ),
        recurrent_state=jax.random.normal(
            next(rngs), (NUM_SEQS, N_V, D_K, D_V), dtype=jnp.float32
        ),
        query_start_loc=jnp.arange(NUM_SEQS + 1, dtype=jnp.int32),
        # A non-identity slot map, so a wrong roll lands on another slot.
        state_indices=jnp.array([3, 0, 5, 1, 4, 2], dtype=jnp.int32),
        read_offsets=jnp.zeros((NUM_SEQS,), dtype=jnp.int32),
        distribution=jnp.array([NUM_DECODE, NUM_SEQS, NUM_SEQS], dtype=jnp.int32),
        seq_lens=jnp.array([11, 7, 9, 12, 5, 8], dtype=jnp.int32),
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D_K,
        d_v=D_V,
        kernel_size=KERNEL,
    )


def _rolled(inputs, shift):
    parts = ("query_start_loc", "seq_lens", "state_indices", "read_offsets")
    return jnp.concatenate([jnp.roll(inputs[k], -shift) for k in parts])


def _run(inputs, per_seq_rolled=None):
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    pltpu.reset_tpu_interpret_mode_state()
    try:
        with (
            use_abstract_mesh(mesh),
            mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", False),
            pltpu.force_tpu_interpret_mode(
                pltpu.InterpretParams(
                    dma_execution_mode="eager", skip_floating_point_ops=False
                )
            ),
        ):
            # The wrapper donates the state operands; give each run its own.
            states = {k: jnp.copy(inputs[k]) for k in ("conv_state", "recurrent_state")}
            result = wrapper.fused_conv1d_gdn(
                per_seq_rolled=per_seq_rolled,
                decode_tile_size=2,
                # CPU interpret mode needs C = 1 tiles on the prefill path.
                mixed_tile_size=1,
                **{**inputs, **states},
            )
            return [np.asarray(x) for x in jax.tree.leaves(result)]
    finally:
        pltpu.reset_tpu_interpret_mode_state()


def test_pre_rolled_per_seq_inputs_match_in_kernel_rolls():
    inputs = _inputs()
    expected = _run(inputs)
    got = _run(inputs, _rolled(inputs, NUM_DECODE))
    for e, g in zip(expected, got):
        np.testing.assert_array_equal(g, e)

    # Positive control: rolled to the wrong sequence, the prefill rows differ.
    wrong = _run(inputs, _rolled(inputs, NUM_DECODE - 1))
    assert any(not np.array_equal(w, e) for w, e in zip(wrong, expected))


@pytest.mark.parametrize("offsets_per_seq", [False, True])
def test_step_graph_rolls_match_jnp_roll(offsets_per_seq):
    rng = np.random.default_rng(0)
    seqs, blocks, window = 7, 32, 4
    qsl = np.cumsum(np.r_[0, rng.integers(1, 5, seqs)]).astype(np.int32)
    seq_lens = rng.integers(1, 50, seqs).astype(np.int32)
    state_indices = rng.choice(np.arange(1, blocks), seqs, replace=False)
    slot_offsets = rng.integers(0, window, blocks).astype(np.int32)
    ckpt = rng.integers(1, blocks, (seqs, window)).astype(np.int32)
    offsets = slot_offsets[state_indices]
    for shift in range(seqs + 1):
        packed, rolled_ckpt = gdn_attention_op._roll_per_seq_inputs(
            torch.from_numpy(qsl),
            torch.from_numpy(seq_lens),
            torch.from_numpy(state_indices.astype(np.int32)),
            torch.from_numpy(offsets if offsets_per_seq else slot_offsets),
            torch.from_numpy(ckpt),
            torch.tensor(shift, dtype=torch.int32),
            offsets_per_seq=offsets_per_seq,
        )
        want = np.concatenate(
            [np.roll(x, -shift) for x in (qsl, seq_lens, state_indices, offsets)]
        )
        np.testing.assert_array_equal(packed.numpy(), want)
        np.testing.assert_array_equal(rolled_ckpt.numpy(), np.roll(ckpt, -shift, 0))


@pytest.mark.parametrize("spec", [False, True])
def test_pooled_entry_places_rolled_inputs_after_optional_ones(monkeypatch, spec):
    from vllm_torchtpu.layers.core import gdn_attention

    seen = {}

    def fake_shard_map(fn, *, mesh, in_specs, out_specs, check_vma):
        def run(*args):
            seen["args"], seen["specs"] = args, in_specs
            return args[3], jnp.zeros((args[0].shape[0], 4))

        return run

    monkeypatch.setattr(gdn_attention.jax, "shard_map", fake_shard_map)
    seqs, blocks = 6, 64
    rolled = jnp.arange(4 * seqs + 1, dtype=jnp.int32)
    ckpt = jnp.ones((seqs, 4), jnp.int32) if spec else None
    rolled_ckpt = ckpt + 1 if spec else None
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))
    gdn_attention.run_jax_gdn_attention_pooled(
        jnp.zeros((24, 8)),
        jnp.zeros((24, 2)),
        jnp.zeros((24, 2)),
        jnp.zeros((blocks, 16, 2, 8)),
        jnp.zeros((8, 1, 4)),
        None,
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        jnp.arange(seqs, dtype=jnp.int32),
        jnp.arange(seqs + 1, dtype=jnp.int32) * 4,
        jnp.full((3,), seqs, jnp.int32),
        jnp.full((seqs,), 100, jnp.int32),
        jnp.zeros((seqs,), jnp.int32) if spec else None,
        ckpt,
        n_kq=1,
        n_v=2,
        d_k=4,
        d_v=4,
        kernel_size=4,
        pool_block_tokens=16,
        mesh=mesh,
        num_spec_tokens=3 if spec else 0,
        read_offsets_per_seq=True,
        per_seq_rolled=rolled,
        per_seq_rolled_ckpt=rolled_ckpt,
    )
    args, specs = seen["args"], seen["specs"]
    assert len(args) == len(specs) == 16
    assert (args[12] is None) == (not spec) and (args[13] is None) == (not spec)
    np.testing.assert_array_equal(np.asarray(args[14]), np.asarray(rolled))
    if spec:
        np.testing.assert_array_equal(np.asarray(args[15]), np.asarray(rolled_ckpt))
    else:
        assert args[15] is None and specs[15] is None
