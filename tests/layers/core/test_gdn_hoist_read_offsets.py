# SPDX-License-Identifier: Apache-2.0
"""Pooled GDN ``read_offsets_per_seq`` (VLLM_TPU_GDN_HOIST_READ_OFFSETS).

The caller gathers ``slot_read_offsets[state_indices]`` once per step and the
pooled entry passes it through unchanged; the per-layer kernel then sees the
same per-sequence offsets as when it gathers them itself.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.core import gdn_attention


@pytest.mark.parametrize("per_seq", [False, True])
def test_per_seq_offsets_reach_the_kernel_unchanged(monkeypatch, per_seq):
    seen = {}

    def fake_shard_map(fn, *, mesh, in_specs, out_specs, check_vma):
        def run(*args):
            seen["offsets"] = np.asarray(args[12])
            seen["ckpt"] = np.asarray(args[13])
            return args[3], jnp.zeros((args[0].shape[0], 4))

        return run

    monkeypatch.setattr(gdn_attention.jax, "shard_map", fake_shard_map)
    rng = np.random.default_rng(0)
    blocks, seqs = 64, 6
    slot_offsets = jnp.asarray(rng.integers(0, 4, blocks), jnp.int32)
    state_indices = jnp.asarray(rng.choice(np.arange(1, blocks), seqs), jnp.int32)
    ckpt = jnp.asarray(rng.integers(1, blocks, (seqs, 4)), jnp.int32)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))
    offsets = slot_offsets[state_indices] if per_seq else slot_offsets
    gdn_attention.run_jax_gdn_attention_pooled(
        jnp.zeros((24, 8)),
        jnp.zeros((24, 2)),
        jnp.zeros((24, 2)),
        jnp.zeros((blocks, 16, 2, 8)),
        jnp.zeros((8, 1, 4)),
        None,
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        state_indices,
        jnp.arange(seqs + 1, dtype=jnp.int32) * 4,
        jnp.full((3,), seqs, jnp.int32),
        jnp.full((seqs,), 100, jnp.int32),
        offsets,
        ckpt,
        n_kq=1,
        n_v=2,
        d_k=4,
        d_v=4,
        kernel_size=4,
        pool_block_tokens=16,
        mesh=mesh,
        num_spec_tokens=3,
        read_offsets_per_seq=per_seq,
    )
    np.testing.assert_array_equal(
        seen["offsets"], np.asarray(slot_offsets)[np.asarray(state_indices)]
    )
    np.testing.assert_array_equal(seen["ckpt"], np.asarray(ckpt))
