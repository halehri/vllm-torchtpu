"""GDN decode (MTP verify) microbench at the Qwen3.8-Flash TP8 geometry.

Per rank: 2 key heads, 6 value heads, d_k = d_v = 128 (conv dim 1280),
conv width 4, MTP K = 3 (4-token verify windows, 4 checkpoints per
sequence), FP32 recurrent state. Dense state tensors (the serving pool keeps
the same per-checkpoint bytes). Sweeps the decode tile and the empty
PER_SEQ pass, and reports a state-traffic roofline.
"""

import functools
import json
import os
import sys
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.kernels.gdn.v3 import tiling, wrapper  # noqa: E402

N_KQ, N_V, D, K = 2, 6, 128, 4
SPEC = 3
W = SPEC + 1
ITERS = 20
HBM_BYTES_PER_US = 3.5e6  # ~3.5 TB/s per TensorCore


def make(num_seqs, rng):
    conv_dim = N_KQ * D * 2 + N_V * D
    tokens = num_seqs * W
    blocks = 1 + num_seqs * W
    keys = iter(jax.random.split(jax.random.key(rng), 12))
    return dict(
        qkv=jax.random.normal(next(keys), (tokens, conv_dim), jnp.bfloat16),
        b=jax.random.normal(next(keys), (tokens, N_V), jnp.bfloat16),
        a=jax.random.normal(next(keys), (tokens, N_V), jnp.bfloat16),
        conv_state=jax.random.normal(next(keys), (blocks, K - 1, conv_dim)),
        recurrent_state=jax.random.normal(next(keys), (blocks, N_V, D, D)) * 0.1,
        conv_weight=jax.random.normal(next(keys), (conv_dim, 1, K), jnp.bfloat16),
        conv_bias=None,
        a_log=jax.random.normal(next(keys), (N_V,)),
        dt_bias=jax.random.normal(next(keys), (N_V,)),
        query_start_loc=jnp.arange(num_seqs + 1, dtype=jnp.int32) * W,
        state_indices=jnp.arange(num_seqs, dtype=jnp.int32) * W + 1,
        distribution=jnp.full((3,), num_seqs, jnp.int32),
        seq_lens=jnp.full((num_seqs,), 4000, jnp.int32),
        read_offsets=jnp.zeros((num_seqs,), jnp.int32),
    )


def roofline_us(num_seqs):
    state = N_V * D * D * 4
    conv = (K - 1) * (N_KQ * D * 2 + N_V * D) * 4
    read = num_seqs * (state + conv)
    write = num_seqs * W * (state + conv)
    return (read + write) / HBM_BYTES_PER_US


def bench(args, **static):
    fn = functools.partial(
        wrapper.fused_conv1d_gdn.__wrapped__,
        n_kq=N_KQ, n_v=N_V, d_k=D, d_v=D, kernel_size=K, num_spec_tokens=SPEC,
        **static,
    )

    @jax.jit
    def loop(conv_state, recurrent_state):
        def body(i, carry):
            conv, rec, acc = carry
            (conv, rec), out = fn(**{**args, "conv_state": conv, "recurrent_state": rec})
            return conv, rec, acc + out.reshape(-1)[0].astype(jnp.float32)
        return jax.lax.fori_loop(
            0, ITERS, body, (conv_state, recurrent_state, jnp.float32(0))
        )

    jax.block_until_ready(loop(args["conv_state"], args["recurrent_state"]))
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        jax.block_until_ready(loop(args["conv_state"], args["recurrent_state"]))
        samples.append((time.perf_counter() - t0) / ITERS * 1e6)
    (conv, rec), out = jax.jit(fn)(**args)
    return float(np.median(samples)), (np.asarray(out, np.float32), np.asarray(rec))


def test_gdn_decode():
    results = []
    for num_seqs in (1, 6, 12):
        args = make(num_seqs, num_seqs)
        ref = None
        arms = [("dynamic", None, {})]
        arms += [(f"T{t}", t, {}) for t in (1, 2, 4, 8) if t <= num_seqs]
        arms += [("dynamic_batched_only", None, dict(batched_only=True))]
        original = tiling.get_tile_sizes
        for name, tile, static in arms:

            def forced(*a, tile=tile, **kw):
                t, c = original(*a, **kw)
                return (tile or t), c

            with mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", True), \
                    mock.patch.object(tiling, "get_tile_sizes", forced):
                try:
                    us, got = bench(args, **static)
                    if ref is None:
                        ref = got
                    diff = max(float(np.max(np.abs(a - b))) for a, b in zip(got, ref))
                except Exception as e:  # noqa: BLE001
                    us, diff = repr(e)[:200], None
            row = dict(seqs=num_seqs, rows=num_seqs * W, arm=name, us=us,
                       roofline_us=roofline_us(num_seqs), max_diff=diff)
            print("GDN", json.dumps(row), flush=True)
            results.append(row)
    out = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
    Path(out, "bench_gdn_decode.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    test_gdn_decode()
