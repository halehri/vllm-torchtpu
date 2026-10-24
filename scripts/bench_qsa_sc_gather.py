"""SparseCore gather of QSA-selected KV into contiguous staging: throughput.

Per rank and QSA layer, a verify step with R rows selects 512 compressed groups
(+ the partial tail) per row; each group is 4 tokens x 1 KiB of FP8 paired KV
(one 4 KiB micro-page). Staging them contiguously would let RPA fetch one DMA
per KV tile (measured ~35% off the decode step). This times:
  1. the SC gather alone (ragged_gather_v2 over a [groups, 1024] f32 view),
  2. the 12-layer RPA decode step alone (lean + slot + shared, as served),
  3. both in one program, the gather independent of the RPA (as it would be
     for the next layer), to see how much of the gather hides under the TC.
"""

import functools
import itertools
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import bench_qsa_rpa_decode as b  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa  # noqa: E402
from vllm_torchtpu.kernels.sparse_core.ragged_gather_v2 import ragged_gather_v2  # noqa: E402

ITERS = 20
GROUP_WORDS = 1024  # 4 tokens x 1 KiB / 4 bytes


def timed(fn, *args):
    out = fn(*args)
    jax.block_until_ready(out)
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - t0) / ITERS * 1e6)
    return float(np.median(samples))


def test_sc_gather():
    rng = np.random.default_rng(0)
    groups = 4096 * 512 // 4  # same byte size as the 4096-page bench pool
    source = jax.random.normal(jax.random.key(1), (groups, GROUP_WORDS), jnp.float32)
    cache = b.make_cache()
    results = []
    for reqs, rows in ((4, 16), (6, 24), (8, 32)):
        n = rows * 513
        idx = jnp.asarray(rng.integers(1, groups, (12, n)), jnp.int32)
        start, end = jnp.zeros((1,), jnp.int32), jnp.full((1,), n, jnp.int32)

        @jax.jit
        def gather_loop(source, idx):
            def body(i, acc):
                out = jnp.float32(0)
                for layer in range(12):
                    dyn = (acc * 0).astype(jnp.int32)
                    g = ragged_gather_v2(
                        source, idx[layer] + dyn, start + dyn, end + dyn
                    )
                    out = out + g[0, 0] + g[-1, -1]
                return acc + out * 0
            return jax.lax.fori_loop(0, ITERS, body, jnp.float32(0))

        gather_us = timed(gather_loop, source, idx)

        # The 12-layer RPA step (bench_qsa_rpa_decode's step, served flags).
        base = b.make_inputs(rng, requests=reqs, rows=rows, overlap=0.0)
        lengths = np.asarray(base[3])
        layers = []
        for _ in range(12):
            q = rng.standard_normal((rows, b.Q_HEADS, b.D)).astype(np.float32)
            g_ = rng.standard_normal((rows, b.Q_HEADS, b.D)).astype(np.float32)
            layers.append((jnp.asarray(q, jnp.bfloat16), jnp.asarray(g_, jnp.bfloat16),
                           jnp.asarray(b.pick_blocks(rng, lengths, "random")), base[3], base[4]))
        qs = jnp.stack([x[0] for x in layers])
        gates = jnp.stack([x[1] for x in layers])
        blocks = jnp.stack([x[2] for x in layers])
        attend = functools.partial(
            qsa_rpa.decode_attention, kv_heads=1, ratio=b.RATIO, k_scale=1.0,
            v_scale=1.0, fused=False, shared_schedule=True,
        )
        os.environ["VLLM_TPU_QSA_DECODE_LEAN_FETCH"] = "1"
        os.environ["VLLM_TPU_QSA_DECODE_SLOT_READ"] = "1"

        def make_loop(with_gather):
            @functools.partial(jax.jit, donate_argnums=(0,))
            def loop(cache, qs, source, idx):
                def body(i, carry):
                    cache, acc = carry
                    for layer in range(12):
                        out, cache = attend(
                            qs[layer], gates[layer], blocks[layer], base[3], base[4],
                            cache, **({"kv_head_offset": 1} if layer % 2 else {}),
                        )
                        acc = acc + out.astype(jnp.float32).sum()
                        if with_gather:
                            # Independent of this layer's attention, as the next
                            # layer's gather would be.
                            dyn = (acc * 0).astype(jnp.int32)
                            g = ragged_gather_v2(
                                source, idx[layer] + dyn, start + dyn, end + dyn
                            )
                            acc = acc + g[0, 0] * 0
                    return cache, acc
                return jax.lax.fori_loop(0, ITERS, body, (cache, jnp.float32(0)))
            return loop

        rpa_us = None
        both_us = None
        for name, loop in (("rpa", make_loop(False)), ("both", make_loop(True))):
            out = loop(cache, qs, source, idx)
            cache = out[0]
            jax.block_until_ready(out)
            samples = []
            for _ in range(5):
                t0 = time.perf_counter()
                out = loop(cache, qs, source, idx)
                cache = out[0]
                jax.block_until_ready(out)
                samples.append((time.perf_counter() - t0) / ITERS * 1e6)
            if name == "rpa":
                rpa_us = float(np.median(samples))
            else:
                both_us = float(np.median(samples))
        row = dict(rows=rows, gather_us_per_step=gather_us,
                   gather_mb_per_layer=n * GROUP_WORDS * 4 / 1e6,
                   gather_gbps=n * GROUP_WORDS * 4 * 12 / (gather_us * 1e-6) / 1e9,
                   rpa_us_per_step=rpa_us, rpa_plus_gather_us_per_step=both_us,
                   exposed_gather_us=both_us - rpa_us,
                   one_dma_saving_us=0.36 * rpa_us)
        print("SCG", json.dumps(row), flush=True)
        results.append(row)
    out = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
    Path(out, "bench_qsa_sc_gather.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    test_sc_gather()
