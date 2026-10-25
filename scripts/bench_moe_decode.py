# SPDX-License-Identifier: Apache-2.0
"""Microbench: the fused MoE decode kernel at Qwen3.8-Flash's served EP shapes.

8 ranks, 64 FP8 experts each (h 2560, i 640), TP-sliced shared expert (i 128),
top-8 over 512. Per token count, the kernel's time for:
  uniform  random top-8 over all 512 experts (the roofline case)
  one      every token picks one expert on every rank (fixed cost + 1 expert)
  hot      all picks inside rank 0's experts (worst EP straggler)
plus a bare psum of [m, h] bf16 (the in-kernel all-reduce's floor). Prints
MOEBW lines with the expected distinct experts per rank and the implied
weight-read bandwidth.
"""

import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.fused_moe_decode import fused_moe_decode

N, H, INTER, INTER_S, E, TOPK = 8, 2560, 640, 128, 64, 8
REPS = 48  # calls chained inside one jit, like 48 layers, so dispatch amortizes
F8 = jnp.float8_e4m3fn
EXPERT_BYTES = H * 2 * INTER + INTER * H  # FP8 w13 + w2


def _time(f, args, iters=50):
    jax.block_until_ready(f(*args))
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(f(*args))
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _weights(mesh, rank):
    def gen(r):
        ks = jax.random.split(jax.random.fold_in(jax.random.key(3), r[0, 0]), 4)

        def q(w, axis):
            s = jnp.max(jnp.abs(w), axis=axis, keepdims=True) / 448.0
            return (w / s).astype(F8), s

        w1, w1s = q(jax.random.normal(ks[0], (E, H, 2 * INTER)) * 0.02, 1)
        w2, w2s = q(jax.random.normal(ks[1], (E, INTER, H)) * 0.02, 1)
        sw1, sw1s = q(jax.random.normal(ks[2], (H, 2 * INTER_S)) * 0.02, 0)
        sw2, sw2s = q(jax.random.normal(ks[3], (INTER_S, H)) * 0.02, 0)
        return w1, w2, w1s[:, :, None, :], w2s[:, :, None, :], sw1, sw1s, sw2, sw2s

    return jax.jit(
        jax.shard_map(gen, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False)
    )(rank)


def _ids(kind, m, rng):
    if kind == "uniform":
        return np.stack([rng.choice(N * E, TOPK, replace=False) for _ in range(m)])
    if kind == "one":
        return np.tile(np.arange(N) * E, (m, 1))
    return np.stack([rng.choice(E, TOPK, replace=False) for _ in range(m)])  # hot


def test_moe_decode_bandwidth():
    mesh = Mesh(np.asarray(jax.devices()[:N]), ("d",))
    sh = NamedSharding(mesh, P("d"))
    rank = jax.device_put(np.arange(N, dtype=np.int32).reshape(N, 1), sh)
    weights = _weights(mesh, rank)
    gate = jax.device_put(
        jnp.asarray(np.random.default_rng(1).normal(size=(H, 1)) * 0.02, jnp.bfloat16),
        NamedSharding(mesh, P()),
    )
    specs = (P("d"),) * 10 + (P(), P("d"), P("d"))

    def body(r, x, w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s, g, tw, ti):
        for _ in range(REPS):
            x = fused_moe_decode(
                r.reshape(-1), x, w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s, g, tw, ti,
                n_ranks=N, expert_start=r.reshape(()) * E,
            )
        return x

    def chained_psum(x):
        for _ in range(REPS):
            x = jax.lax.psum(x, "d") * 0.125
        return x

    fused = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=specs, out_specs=P("d"),
                                  check_vma=False))
    psum = jax.jit(jax.shard_map(chained_psum, mesh=mesh,
                                 in_specs=P("d"), out_specs=P("d"), check_vma=False))
    rng = np.random.default_rng(0)
    for m in (4, 8, 16, 32, 64, 128):
        x = jax.device_put(
            jnp.tile(jnp.asarray(rng.normal(size=(m, H)), jnp.bfloat16), (N, 1)), sh)
        t = _time(psum, (x,)) / REPS
        print("MOEBW " + json.dumps(dict(kind="psum", m=m, us=t * 1e6)), flush=True)
        tw = rng.dirichlet(np.ones(TOPK), size=m).astype(np.float32)
        tw = jax.device_put(jnp.tile(jnp.asarray(tw, jnp.bfloat16), (N, 1)), sh)
        for kind in ("uniform", "one", "hot"):
            ids = _ids(kind, m, rng).astype(np.int32)
            local = [len({int(e) - r * E for e in ids.ravel() if r * E <= e < (r + 1) * E})
                     for r in range(N)]
            ti = jax.device_put(jnp.tile(jnp.asarray(ids), (N, 1)), sh)
            try:
                t = _time(fused, (rank, x, *weights, gate, tw, ti)) / REPS
            except Exception as e:  # e.g. a VMEM limit at large m
                print("MOEBW " + json.dumps(dict(kind=kind, m=m, error=str(e)[:300])), flush=True)
                continue
            nb = max(local) * EXPERT_BYTES
            print("MOEBW " + json.dumps(dict(
                kind=kind, m=m, us=t * 1e6, max_local_experts=max(local),
                mean_local_experts=sum(local) / N,
                gbps_max_rank=nb / t / 1e9)), flush=True)
