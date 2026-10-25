# SPDX-License-Identifier: Apache-2.0
"""Microbench: small-payload all-reduce of [m, 2560] bf16 over the 8 TCs.

  psum     XLA's all-reduce
  barrier  a Pallas call that only signals and waits on every peer (sync floor)
  twoshot  the fused MoE decode kernel's scheme: barrier, all-to-all of column
           chunks + local FP32 sum, then an all-gather of the reduced chunks
  oneshot  barrier, every rank pushes its whole partial to every peer, then
           each rank sums the n partials it holds (one hop, no all-gather)

Each is chained 48 times in one jit (like 48 layers), so host dispatch
amortizes. Prints ARBW lines (µs per call) and checks against psum.
"""

import functools
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

N, H, REPS = 8, 2560, 48
HP = -(-H // (N * 128)) * (N * 128)  # 3072: lane-aligned column chunks
CHUNK = HP // N


def _time(f, args, iters=50):
    jax.block_until_ready(f(*args))
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(f(*args))
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _barrier(n):
    barrier = pltpu.get_barrier_semaphore()
    for d in range(n):
        pl.semaphore_signal(barrier, inc=1, device_id=(jnp.int32(d),),
                            device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(barrier, n)


def _kernel(rank_ref, x_ref, out_ref, recv, red, send_sems, recv_sem, ag_sems, ag_sem,
            *, mode):
    me = rank_ref[0]
    _barrier(N)
    if mode == "barrier":
        out_ref[...] = x_ref[...]
        return
    if mode == "oneshot":
        copies = [
            pltpu.make_async_remote_copy(
                x_ref, recv.at[me], send_sems.at[d], recv_sem,
                device_id=(jnp.int32(d),), device_id_type=pl.DeviceIdType.MESH)
            for d in range(N)
        ]
        for c in copies:
            c.start()
        pltpu.make_async_copy(recv, recv, recv_sem).wait()
        acc = recv[0].astype(jnp.float32)
        for s in range(1, N):
            acc = acc + recv[s].astype(jnp.float32)
        out_ref[...] = acc.astype(out_ref.dtype)
        for c in copies:
            c.wait_send()
        return
    # twoshot
    copies = [
        pltpu.make_async_remote_copy(
            x_ref.at[:, pl.ds(d * CHUNK, CHUNK)], recv.at[me], send_sems.at[d], recv_sem,
            device_id=(jnp.int32(d),), device_id_type=pl.DeviceIdType.MESH)
        for d in range(N)
    ]
    for c in copies:
        c.start()
    pltpu.make_async_copy(recv, recv, recv_sem).wait()
    acc = recv[0].astype(jnp.float32)
    for s in range(1, N):
        acc = acc + recv[s].astype(jnp.float32)
    red[...] = acc.astype(red.dtype)
    for c in copies:
        c.wait_send()
    col = pl.multiple_of(me * CHUNK, CHUNK)
    gather = [
        pltpu.make_async_remote_copy(
            red, out_ref.at[:, pl.ds(col, CHUNK)], ag_sems.at[d], ag_sem,
            device_id=(jnp.int32(d),), device_id_type=pl.DeviceIdType.MESH)
        for d in range(N)
    ]
    for c in gather:
        c.start()
    pltpu.make_async_copy(out_ref, out_ref, ag_sem).wait()
    for c in gather:
        c.wait_send()


def all_reduce(rank, x, *, mode):
    m, w = x.shape
    slot = (m, w) if mode != "twoshot" else (m, CHUNK)
    return pl.pallas_call(
        functools.partial(_kernel, mode=mode),
        out_shape=jax.ShapeDtypeStruct((m, w), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((N, *slot), x.dtype),
                pltpu.VMEM((m, CHUNK), x.dtype),
                pltpu.SemaphoreType.DMA((N,)),
                pltpu.SemaphoreType.DMA(()),
                pltpu.SemaphoreType.DMA((N,)),
                pltpu.SemaphoreType.DMA(()),
            ],
        ),
        compiler_params=pltpu.CompilerParams(collective_id=0),
        name=f"ar_{mode}_m{m}",
    )(rank.reshape(1).astype(jnp.int32), x)


def test_small_allreduce_latency():
    mesh = Mesh(np.asarray(jax.devices()[:N]), ("d",))
    sh = NamedSharding(mesh, P("d"))
    rank = jax.device_put(np.arange(N, dtype=np.int32).reshape(N, 1), sh)
    rng = np.random.default_rng(0)

    def chain(step):
        def body(r, x):
            for _ in range(REPS):
                x = step(r, x) * 0.125
            return x
        return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("d"), P("d")),
                                     out_specs=P("d"), check_vma=False))

    def single(step):
        return jax.jit(jax.shard_map(step, mesh=mesh, in_specs=(P("d"), P("d")),
                                     out_specs=P("d"), check_vma=False))

    steps = {
        "psum": lambda r, x: jax.lax.psum(x, "d"),
        "scale_only": lambda r, x: x,
        "barrier": functools.partial(all_reduce, mode="barrier"),
        "twoshot": functools.partial(all_reduce, mode="twoshot"),
        "oneshot": functools.partial(all_reduce, mode="oneshot"),
    }
    for m in (4, 8, 16, 32, 64):
        for kind, step in steps.items():
            w = HP if kind in ("twoshot",) else H
            x = jax.device_put(jnp.asarray(rng.normal(size=(N * m, w)), jnp.bfloat16), sh)
            try:
                if kind in ("twoshot", "oneshot"):
                    got = np.asarray(single(step)(rank, x), np.float32).reshape(N, m, w)
                    want = np.asarray(single(steps["psum"])(rank, x), np.float32).reshape(N, m, w)
                    err = float(np.abs(got - want).max() / np.abs(want).max())
                else:
                    err = 0.0
                t = _time(chain(step), (rank, x)) / REPS
            except Exception as e:
                print("ARBW " + json.dumps(dict(kind=kind, m=m, error=str(e)[:400])), flush=True)
                continue
            print("ARBW " + json.dumps(dict(kind=kind, m=m, us=t * 1e6, rel_err=err)), flush=True)
