# SPDX-License-Identifier: Apache-2.0
"""Microbench: XLA psum of [m, 2560] bf16 over 8 TCs under libtpu flag variants.

Each variant runs in a fresh process (LIBTPU_INIT_ARGS is read at startup) and
prints ARFLAG lines: µs per call for a chain of 48 psums and for a chain of 48
plain scales (the jit launch overhead to subtract).
"""

import json
import os
import subprocess
import sys

VARIANTS = {
    "default": "",
    "binomial_1m": "--xla_tpu_small_all_reduce_binomial_threshold_bytes=1048576",
    "binomial_256k": "--xla_tpu_small_all_reduce_binomial_threshold_bytes=262144",
    "latbound_1m": "--xla_tpu_latency_bound_ar_decomposer_threshold_bytes=1048576",
    "short_chunks_1": "--xla_tpu_inference_short_all_reduce_max_tensor_chunks=1",
    "short_chunks_64": "--xla_tpu_inference_short_all_reduce_max_tensor_chunks=64",
}

CHILD = r"""
import json, statistics, sys, time
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
N, H, REPS = 8, 2560, 48
mesh = Mesh(np.asarray(jax.devices()[:N]), ("d",))
sh = NamedSharding(mesh, P("d"))
def chain(step):
    def body(x):
        for _ in range(REPS):
            x = step(x) * 0.125
        return x
    return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False))
def t(f, x, iters=50):
    jax.block_until_ready(f(x)); s = []
    for _ in range(iters):
        a = time.perf_counter(); jax.block_until_ready(f(x)); s.append(time.perf_counter() - a)
    return statistics.median(s) / REPS * 1e6
psum, scale = chain(lambda x: jax.lax.psum(x, "d")), chain(lambda x: x)
rng = np.random.default_rng(0)
for m in (4, 8, 16, 24, 32, 64, 128):
    x = jax.device_put(jnp.asarray(rng.normal(size=(N * m, H)), jnp.bfloat16), sh)
    print("ARFLAG " + json.dumps(dict(variant=sys.argv[1], m=m, psum_us=t(psum, x), scale_us=t(scale, x))), flush=True)
"""


def test_allreduce_flag_variants():
    for name, flags in VARIANTS.items():
        env = dict(os.environ)
        env["LIBTPU_INIT_ARGS"] = (env.get("LIBTPU_INIT_ARGS", "") + " " + flags).strip()
        r = subprocess.run([sys.executable, "-c", CHILD, name], env=env,
                           capture_output=True, text=True, timeout=900)
        lines = [l for l in r.stdout.splitlines() if l.startswith("ARFLAG")]
        for l in lines:
            print(l, flush=True)
        if r.returncode or not lines:
            print("ARFLAG " + json.dumps(dict(variant=name, error=r.stderr[-600:])), flush=True)
