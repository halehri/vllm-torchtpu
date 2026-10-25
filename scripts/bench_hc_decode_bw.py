"""HC decode mix: achieved HBM bandwidth vs a weight-copy roofline.

Per call the fused decode mix (and its plain-JAX reference) streams the HC's
down/up/inject/norm weights (~13 MB bf16 at Qwen3.8-Flash: 4 x 2560 streams,
low rank 320). Times N back-to-back calls inside one jit (no host overhead)
at 4/8/32/64 rows and compares GB/s with a plain copy of the same bytes.
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.kernels.qwen4_exp.hc_decode import hc_decode_mix, hc_mix_reference  # noqa: E402

BRANCHES, HIDDEN, LOWRANK, EPS = 4, 2560, 320, 1e-6
WIDTH = BRANCHES * HIDDEN
N = 48  # ~one decode step's worth of HC mixes per iteration


def weights(rng, layers):
    def w(shape, scale):
        return jnp.asarray(rng.normal(0, scale, (layers, *shape)), jnp.bfloat16)

    return (w((WIDTH,), 0.3), w((LOWRANK, WIDTH), 0.02), w((WIDTH, LOWRANK), 0.05),
            w((BRANCHES, WIDTH), 0.02))


def nbytes(ws):
    return sum(int(np.prod(a.shape[1:])) * 2 for a in ws)


def timed(f, *args, iters=10):
    f(*args)
    jax.block_until_ready(f(*args))
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(f(*args))
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def test_hc_decode_bandwidth():
    rng = np.random.default_rng(0)
    ws = weights(rng, N)  # distinct weights per call, like 48 layers
    per_call = nbytes(ws)

    @jax.jit
    def copy_all(ws):
        return [a + jnp.bfloat16(0) for a in ws]

    t_copy = timed(copy_all, ws)
    copy_bw = 2 * per_call * N / t_copy / 1e9  # read + write
    print("HCBW " + json.dumps(dict(kind="copy", bytes_per_call=per_call,
                                    ms_per_call=t_copy / N * 1e3, gbps=copy_bw)), flush=True)
    for rows in (4, 8, 32, 64):
        x = jnp.asarray(rng.normal(0, 1, (rows, WIDTH)), jnp.bfloat16)
        for kind, fn in (("fused", hc_decode_mix), ("xla", hc_mix_reference)):

            @jax.jit
            def run(x, ws, fn=fn):
                def body(i, acc):
                    w = [a[i] for a in ws]
                    mixed, inj = fn(x, *w, branches=BRANCHES, hidden=HIDDEN, eps=EPS)[:2]
                    return acc + mixed[0, 0].astype(jnp.float32) + inj[0, 0]
                return jax.lax.fori_loop(0, N, body, jnp.float32(0))

            t = timed(run, x, ws)
            gbps = per_call * N / t / 1e9
            print("HCBW " + json.dumps(dict(kind=kind, rows=rows, ms_per_call=t / N * 1e3,
                                            gbps=gbps, frac_of_copy_read=gbps / (copy_bw / 2))),
                  flush=True)
