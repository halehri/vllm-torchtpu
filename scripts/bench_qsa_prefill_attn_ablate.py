"""Bench-only: which part of a query block's fixed cost dominates.

Times the sparse prefill attention with the kernel's bench-only ablation
switches (QSA_PREFILL_ABLATE: notiles / noq / noinit / noepi), at 2048 and
8192 rows with 2 KV tiles per 512-row block (fixed-cost regime). Outputs of the
ablated variants are wrong by design; only the time deltas matter.
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_qsa_prefill_attn_fixed import fn, make_step  # noqa: E402

VARIANTS = ["", "notiles", "noq", "noinit", "noepi", "noq,noinit,noepi", "noq,noinit,noepi,notiles"]
ITERS = 30


def test_ablate():
    rng = np.random.default_rng(11)
    for tokens in (2048, 8192):
        inputs = make_step(rng, tokens, 2)
        base = None
        for variant in VARIANTS:
            os.environ["QSA_PREFILL_ABLATE"] = variant
            jax.clear_caches()
            compiled = jax.jit(fn).lower(*inputs).compile()
            for _ in range(3):
                jax.block_until_ready(compiled(*inputs))
            samples = []
            for _ in range(ITERS):
                start = time.perf_counter()
                jax.block_until_ready(compiled(*inputs))
                samples.append((time.perf_counter() - start) * 1000)
            ms = statistics.median(samples)
            base = ms if base is None else base
            print("PABL " + json.dumps(dict(tokens=tokens, variant=variant or "full",
                                            ms=ms, delta_ms=ms - base)), flush=True)
