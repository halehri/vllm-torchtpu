"""QSA sparse prefill attention: what one KV tile visit costs, part by part.

Same per-rank SP call as bench_qsa_compaction.py (1024 rows, u04-matched
spread). Per variant (a fresh process, since the kernel reads its bench-only
QSA_PREFILL_ABLATE switches at trace time), times the call at 24K and 160K
context; per-tile cost = (t160 - t24) / (tiles160 - tiles24). Variants drop one
part of the tile body (bitset->mask expansion, causal compare, FP8 key/value
fix-ups, QK matmul, mask select, exp, PV matmul, KV DMA) or swap in a cheaper
form (BF16 exp, the BF16 MXU path). Prints TCOST lines.
"""

import json
import os
import subprocess
import sys

VARIANTS = {
    "base": {},
    "nomask": {"QSA_PREFILL_ABLATE": "nomask"},
    "nocausal": {"QSA_PREFILL_ABLATE": "nocausal"},
    "nomask_nocausal": {"QSA_PREFILL_ABLATE": "nomask,nocausal"},
    "nokvfix": {"QSA_PREFILL_ABLATE": "nokvfix"},
    "noqk": {"QSA_PREFILL_ABLATE": "noqk"},
    "nowhere": {"QSA_PREFILL_ABLATE": "nowhere"},
    "noexp": {"QSA_PREFILL_ABLATE": "noexp"},
    "bf16exp": {"QSA_PREFILL_ABLATE": "bf16exp"},
    "nopv": {"QSA_PREFILL_ABLATE": "nopv"},
    "noqk_nopv": {"QSA_PREFILL_ABLATE": "noqk,nopv"},
    "nodma": {"QSA_PREFILL_ABLATE": "nodma"},
    "bf16mxu": {"VLLM_TPU_QSA_PREFILL_BF16_MXU": "1"},
}

CHILD = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts").resolve()))
import numpy as np
import bench_qsa_compaction as b
rng = np.random.default_rng(11)
out = {}
for context, groups, ratio in (b.BUCKETS[0], b.BUCKETS[-1]):
    tiles = min(-(-int(groups * b.RATIO * ratio) // b.KV_TILE), context // b.KV_TILE)
    inputs = b.make_inputs(rng, context, tiles * b.KV_TILE // b.RATIO, groups)
    out[context] = (tiles, b.timed(b.attend, *inputs))
(t0, a), (t1, c) = out[b.BUCKETS[0][0]], out[b.BUCKETS[-1][0]]
print("TCOST " + json.dumps(dict(variant=sys.argv[1], ms24=a, ms160=c, tiles24=t0,
      tiles160=t1, us_per_tile=(c - a) / (t1 - t0) * 1e3,
      fixed_ms=a - t0 * (c - a) / (t1 - t0))), flush=True)
"""


def test_tile_cost_split():
    for name, extra in VARIANTS.items():
        env = dict(os.environ)
        env.pop("QSA_PREFILL_ABLATE", None)
        env.update(extra)
        r = subprocess.run([sys.executable, "-c", CHILD, name], env=env,
                           capture_output=True, text=True, timeout=1200)
        lines = [l for l in r.stdout.splitlines() if l.startswith("TCOST")]
        for l in lines:
            print(l, flush=True)
        if r.returncode or not lines:
            print("TCOST " + json.dumps(dict(variant=name, error=r.stderr[-800:])), flush=True)
