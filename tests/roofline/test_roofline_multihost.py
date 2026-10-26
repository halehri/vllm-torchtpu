# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the measured 32-rank collective table and tpu7x-measured chip."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from vllm_torchtpu.roofline import (
    CHIPS,
    ModelGeometry,
    Parallel,
    StepFloor,
    StepShape,
    step_floor,
)
from vllm_torchtpu.roofline.collectives import Collective, CollectiveTable
from vllm_torchtpu.roofline.measured import TPU7X_32R

# hidden 8192 makes a 32-row step carry a 512 KiB all-reduce, a measured payload.
WIDE_GEOM = ModelGeometry(
    hidden=8192,
    num_layers=4,
    layer_types=("linear_attention",) * 3 + ("full_attention",),
    num_heads=64,
    num_kv_heads=4,
    head_dim=256,
    attn_output_gate=True,
    linear_k_heads=16,
    linear_v_heads=64,
    linear_k_dim=128,
    linear_v_dim=128,
    conv_kernel=4,
    num_experts=512,
    top_k=10,
    moe_inter=1024,
    shared_inter=1024,
    vocab=248320,
    mtp_layers=0,
)
TP32 = Parallel(tp=32)
KIND = {"dispatch_ag": "ag", "route_ag": "ag", "restore_ag": "ag", "ar": "ar"}


def _query(key: tuple[str, int], payload: float) -> Collective:
    return Collective(key[0], key[1], 0.0, 0, 0.0, KIND[key[0]], payload)


def _max_spread(table: CollectiveTable) -> float:
    """Largest max - min across the raw samples at one payload."""
    return max(
        max(s) - min(s) for pts in table.rows.values() for _, s in pts if len(s) > 1
    )


def _chain_collectives(sf: StepFloor) -> list[Collective]:
    return [c for m in sf.modules for c in m.collectives if c.group > 1]


def test_measured_chip_prices_chain_from_table() -> None:
    chip = CHIPS["tpu7x-measured"]
    shape = StepShape(batch=32, rows_per_seq=1, context=1024)
    sf = step_floor(WIDE_GEOM, chip, TP32, shape)
    colls = _chain_collectives(sf)
    ars = [c for c in colls if c.kind == "ar" and c.payload == 512 * 1024]
    assert ars
    for c in ars:
        t, src = TPU7X_32R.time_s(c)
        assert src == "measured"
        assert t == pytest.approx(30.77e-6)
    assert sf.chain_s == pytest.approx(sum(TPU7X_32R.time_s(c)[0] for c in colls))

    no_ceiling = replace(chip, hbm_bw=math.inf, bf16_flops=math.inf, fp8_flops=math.inf)
    assert step_floor(WIDE_GEOM, no_ceiling, TP32, shape).binding == "chain"

    base = step_floor(WIDE_GEOM, CHIPS["tpu7x"], TP32, shape)
    assert CHIPS["tpu7x"].collective_table is None
    assert base.chain_s == pytest.approx(base.ici_s)
    assert base.extrapolated_chain_s == 0.0


def test_table_loo_interior_within_pooled_spread() -> None:
    tol = _max_spread(TPU7X_32R)
    checked = 0
    for key, pts in TPU7X_32R.rows.items():
        for i, (payload, samples) in enumerate(pts):
            for j, held_out in enumerate(samples):
                rest = samples[:j] + samples[j + 1 :]
                kept = pts[:i] + (((payload, rest),) if rest else ()) + pts[i + 1 :]
                if not kept[0][0] < payload < kept[-1][0]:
                    continue
                t, _ = CollectiveTable({key: kept}).time_s(_query(key, payload))
                assert abs(t - held_out) <= tol, (key, payload, t, held_out, tol)
                checked += 1
    assert checked > 0


def test_table_marks_extrapolation_outside_hull() -> None:
    ar = TPU7X_32R.rows[("ar", 32)]
    small = CollectiveTable({("ar", 32): tuple(p for p in ar if p[0] <= 1024 * 1024)})
    t, src = small.time_s(_query(("ar", 32), 4096 * 1024))
    assert src == "extrapolated"
    assert t == pytest.approx(38.63e-6)

    chip = replace(CHIPS["tpu7x"], collective_table=small)
    sf = step_floor(WIDE_GEOM, chip, TP32, StepShape(batch=256, context=1024))
    assert sf.extrapolated_chain_s > 0.0


def test_context_invariance_same_payload() -> None:
    # c=256 at 8K context, same payloads as the table's c=256 rows at 1K context.
    tol = _max_spread(TPU7X_32R)
    for key, d1_s in ((("restore_ag", 32), 42.35e-6), (("ar", 32), 95.43e-6)):
        t, src = TPU7X_32R.time_s(_query(key, 4096 * 1024))
        assert src == "measured"
        assert abs(d1_s - t) <= tol
