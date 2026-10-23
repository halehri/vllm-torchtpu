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
"""Hard and serial step floors for standard and speculative decode."""

from __future__ import annotations

from dataclasses import dataclass

from .chips import ChipSpec
from .geometry import ModelGeometry
from .modules import Dtypes, ModuleCost, Parallel, StepShape, pass_costs


@dataclass(frozen=True)
class StepFloor:
    """Resource times and binding bottleneck for one decode step or cycle."""

    modules: tuple[ModuleCost, ...]
    hbm_s: float
    mm_s: float
    ici_s: float
    hard_floor_s: float
    serial_floor_s: float
    binding: str


def _combine_modules(modules: tuple[ModuleCost, ...], chip: ChipSpec) -> StepFloor:
    hbm_s = sum(m.hbm_bytes for m in modules) / chip.hbm_bw
    mm_s = (
        sum(m.fp8_flops for m in modules) / chip.fp8_flops
        + sum(m.bf16_flops for m in modules) / chip.bf16_flops
    )
    ici_s = (
        sum(m.ici_bytes for m in modules) / chip.link_bw
        if chip.link_bw is not None and chip.link_bw > 0
        else 0.0
    )
    hard_floor_s = max(hbm_s, mm_s, ici_s)
    serial_floor_s = (
        sum(
            max(
                m.hbm_bytes / chip.hbm_bw,
                m.fp8_flops / chip.fp8_flops + m.bf16_flops / chip.bf16_flops,
            )
            for m in modules
        )
        + ici_s
    )
    if hard_floor_s == hbm_s:
        binding = "hbm"
    elif hard_floor_s == mm_s:
        binding = "matmul"
    else:
        binding = "ici"
    return StepFloor(
        modules=modules,
        hbm_s=hbm_s,
        mm_s=mm_s,
        ici_s=ici_s,
        hard_floor_s=hard_floor_s,
        serial_floor_s=serial_floor_s,
        binding=binding,
    )


DEFAULT_DTYPES = Dtypes()


def step_floor(
    geom: ModelGeometry,
    chip: ChipSpec,
    par: Parallel,
    shape: StepShape,
    dt: Dtypes = DEFAULT_DTYPES,
) -> StepFloor:
    """Compute the floor for one non-speculative or verify forward pass."""
    return _combine_modules(pass_costs(geom, par, shape, dt), chip)


def spec_cycle_floor(
    geom: ModelGeometry,
    chip: ChipSpec,
    par: Parallel,
    batch: int,
    context: int,
    k: int,
    dt: Dtypes = DEFAULT_DTYPES,
) -> StepFloor:
    """Compute the floor for one speculative cycle (verify + k draft passes)."""
    verify_shape = StepShape(
        batch=batch,
        rows_per_seq=k + 1,
        context=context,
        state_writes=k + 1,
    )
    verify_mods = list(pass_costs(geom, par, verify_shape, dt))
    if k <= 0:
        return _combine_modules(tuple(verify_mods), chip)

    draft_geom = ModelGeometry(
        hidden=geom.hidden,
        num_layers=1,
        layer_types=("full_attention",),
        num_heads=geom.num_heads,
        num_kv_heads=geom.num_kv_heads,
        head_dim=geom.head_dim,
        attn_output_gate=geom.attn_output_gate,
        linear_k_heads=0,
        linear_v_heads=0,
        linear_k_dim=0,
        linear_v_dim=0,
        conv_kernel=geom.conv_kernel,
        num_experts=geom.num_experts,
        top_k=geom.top_k,
        moe_inter=geom.moe_inter,
        shared_inter=geom.shared_inter,
        vocab=geom.vocab,
        mtp_layers=geom.mtp_layers,
        dense_inter=geom.dense_inter,
    )
    draft_par = Parallel(tp=par.tp, kv_path="standard")
    draft_shape = StepShape(
        batch=batch, rows_per_seq=1, context=context, state_writes=1
    )
    one_draft = pass_costs(draft_geom, draft_par, draft_shape, dt)

    fc_elems = float(2 * geom.hidden * geom.hidden // par.tp)
    fc_fp8 = 2.0 * batch * fc_elems if dt.weight <= 1.0 else 0.0
    fc_bf16 = 0.0 if dt.weight <= 1.0 else 2.0 * batch * fc_elems
    draft_fc = ModuleCost(
        name="draft_fc",
        hbm_bytes=fc_elems * dt.weight * k,
        fp8_flops=fc_fp8 * k,
        bf16_flops=fc_bf16 * k,
        ici_bytes=0.0,
        n_collectives=0,
    )
    for m in one_draft:
        verify_mods.append(
            ModuleCost(
                name=f"draft_{m.name}",
                hbm_bytes=m.hbm_bytes * k,
                fp8_flops=m.fp8_flops * k,
                bf16_flops=m.bf16_flops * k,
                ici_bytes=m.ici_bytes * k,
                n_collectives=m.n_collectives * k,
            )
        )
    verify_mods.append(draft_fc)
    return _combine_modules(tuple(verify_mods), chip)
