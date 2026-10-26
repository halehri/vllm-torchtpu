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
"""Per-concurrency TPOT floor, interactivity ceiling, and throughput ceiling."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from .chips import ChipSpec
from .geometry import ModelGeometry
from .modules import Dtypes, Parallel, StepShape, ep_busiest_experts
from .step import spec_cycle_floor, step_floor

_FUSED_EP_TILE_M = 128
_GMM_TILE_M = 48
_ONEHOT_PERMUTE_THRESHOLD = 512
_DEFAULT_COLL_LATENCY_US = 15.4
_FP8_QUANT_OVERHEAD_S = 3.5e-6


@dataclass(frozen=True)
class ConcurrencyRow:
    """Serving ceiling row for one decode concurrency."""

    c: int
    rows: int
    binding: str
    hard_floor_ms: float
    serial_floor_ms: float
    tpot_floor_ms: float
    interactivity_ceil_tok_s: float
    output_tput_ceil_per_chip: float
    flag: str


@dataclass(frozen=True)
class BucketRegimeRow:
    """Selected residual and MoE regime for one compiled token bucket."""

    tokens: int
    node_tokens: int
    residual: str
    moe_dispatch: str
    binding: str
    hard_floor_ms: float
    serial_floor_ms: float
    flag: str = "-"


@dataclass(frozen=True)
class BucketRegimePlan:
    """Per-bucket schedule and analytical crossover thresholds."""

    sp_overlap_min_tokens: int
    fused_ep_min_node_tokens: int
    buckets: tuple[BucketRegimeRow, ...]


def plan(
    geom: ModelGeometry,
    chip: ChipSpec,
    par: Parallel,
    dt: Dtypes,
    context: int,
    concurrencies: Sequence[int],
    spec_window: int = 0,
    acceptance_length: float = 1.0,
    collective_latency_us: float | None = None,
    tensor_cores_per_chip: int = 2,
) -> list[ConcurrencyRow]:
    """Compute ConcurrencyRow ceilings across a list of concurrencies."""
    chips = par.tp / float(tensor_cores_per_chip)
    tokens = 1.0 if spec_window == 0 else float(acceptance_length)
    lat_s = (collective_latency_us or 0.0) * 1e-6
    rows_out: list[ConcurrencyRow] = []
    for c in concurrencies:
        if spec_window > 0:
            sf = spec_cycle_floor(
                geom,
                chip,
                par,
                c,
                context,
                spec_window,
                dt,
                collective_latency_s=lat_s,
            )
        else:
            sf = step_floor(
                geom,
                chip,
                par,
                StepShape(batch=c, rows_per_seq=1, context=context),
                dt,
                collective_latency_s=lat_s,
            )
        hard_ms = sf.hard_floor_s * 1e3
        serial_ms = sf.serial_floor_s * 1e3
        tpot_ms = hard_ms / tokens
        interactivity = 1000.0 / tpot_ms
        output_tput = (c * tokens) / sf.hard_floor_s / chips
        flag = "MEASURE" if sf.binding == "chain" else "-"
        rows_out.append(
            ConcurrencyRow(
                c=c,
                rows=c * (spec_window + 1),
                binding=sf.binding,
                hard_floor_ms=hard_ms,
                serial_floor_ms=serial_ms,
                tpot_floor_ms=tpot_ms,
                interactivity_ceil_tok_s=interactivity,
                output_tput_ceil_per_chip=output_tput,
                flag=flag,
            )
        )
    return rows_out


def _sp_fp8_overlap_delta_s(
    geom: ModelGeometry,
    chip: ChipSpec,
    tp: int,
    t: int,
    lat_s: float,
    ranks_per_chip: int = 2,
) -> float:
    """Per-step time delta (s) of sharded RS + FP8 AG spine vs two-shot BF16 AR."""
    if tp <= 1:
        return 0.0
    ring = (tp - 1) / tp
    rpc = getattr(chip, "ranks_per_chip", ranks_per_chip)
    offchip = max(0.0, (tp - rpc) / (tp - 1))
    wire_ar = 2.0 * (2.0 * ring * 2.0 * t * geom.hidden)
    t_ici_ar = (wire_ar * offchip) / chip.ici_bw + 4.0 * lat_s
    t_norm_gate_ar = (2.0 * 2.0 * t * geom.hidden * 2.0) / chip.hbm_bw + (
        2.0 * t * geom.hidden * geom.num_experts
    ) / chip.bf16_flops
    wire_sp = (
        2.0 * (ring * 2.0 * t * geom.hidden)
        + 2.0 * (ring * t * (geom.hidden + 2.0))
        + (ring * 2.0 * t * geom.num_experts)
    )
    t_ici_sp = (
        (wire_sp * offchip) / chip.ici_bw + 5.0 * lat_s + 2.0 * _FP8_QUANT_OVERHEAD_S
    )
    t_loc = max(1, t // tp)
    t_norm_gate_sp = (2.0 * 2.0 * t_loc * geom.hidden * 2.0) / chip.hbm_bw + (
        2.0 * t_loc * geom.hidden * geom.num_experts
    ) / chip.bf16_flops
    return geom.num_layers * ((t_ici_sp + t_norm_gate_sp) - (t_ici_ar + t_norm_gate_ar))


def _fused_ep_moe_delta_s(
    geom: ModelGeometry,
    chip: ChipSpec,
    tp: int,
    t: int,
    node_tokens: int,
    lat_s: float,
    ranks_per_chip: int = 2,
    w_expert_bytes: float = 1.0,
) -> float:
    """Per-step time delta (s) of fused_ep_moe_v2 (a2a_push) vs _gmm_ep_fallback."""
    if tp <= 1 or geom.num_experts <= 0:
        return 0.0
    e_busy = ep_busiest_experts(geom.num_experts, geom.top_k, t, tp)
    t_hbm_w = (
        e_busy * 3.0 * geom.hidden * geom.moe_inter * w_expert_bytes
    ) / chip.hbm_bw
    gmm_tiles = max(1, math.ceil((t * geom.top_k / tp) / _GMM_TILE_M))
    t_mxu_gmm = (
        2.0 * gmm_tiles * _GMM_TILE_M * 3.0 * geom.hidden * geom.moe_inter
    ) / chip.fp8_flops
    fep_pairs = max(t * geom.top_k / tp, e_busy * _FUSED_EP_TILE_M)
    t_mxu_fep = (2.0 * fep_pairs * 3.0 * geom.hidden * geom.moe_inter) / chip.fp8_flops
    t_exp_fep = max(t_hbm_w, t_mxu_fep)
    t_exp_gmm = max(t_hbm_w, t_mxu_gmm)
    routed_local = (t * geom.top_k) / tp
    gmm_inter_bytes = (
        routed_local * (2.0 * geom.hidden + 3.0 * geom.moe_inter) * 2.0
        + 2.0 * t * geom.hidden * 2.0
        + (t - t / tp) * geom.num_experts * 4.0
    )
    t_gmm_mem = gmm_inter_bytes / chip.hbm_bw
    rpc = getattr(chip, "ranks_per_chip", ranks_per_chip)
    offchip = max(0.0, (tp - rpc) / (tp - 1))
    wire_saved = (1.75 - 1.09375) * 2.0 * t * geom.hidden * offchip
    if t * geom.top_k > _ONEHOT_PERMUTE_THRESHOLD:
        coll_launch_saved_s = 5.5 * lat_s
    elif node_tokens <= tp * tp:
        coll_launch_saved_s = -1.5 * lat_s
    else:
        coll_launch_saved_s = 0.5 * lat_s
    return geom.num_layers * (
        (t_exp_fep - t_exp_gmm)
        - (wire_saved / chip.ici_bw + t_gmm_mem + coll_launch_saved_s)
    )


def recommend_bucket_regimes(
    geom: ModelGeometry,
    chip: ChipSpec,
    par: Parallel,
    dt: Dtypes,
    compile_sizes: Sequence[int],
    context: int = 0,
    collective_latency_us: float | None = None,
    ep_groups: int | None = None,
) -> BucketRegimePlan:
    """Select the lowest-serial-floor (residual, moe_dispatch) regime per bucket."""
    lat_us = (
        _DEFAULT_COLL_LATENCY_US
        if collective_latency_us is None
        else collective_latency_us
    )
    lat_s = lat_us * 1e-6
    groups = par.tp if ep_groups is None else max(1, ep_groups)
    rows_out: list[BucketRegimeRow] = []
    sp_min: int | None = None
    fep_min_node: int | None = None

    base_par = replace(par, residual="replicated", moe_dispatch="allreduce")
    for t in sorted(int(s) for s in compile_sizes):
        node_tokens = t * groups
        base_sf = step_floor(
            geom=geom,
            chip=chip,
            par=base_par,
            shape=StepShape(batch=t, rows_per_seq=1, context=context),
            dt=dt,
            collective_latency_s=lat_s,
        )
        sp_delta = (
            _sp_fp8_overlap_delta_s(
                geom, chip, par.tp, t, lat_s, ranks_per_chip=par.ranks_per_chip
            )
            if par.moe == "ep" and par.tp > 1
            else 0.0
        )
        fep_delta = (
            _fused_ep_moe_delta_s(
                geom,
                chip,
                par.tp,
                t,
                node_tokens,
                lat_s,
                ranks_per_chip=par.ranks_per_chip,
                w_expert_bytes=dt.w_expert,
            )
            if par.moe == "ep" and par.tp > 1
            else 0.0
        )
        res_win = "sp" if sp_delta < 0.0 else "replicated"
        disp_win = "a2a_push" if fep_delta < 0.0 else "allreduce"
        best_score = base_sf.serial_floor_s + min(0.0, sp_delta) + min(0.0, fep_delta)
        if res_win == "sp" and sp_min is None:
            sp_min = t
        if disp_win == "a2a_push" and fep_min_node is None:
            fep_min_node = node_tokens
        flag = "MEASURE" if base_sf.extrapolated_chain_s > 0.0 else "-"
        rows_out.append(
            BucketRegimeRow(
                tokens=t,
                node_tokens=node_tokens,
                residual=res_win,
                moe_dispatch=disp_win,
                binding=base_sf.binding,
                hard_floor_ms=base_sf.hard_floor_s * 1e3,
                serial_floor_ms=best_score * 1e3,
                flag=flag,
            )
        )

    max_t = max(compile_sizes) if compile_sizes else 8192
    return BucketRegimePlan(
        sp_overlap_min_tokens=sp_min if sp_min is not None else max_t * 2,
        fused_ep_min_node_tokens=(
            fep_min_node if fep_min_node is not None else max_t * groups * 2
        ),
        buckets=tuple(rows_out),
    )
