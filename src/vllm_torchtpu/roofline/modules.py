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
"""Per-module HBM byte, MXU FLOP, and collective payload equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .geometry import ModelGeometry


@dataclass(frozen=True)
class Dtypes:
    """Element widths in bytes for each tensor role."""

    weight: float = 1.0
    kv: float = 2.0
    state: float = 4.0
    conv: float = 2.0
    router: float = 2.0
    lm_head: float = 2.0


@dataclass(frozen=True)
class Parallel:
    """Tensor-parallel rank count and KV group layout."""

    tp: int
    kv_path: Literal["standard", "fused"] = "standard"


@dataclass(frozen=True)
class StepShape:
    """Per-step batch size, token rows per sequence, context, and state writes."""

    batch: int = 1
    rows_per_seq: int = 1
    context: int = 0
    state_writes: int = 1


@dataclass(frozen=True)
class ModuleCost:
    """Per-TensorCore resource work for one module class across one pass."""

    name: str
    hbm_bytes: float
    fp8_flops: float
    bf16_flops: float
    ici_bytes: float
    n_collectives: int


def experts_touched(num_experts: int, top_k: int, rows: int) -> float:
    """Expected distinct routed experts touched under uniform routing."""
    if num_experts <= 0 or top_k <= 0 or rows <= 0:
        return 0.0
    if top_k >= num_experts:
        return float(num_experts)
    return float(num_experts) * (1.0 - (1.0 - top_k / num_experts) ** rows)


def _split_matmul_flops(
    rows: int, weight_elems: float, weight_bytes: float
) -> tuple[float, float]:
    """Route quantized (<=1 B) matmuls to FP8 peak and wider matmuls to bf16 peak."""
    flops = 2.0 * rows * weight_elems
    if weight_bytes <= 1.0:
        return (flops, 0.0)
    return (0.0, flops)


def pass_costs(
    geom: ModelGeometry,
    par: Parallel,
    shape: StepShape,
    dt: Dtypes,
) -> tuple[ModuleCost, ...]:
    """Compute per-module resource work for one forward pass."""
    tp = par.tp
    if tp <= 0:
        raise ValueError(f"Parallel.tp must be >= 1, got {tp}")
    rows = shape.batch * shape.rows_per_seq
    n_coll_per_layer = 1 if tp > 1 else 0
    ar_payload = float(rows * geom.hidden * 2) if tp > 1 else 0.0

    n_att = sum(1 for t in geom.layer_types if t == "full_attention")
    n_gdn = sum(1 for t in geom.layer_types if t == "linear_attention")

    kvh = max(1, geom.num_kv_heads // tp) if geom.num_kv_heads > 0 else 0
    q_mult = 2 if geom.attn_output_gate else 1
    q_out = geom.num_heads * geom.head_dim * q_mult
    att_w_elems = float(
        n_att
        * (
            geom.hidden * (q_out // tp)
            + 2 * geom.hidden * kvh * geom.head_dim
            + (geom.num_heads // tp) * geom.head_dim * geom.hidden
        )
    )
    kv_group = max(1, tp // geom.num_kv_heads) if geom.num_kv_heads > 0 else 1
    kv_div = kv_group if par.kv_path == "fused" else 1
    kv_bytes = (
        float(shape.batch * kvh * geom.head_dim * 2 * n_att * shape.context)
        * dt.kv
        / kv_div
    )
    att_fp8, att_bf16 = _split_matmul_flops(rows, att_w_elems, dt.weight)
    att_scores = float(
        rows * 4 * shape.context * geom.head_dim * (geom.num_heads // tp) * n_att
    )
    out: list[ModuleCost] = []
    if n_att > 0:
        out.append(
            ModuleCost(
                name="gqa",
                hbm_bytes=att_w_elems * dt.weight + kv_bytes,
                fp8_flops=att_fp8,
                bf16_flops=att_bf16 + att_scores,
                ici_bytes=ar_payload * n_att,
                n_collectives=n_att * n_coll_per_layer,
            )
        )

    kh = max(1, geom.linear_k_heads // tp) if geom.linear_k_heads > 0 else 0
    vh = max(1, geom.linear_v_heads // tp) if geom.linear_v_heads > 0 else 0
    qkv_r = 2 * kh * geom.linear_k_dim + vh * geom.linear_v_dim
    z_r = vh * geom.linear_v_dim
    gdn_w_elems = float(
        n_gdn
        * (
            geom.hidden * (qkv_r + z_r)
            + geom.hidden * (2 * vh)
            + qkv_r * geom.conv_kernel
            + z_r * geom.hidden
        )
    )
    rec_state = float(n_gdn * vh * geom.linear_k_dim * geom.linear_v_dim) * dt.state
    conv_state = float(n_gdn * qkv_r * max(0, geom.conv_kernel - 1)) * dt.conv
    gdn_state_bytes = float(shape.batch * (1 + shape.state_writes)) * (
        rec_state + conv_state
    )
    gdn_fp8, gdn_bf16 = _split_matmul_flops(rows, gdn_w_elems, dt.weight)
    if n_gdn > 0:
        out.append(
            ModuleCost(
                name="gdn",
                hbm_bytes=gdn_w_elems * dt.weight + gdn_state_bytes,
                fp8_flops=gdn_fp8,
                bf16_flops=gdn_bf16,
                ici_bytes=ar_payload * n_gdn,
                n_collectives=n_gdn * n_coll_per_layer,
            )
        )

    if geom.num_experts > 0:
        ed = experts_touched(geom.num_experts, geom.top_k, rows)
        routed_elems = (
            float(geom.num_layers) * ed * 3.0 * geom.hidden * (geom.moe_inter // tp)
        )
        isr = max(128, geom.shared_inter // tp) if geom.shared_inter > 0 else 0
        shared_elems = float(geom.num_layers * 3 * geom.hidden * isr)
        router_elems = float(geom.num_layers * geom.hidden * geom.num_experts)
        moe_w_bytes = (
            routed_elems + shared_elems
        ) * dt.weight + router_elems * dt.router
        act_elems = float(
            geom.num_layers
            * (
                geom.top_k * 3 * geom.hidden * (geom.moe_inter // tp)
                + 3 * geom.hidden * isr
            )
        )
        moe_fp8, moe_bf16 = _split_matmul_flops(rows, act_elems, dt.weight)
        router_flops = 2.0 * rows * router_elems
        out.append(
            ModuleCost(
                name="moe",
                hbm_bytes=moe_w_bytes,
                fp8_flops=moe_fp8,
                bf16_flops=moe_bf16 + router_flops,
                ici_bytes=ar_payload * geom.num_layers,
                n_collectives=geom.num_layers * n_coll_per_layer,
            )
        )
    else:
        mlp_inter = geom.dense_inter // tp
        mlp_elems = float(geom.num_layers * 3 * geom.hidden * mlp_inter)
        mlp_fp8, mlp_bf16 = _split_matmul_flops(rows, mlp_elems, dt.weight)
        out.append(
            ModuleCost(
                name="mlp",
                hbm_bytes=mlp_elems * dt.weight,
                fp8_flops=mlp_fp8,
                bf16_flops=mlp_bf16,
                ici_bytes=ar_payload * geom.num_layers,
                n_collectives=geom.num_layers * n_coll_per_layer,
            )
        )

    lm_elems = float(geom.vocab * geom.hidden // tp)
    out.append(
        ModuleCost(
            name="lm_head",
            hbm_bytes=lm_elems * dt.lm_head,
            fp8_flops=0.0,
            bf16_flops=2.0 * rows * lm_elems,
            ici_bytes=0.0,
            n_collectives=0,
        )
    )
    return tuple(out)
