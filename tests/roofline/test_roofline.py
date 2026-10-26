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
"""Unit tests for decode roofline bucket regime recommendations."""

from __future__ import annotations

from vllm_torchtpu.roofline import (
    CHIPS,
    BucketRegimePlan,
    Dtypes,
    ModelGeometry,
    Parallel,
    recommend_bucket_regimes,
)


def _qwen3_5_397b_geom() -> ModelGeometry:
    layer_types = tuple(
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(60)
    )
    return ModelGeometry(
        hidden=4096,
        num_layers=60,
        layer_types=layer_types,
        num_heads=32,
        num_kv_heads=2,
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
        mtp_layers=1,
    )


def test_recommend_bucket_regimes_crossover() -> None:
    geom = _qwen3_5_397b_geom()
    chip = CHIPS["tpu7x"]
    par = Parallel(tp=8, moe="ep")
    dt = Dtypes(w_dense=1.0, w_expert=1.0, w_aux=1.0, kv=1.0, act="a8")
    sizes = [4, 16, 64, 128, 512, 1024, 8192]
    bplan = recommend_bucket_regimes(
        geom=geom,
        chip=chip,
        par=par,
        dt=dt,
        compile_sizes=sizes,
        context=8192,
    )
    assert isinstance(bplan, BucketRegimePlan)
    assert bplan.fused_ep_min_node_tokens == 128
    assert bplan.sp_overlap_min_tokens == 1024
    by_t = {r.tokens: r for r in bplan.buckets}
    assert (by_t[4].residual, by_t[4].moe_dispatch) == (
        "replicated",
        "allreduce",
    )
    assert (by_t[16].residual, by_t[16].moe_dispatch) == (
        "replicated",
        "a2a_push",
    )
    assert (by_t[64].residual, by_t[64].moe_dispatch) == (
        "replicated",
        "a2a_push",
    )
    assert (by_t[128].residual, by_t[128].moe_dispatch) == (
        "replicated",
        "a2a_push",
    )
    assert (by_t[512].residual, by_t[512].moe_dispatch) == (
        "replicated",
        "a2a_push",
    )
    assert (by_t[1024].residual, by_t[1024].moe_dispatch) == (
        "sp",
        "a2a_push",
    )
    assert (by_t[8192].residual, by_t[8192].moe_dispatch) == (
        "sp",
        "a2a_push",
    )


def test_recommend_bucket_regimes_flags_extrapolated_chain() -> None:
    geom = _qwen3_5_397b_geom()
    par = Parallel(tp=32, moe="ep")
    dt = Dtypes(w_dense=1.0, w_expert=1.0, w_aux=1.0, kv=1.0, act="a8")
    measured_plan = recommend_bucket_regimes(
        geom=geom,
        chip=CHIPS["tpu7x-measured"],
        par=par,
        dt=dt,
        compile_sizes=[4, 64, 1024],
        context=8192,
    )
    by_t = {r.tokens: r for r in measured_plan.buckets}
    assert by_t[4].flag == "MEASURE"
    assert by_t[64].flag == "-"
    assert by_t[1024].flag == "MEASURE"

    analytic_plan = recommend_bucket_regimes(
        geom=geom,
        chip=CHIPS["tpu7x"],
        par=par,
        dt=dt,
        compile_sizes=[4, 64, 1024],
        context=8192,
    )
    assert all(r.flag == "-" for r in analytic_plan.buckets)
