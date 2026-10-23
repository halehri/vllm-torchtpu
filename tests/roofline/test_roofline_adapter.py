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
"""Unit tests for Megakernel Plan/Profile adapter and standalone reconciliation."""

from __future__ import annotations

import ast
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from vllm_torchtpu.megakernel.roofline import (
    floor_for_profile,
    floor_from_plan,
    reconcile_plan_vs_standalone,
)
from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    ModelGeometry,
    Parallel,
    StepShape,
)


@dataclass(frozen=True)
class _FakeWeightDecl:
    name: str
    owner: str
    K: int
    N: int
    dtype: str
    stream_ring: str = "static"
    lead_dims: tuple[int, ...] = ()


@dataclass(frozen=True)
class _FakeKVCacheDecl:
    hd: int
    kvh: int
    page_size: int
    page_dtype: str


@dataclass(frozen=True)
class _FakeRecurrentCacheDecl:
    kh: int
    vh: int
    KD: int
    VD: int
    rec_dtype: str
    conv_taps: int = 3
    conv_dtype: str = "bfloat16"


@dataclass(frozen=True)
class _FakeBlockSpec:
    name: str
    reduction: str | None
    cache_decl: object = None


@dataclass(frozen=True)
class _FakePlan:
    block_layer_counts: dict[str, int]
    weight_decls: tuple[_FakeWeightDecl, ...]
    block_specs: tuple[_FakeBlockSpec, ...]
    smalls_shape: tuple[tuple[int, int, int], str]


@dataclass(frozen=True)
class _FakeExecSpec:
    B: int = 1
    T: int = 1
    R: int = 8
    ctx: int = 1024


QWEN35_GEOM = ModelGeometry(
    hidden=4096,
    num_layers=60,
    layer_types=tuple(["linear_attention"] * 45 + ["full_attention"] * 15),
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

QWEN35_TP8_PLAN = _FakePlan(
    block_layer_counts={"gdn": 45, "gqa": 15, "moe": 60},
    weight_decls=(
        _FakeWeightDecl("lin_in", "gdn", 4096, 2688, "float8_e4m3fn"),
        _FakeWeightDecl("lin_out", "gdn", 1024, 4096, "float8_e4m3fn"),
        _FakeWeightDecl("att_in", "gqa", 4096, 2560, "float8_e4m3fn"),
        _FakeWeightDecl("att_out", "gqa", 1024, 4096, "float8_e4m3fn"),
        _FakeWeightDecl("router", "moe", 4096, 512, "bfloat16"),
        _FakeWeightDecl("sh13", "moe", 4096, 256, "float8_e4m3fn"),
        _FakeWeightDecl("sh2", "moe", 128, 4096, "float8_e4m3fn"),
        _FakeWeightDecl("e13", "moe", 4096, 256, "float8_e4m3fn", "dynamic", (512,)),
        _FakeWeightDecl("e2", "moe", 128, 4096, "float8_e4m3fn", "dynamic", (512,)),
        _FakeWeightDecl("es", "moe", 1, 4352, "float32", "hbm", (512,)),
    ),
    block_specs=(
        _FakeBlockSpec(
            "gdn",
            "allreduce",
            _FakeRecurrentCacheDecl(kh=2, vh=8, KD=128, VD=128, rec_dtype="bfloat16"),
        ),
        _FakeBlockSpec(
            "gqa",
            "allreduce",
            _FakeKVCacheDecl(hd=256, kvh=1, page_size=128, page_dtype="float8_e4m3fn"),
        ),
        _FakeBlockSpec("moe", "allreduce", None),
    ),
    smalls_shape=((60, 1, 30464), "float32"),
)


def test_qwen35_tp8_reconciles_to_zero_without_plan_extras() -> None:
    ex = _FakeExecSpec(B=1, T=1, R=8, ctx=1024)
    rows = reconcile_plan_vs_standalone(
        QWEN35_TP8_PLAN,
        ex,
        QWEN35_GEOM,
        Parallel(tp=8, kv_path="fused"),
        StepShape(batch=1, rows_per_seq=1, context=1024),
        Dtypes(weight=1.0, kv=1.0, state=2.0, conv=2.0),
        include_plan_extras=False,
    )
    for r in rows:
        assert r.delta_hbm_bytes == pytest.approx(0.0, abs=1e-6), r
        assert r.delta_flops == pytest.approx(0.0, abs=1e-6), r


def test_qwen35_tp8_plan_extras_match_ba_pad_es_and_smalls() -> None:
    ex = _FakeExecSpec(B=1, T=1, R=8, ctx=1024)
    sf = floor_from_plan(
        QWEN35_TP8_PLAN,
        ex,
        chip=CHIPS["tpu7x"],
        kv_path="fused",
        include_plan_extras=True,
    )
    mod_map = {m.name: m for m in sf.modules}
    assert mod_map["gqa"].hbm_bytes == pytest.approx(222_167_040.0, rel=1e-6)
    assert mod_map["gdn"].hbm_bytes == pytest.approx(708_618_240.0, rel=1e-6)
    assert mod_map["moe"].hbm_bytes == pytest.approx(1_300_193_280.0, rel=1e-6)
    assert mod_map["smalls"].hbm_bytes == pytest.approx(7_311_360.0, rel=1e-6)


def test_megakernel_roofline_wrapper_passes_portability_ast_rules() -> None:
    import vllm_torchtpu.megakernel.roofline as mk_roof

    tree = ast.parse(Path(mk_roof.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            raise AssertionError(f"Absolute import forbidden in megakernel: {node}")
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            assert node.module == "__future__"

PLAN_128E = _FakePlan(
    block_layer_counts={"gqa": 2, "moe": 2},
    weight_decls=(
        _FakeWeightDecl("att_in", "gqa", 2048, 1024, "float8_e4m3fn"),
        _FakeWeightDecl("att_out", "gqa", 512, 2048, "float8_e4m3fn"),
        _FakeWeightDecl("router", "moe", 2048, 128, "bfloat16"),
        _FakeWeightDecl("e13", "moe", 2048, 512, "float8_e4m3fn", "dynamic", (16,)),
        _FakeWeightDecl("e2", "moe", 256, 2048, "float8_e4m3fn", "dynamic", (16,)),
    ),
    block_specs=(
        _FakeBlockSpec(
            "gqa",
            "allreduce",
            _FakeKVCacheDecl(hd=128, kvh=1, page_size=128, page_dtype="float8_e4m3fn"),
        ),
        _FakeBlockSpec("moe", "allreduce", None),
    ),
    smalls_shape=((2, 1, 1024), "float32"),
)


def test_floor_from_plan_infers_non_qwen_expert_and_head_counts() -> None:
    ex = _FakeExecSpec(B=1, T=1, R=8, ctx=1024)
    sf_inferred = floor_from_plan(PLAN_128E, ex, top_k=8, include_plan_extras=False)
    sf_explicit = floor_from_plan(
        PLAN_128E,
        ex,
        q_heads_global=32,
        num_experts_global=128,
        top_k=8,
        include_plan_extras=False,
    )
    assert sf_inferred.hbm_s == pytest.approx(sf_explicit.hbm_s, rel=1e-12)
    assert sf_inferred.mm_s == pytest.approx(sf_explicit.mm_s, rel=1e-12)


def test_floor_for_profile_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_exec_mod = types.ModuleType("vllm_torchtpu.megakernel.exec_spec")
    fake_exec_mod.__dict__["ExecSpec"] = _FakeExecSpec
    fake_plan_mod = types.ModuleType("vllm_torchtpu.megakernel.planner")
    fake_plan_mod.__dict__["plan"] = lambda model_spec, ex, fit_vmem=False: PLAN_128E
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.megakernel.exec_spec", fake_exec_mod)
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.megakernel.planner", fake_plan_mod)

    class _FakeProfile:
        R = 8

        def exec_for(self, base: _FakeExecSpec, B: int, T: int) -> _FakeExecSpec:
            return _FakeExecSpec(B=B, T=T, R=base.R, ctx=base.ctx)

    sf = floor_for_profile(
        object(), _FakeProfile(), B=1, S=1024, T=1, top_k=8, include_plan_extras=False
    )
    sf_explicit = floor_from_plan(
        PLAN_128E,
        _FakeExecSpec(B=1, T=1, R=8, ctx=1024),
        q_heads_global=32,
        num_experts_global=128,
        top_k=8,
        include_plan_extras=False,
    )
    assert sf.hbm_s == pytest.approx(sf_explicit.hbm_s, rel=1e-12)
    assert sf.mm_s == pytest.approx(sf_explicit.mm_s, rel=1e-12)

