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
"""Unit tests for vllm_torchtpu.roofline core accounting and sub-tile rules."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from vllm_torchtpu.roofline import (
    CHIPS,
    ChipSpec,
    Dtypes,
    Parallel,
    StepShape,
    experts_touched,
    from_hf_config,
    pass_costs,
    plan,
    spec_cycle_floor,
    step_floor,
)

QWEN35_CFG: dict[str, object] = {
    "text_config": {
        "hidden_size": 4096,
        "num_hidden_layers": 60,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 15,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "vocab_size": 248320,
        "mtp_num_hidden_layers": 1,
    }
}


def test_qwen35_batch1_bytes_match_worked_example() -> None:
    geom = from_hf_config(QWEN35_CFG)
    mods = {
        m.name: m
        for m in pass_costs(
            geom,
            Parallel(tp=8, kv_path="standard"),
            StepShape(batch=1, rows_per_seq=1, context=90036),
            Dtypes(),
        )
    }
    assert mods["gqa"].hbm_bytes == pytest.approx(220_200_960 + 1_382_952_960, abs=1.0)
    assert mods["gdn"].hbm_bytes == pytest.approx(663_828_480 + 48_015_360, abs=1.0)
    assert mods["moe"].hbm_bytes == pytest.approx(
        943_718_400 + 94_371_840 + 251_658_240, abs=1.0
    )
    assert mods["lm_head"].hbm_bytes == pytest.approx(254_279_680, abs=1.0)
    total = sum(m.hbm_bytes for m in mods.values())
    assert total == pytest.approx(3_859_025_920, abs=1.0)


def test_fused_path_splits_kv_in_group() -> None:
    geom = from_hf_config(QWEN35_CFG)
    mods = {
        m.name: m
        for m in pass_costs(
            geom,
            Parallel(tp=8, kv_path="fused"),
            StepShape(batch=1, rows_per_seq=1, context=90036),
            Dtypes(),
        )
    }
    assert mods["gqa"].hbm_bytes == pytest.approx(220_200_960 + 345_738_240, abs=1.0)
    total = sum(m.hbm_bytes for m in mods.values())
    assert total == pytest.approx(2_821_811_200, abs=1.0)


def test_hard_floor_is_max_of_resources() -> None:
    geom = from_hf_config(QWEN35_CFG)
    sf = step_floor(
        geom,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        StepShape(batch=1, rows_per_seq=1, context=90036),
        Dtypes(),
    )
    assert sf.hard_floor_s == max(sf.hbm_s, sf.mm_s, sf.ici_s)
    assert sf.binding == "hbm"
    assert sf.hbm_s == pytest.approx(3_859_025_920 / 3.7e12, rel=1e-12)
    assert sf.mm_s == pytest.approx(6.8988973870882745e-06, rel=1e-9)

    tiny_mm = ChipSpec("tiny_mm", 100, 1e18, 1e6, 1e6, 100)
    sf_mm = step_floor(
        geom, tiny_mm, Parallel(tp=8), StepShape(batch=1, context=10), Dtypes()
    )
    assert sf_mm.binding == "matmul"


def test_chip_table_matches_jax_tpu_info() -> None:
    jti = pytest.importorskip("jax._src.tpu_info")

    info = jti.get_tpu_info_for_chip(jti.ChipVersion.TPU_7X, 1)
    c = CHIPS["tpu7x"]
    assert c.hbm_bytes == info.hbm_capacity_bytes
    assert c.hbm_bw == pytest.approx(info.mem_bw_bytes_per_second)
    assert c.bf16_flops == pytest.approx(info.bf16_ops_per_second)
    assert c.fp8_flops == pytest.approx(info.fp8_ops_per_second)
    assert c.vmem_bytes == info.vmem_capacity_bytes


def test_collective_payload_and_count() -> None:
    geom = from_hf_config(QWEN35_CFG)
    mods1 = pass_costs(
        geom, Parallel(tp=8), StepShape(batch=1, rows_per_seq=1), Dtypes()
    )
    assert sum(m.n_collectives for m in mods1) == 120
    assert sum(m.ici_bytes for m in mods1) == pytest.approx(120 * 8192)
    mods4 = pass_costs(
        geom, Parallel(tp=8), StepShape(batch=1, rows_per_seq=4), Dtypes()
    )
    assert sum(m.ici_bytes for m in mods4) / 120 == pytest.approx(32768)
    mods_tp1 = pass_costs(
        geom, Parallel(tp=1), StepShape(batch=1, rows_per_seq=1), Dtypes()
    )
    assert sum(m.n_collectives for m in mods_tp1) == 0
    assert sum(m.ici_bytes for m in mods_tp1) == 0.0


def test_spec_rows_and_tokens_per_step() -> None:
    geom = from_hf_config(QWEN35_CFG)
    rows = plan(
        geom,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        Dtypes(),
        context=90036,
        concurrencies=[1, 4],
        spec_window=3,
        acceptance_length=3.39,
    )
    assert [r.rows for r in rows] == [4, 16]
    assert rows[0].tpot_floor_ms == pytest.approx(
        rows[0].hard_floor_ms / 3.39, rel=1e-12
    )
    assert experts_touched(512, 10, 4) == pytest.approx(38.84330928325653, rel=1e-12)
    cyc = spec_cycle_floor(
        geom, CHIPS["tpu7x"], Parallel(tp=8), batch=1, context=90036, k=3
    )
    assert sum(m.hbm_bytes for m in cyc.modules) == pytest.approx(
        7_813_585_288.75, abs=1.0
    )


def test_core_has_no_engine_imports() -> None:
    pkg_dir = Path(__file__).resolve().parents[2] / "src/vllm_torchtpu/roofline"
    banned = {
        "vllm",
        "torch",
        "torch_tpu",
        "jax",
        "jaxlib",
        "numpy",
        "vllm_torchtpu",
    }
    for py_file in sorted(pkg_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                assert node.module.split(".")[0] not in banned


def test_qwen38_tp32_subtile_replication() -> None:
    qwen38_cfg = {
        "text_config": {
            "hidden_size": 6144,
            "num_hidden_layers": 64,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16,
            "num_attention_heads": 64,
            "num_key_value_heads": 16,
            "head_dim": 128,
            "attn_output_gate": True,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 128,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 2048,
            "shared_expert_intermediate_size": 2048,
            "vocab_size": 248320,
        }
    }
    geom = from_hf_config(qwen38_cfg)
    mods = {
        m.name: m
        for m in pass_costs(
            geom,
            Parallel(tp=32, kv_path="standard"),
            StepShape(batch=1, rows_per_seq=1, context=1024),
            Dtypes(weight=1.0),
        )
    }
    # At tp=32: kh=1, vh=4, qkv_r=768, isr=128
    expected_gdn_w = 48 * (6144 * (768 + 512) + 6144 * 8 + 768 * 4 + 512 * 6144)
    expected_gdn_st = 2 * 48 * (4 * 128 * 128 * 4 + 768 * 3 * 2)
    assert mods["gdn"].hbm_bytes == pytest.approx(
        expected_gdn_w + expected_gdn_st, abs=1.0
    )
    ed1 = experts_touched(512, 10, 1)
    expected_moe_w = 64 * (ed1 * 3 * 6144 * 64 + 3 * 6144 * 128 + 6144 * 512 * 2)
    assert mods["moe"].hbm_bytes == pytest.approx(expected_moe_w, abs=1.0)

    # Zero-guard check: non-hybrid attention-only model with shared_inter=0
    dense_att_cfg = {
        "hidden_size": 4096,
        "num_hidden_layers": 4,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 11008,
        "vocab_size": 32000,
    }
    d_geom = from_hf_config(dense_att_cfg)
    d_mods = {
        m.name: m
        for m in pass_costs(
            d_geom, Parallel(tp=8), StepShape(batch=1, rows_per_seq=1), Dtypes()
        )
    }
    assert "gdn" not in d_mods
    assert "mlp" in d_mods


def test_non_hybrid_models_omit_absent_blocks_and_interval_layer_types() -> None:
    interval_cfg = {
        "hidden_size": 4096,
        "num_hidden_layers": 8,
        "full_attention_interval": 4,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "intermediate_size": 11008,
        "vocab_size": 32000,
    }
    geom = from_hf_config(interval_cfg)
    assert geom.layer_types.count("full_attention") == 2
    assert geom.layer_types.count("linear_attention") == 6

    pure_att_geom = from_hf_config({**interval_cfg, "full_attention_interval": 0})
    pure_att_mods = {
        m.name: m
        for m in pass_costs(pure_att_geom, Parallel(tp=8), StepShape(), Dtypes())
    }
    assert "gdn" not in pure_att_mods and "gqa" in pure_att_mods
    pure_lin_geom = from_hf_config(
        {**interval_cfg, "layer_types": ["linear_attention"] * 8}
    )
    pure_lin_mods = {
        m.name: m
        for m in pass_costs(pure_lin_geom, Parallel(tp=8), StepShape(), Dtypes())
    }
    assert "gqa" not in pure_lin_mods and "gdn" in pure_lin_mods
