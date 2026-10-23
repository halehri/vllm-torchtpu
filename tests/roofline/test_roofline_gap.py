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
"""Unit tests for measured-vs-floor gap decomposition and CLI gap subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    ModelGeometry,
    Parallel,
    StepShape,
    compare_measured,
    step_floor,
)
from vllm_torchtpu.roofline.cli import main as cli_main

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


def test_compare_measured_qwen35_tp8_decomposition() -> None:
    sf = step_floor(
        QWEN35_GEOM,
        CHIPS["tpu7x"],
        Parallel(tp=8, kv_path="fused"),
        StepShape(batch=1, rows_per_seq=1, context=1024),
        Dtypes(weight=1.0, kv=1.0, state=2.0, lm_head=1.0),
    )
    measured = {
        "gqa": 0.080,
        "gdn": 0.250,
        "moe": 0.500,
        "lm_head": 0.050,
        "step_ms": 1.000,
    }
    rep = compare_measured(sf, CHIPS["tpu7x"], measured)
    assert rep.status == "OK"
    assert rep.unattributed_ms == pytest.approx(0.120, rel=1e-4)
    assert rep.total_gap_ms == pytest.approx(1.000 - sf.hard_floor_s * 1e3, rel=1e-4)
    shares = sum(m.gap_share for m in rep.modules)
    # Remaining share belongs to unattributed_ms (0.120 ms)
    assert 0.0 < shares < 1.0


def test_compare_measured_flags_sub_floor_model_bug() -> None:
    sf = step_floor(
        QWEN35_GEOM,
        CHIPS["tpu7x"],
        Parallel(tp=8, kv_path="fused"),
        StepShape(batch=1, rows_per_seq=1, context=1024),
        Dtypes(weight=1.0, kv=1.0, state=2.0),
    )
    # MoE floor is ~0.1748 ms; 0.100 ms is >40% below physical HBM floor
    rep = compare_measured(sf, CHIPS["tpu7x"], {"moe": 0.100, "step_ms": 0.900})
    moe_gap = next(m for m in rep.modules if m.name == "moe")
    assert moe_gap.status == "SUB_FLOOR_MODEL_BUG"
    assert rep.status == "SUB_FLOOR_MODEL_BUG"


def test_cli_gap_json_and_fail_on_sub_floor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "text_config": {
                    "hidden_size": 4096,
                    "num_hidden_layers": 60,
                    "layer_types": ["linear_attention"] * 45 + ["full_attention"] * 15,
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
        )
    )
    meas_path = tmp_path / "measured.json"
    meas_path.write_text(json.dumps({"moe": 0.050, "step_ms": 0.200}))
    rc = cli_main(
        [
            "gap",
            "--config",
            str(cfg_path),
            "--measured",
            str(meas_path),
            "--tp",
            "8",
            "--path",
            "fused",
            "--fail-on-sub-floor",
            "--json",
        ]
    )
    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "SUB_FLOOR_MODEL_BUG"

def test_compare_measured_dense_mlp_aliases() -> None:
    dense_geom = ModelGeometry(
        hidden=4096,
        num_layers=4,
        layer_types=("full_attention",) * 4,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_output_gate=False,
        linear_k_heads=0,
        linear_v_heads=0,
        linear_k_dim=0,
        linear_v_dim=0,
        conv_kernel=4,
        num_experts=0,
        top_k=0,
        moe_inter=0,
        shared_inter=0,
        vocab=32000,
        mtp_layers=0,
        dense_inter=11008,
    )
    sf = step_floor(
        dense_geom,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        StepShape(batch=1, rows_per_seq=1, context=1024),
    )
    for alias in ("mlp", "dense_mlp", "ffn"):
        rep = compare_measured(
            sf, CHIPS["tpu7x"], {"gqa": 0.050, alias: 0.100, "lm_head": 0.050, "step_ms": 0.250}
        )
        mlp_gap = next(m for m in rep.modules if m.name == "mlp")
        assert mlp_gap.status == "OK"
        assert mlp_gap.measured_ms == pytest.approx(0.100)

