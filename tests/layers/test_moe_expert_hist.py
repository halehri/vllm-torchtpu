# SPDX-License-Identifier: Apache-2.0
"""Debug expert histograms: route() counts top-k ids, the formatter reports EP load."""

import types

import torch

from vllm_torchtpu.layers.adapter import moe_routing


def test_route_counts_topk_ids(monkeypatch):
    ids = torch.tensor([[0, 5], [5, 7], [1, 5]], dtype=torch.int32)
    monkeypatch.setattr(
        moe_routing, "_route", lambda *a: (torch.ones(3, 2), ids)
    )
    layer = torch.nn.Module()
    layer.register_buffer("expert_hist", torch.zeros(8, dtype=torch.int32))
    for _ in range(2):
        moe_routing.route(layer, torch.zeros(3, 4), torch.zeros(3, 8))
    assert layer.expert_hist.tolist() == [2, 2, 0, 0, 0, 6, 0, 2]


def test_route_without_hist_is_unchanged(monkeypatch):
    out = (torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.int32))
    monkeypatch.setattr(moe_routing, "_route", lambda *a: out)
    assert moe_routing.route(types.SimpleNamespace(), None, None) is out


def test_format_expert_load():
    hist = [2, 2, 0, 0, 0, 6, 0, 2]
    line = moe_routing.format_expert_load("3.mlp.experts", hist, 4)
    assert "picks=12" in line
    assert "ranks=[4, 0, 6, 2]" in line
    assert "max/mean=2.000" in line
    assert "hot=r2" in line
    assert line.endswith("top3=e5(r2)=0.500 e0(r0)=0.167 e1(r0)=0.167")
