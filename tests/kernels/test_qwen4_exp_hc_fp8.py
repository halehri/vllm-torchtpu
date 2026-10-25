# SPDX-License-Identifier: Apache-2.0
"""FP8-weight HC projections (VLLM_TPU_HC_FP8_WEIGHTS) against the BF16 path
on FP8-rounded weights, and their graph-input order across token buckets."""

import numpy as np
import torch
import torch.nn.functional as F

from tests.kernels.test_qwen4_exp_hc_decode import BRANCHES, _module
from vllm_torchtpu import envs

HIDDEN, LOWRANK = 128, 32
NAMES = ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")


def _fp8_module(monkeypatch, source):
    monkeypatch.setattr(envs, "VLLM_TPU_HC_FP8_WEIGHTS", True)
    module = _module(0, hidden=HIDDEN, lowrank=LOWRANK)
    monkeypatch.setattr(envs, "VLLM_TPU_HC_FP8_WEIGHTS", False)
    with torch.no_grad():
        module.hc_norm.weight.copy_(source.hc_norm.weight)
        for name in NAMES:
            param = getattr(module, name).weight
            param.weight_loader(param, getattr(source, name).weight)
    return module


def _fake_quant(source):
    module = _module(0, hidden=HIDDEN, lowrank=LOWRANK)
    with torch.no_grad():
        module.hc_norm.weight.copy_(source.hc_norm.weight)
        for name in NAMES:
            w = getattr(source, name).weight.float()
            scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
            q = (w / scale).to(torch.float8_e4m3fn).float() * scale
            getattr(module, name).weight.copy_(q.to(torch.bfloat16))
    return module


def test_fp8_weights_match_fake_quantized_bf16(monkeypatch):
    source = _module(7, hidden=HIDDEN, lowrank=LOWRANK)
    fp8, ref = _fp8_module(monkeypatch, source), _fake_quant(source)
    assert fp8.input_mix_weight_down.weight.dtype == torch.float8_e4m3fn
    torch.manual_seed(1)
    for tokens in (4, 64):
        hyper = torch.randn(tokens, BRANCHES * HIDDEN).to(torch.bfloat16)
        block = torch.randn(tokens, HIDDEN).to(torch.bfloat16)
        with torch.no_grad():
            got_mixed, got_res = fp8.mix(hyper)
            ref_mixed, ref_res = ref.mix(hyper)
            got = fp8.combine(block, got_res)
            want = ref.combine(block, ref_res)
        for a, b in ((got_mixed, ref_mixed), (got, want)):
            a, b = a.float().numpy(), b.float().numpy()
            # Only the scale's rounding point differs (after the dot vs folded
            # into BF16 weights): a BF16 ulp on a small fraction of elements.
            assert np.abs(a - b).max() <= 2.0**-6 * np.abs(b).max()


def _inputs_by_first_use(module, *args):
    names = []

    def backend(gm, example_inputs):
        position = {node: i for i, node in enumerate(gm.graph.nodes)}
        nodes = [
            n for n in gm.graph.nodes
            if n.op == "placeholder" and ("_parameters_" in n.name or "_buffers_" in n.name)
            and n.users
        ]
        nodes.sort(key=lambda n: min(position[u] for u in n.users))
        names.extend(n.name for n in nodes)
        return gm

    torch._dynamo.reset()
    with torch.no_grad():
        torch.compile(module, backend=backend, fullgraph=True, dynamic=False)(*args)
    torch._dynamo.reset()
    return names


def test_fp8_decode_and_prefill_buckets_use_inputs_in_one_order(monkeypatch):
    source = _module(9, hidden=HIDDEN, lowrank=LOWRANK)
    fp8 = _fp8_module(monkeypatch, source)

    def full_op(x, norm_w, down, up):
        normed = (x * (1.0 + norm_w)).to(torch.bfloat16)
        mixed = F.linear(F.silu(F.linear(normed, down)), up)[:, :HIDDEN]
        return mixed.to(torch.bfloat16), normed

    fp8._prefill_full_op = full_op
    fp8.prefill_full_tile = 128

    class Mix(torch.nn.Module):
        def __init__(self, hc):
            super().__init__()
            self.hc = hc

        def forward(self, x):
            mixed, (_, injection) = self.hc.mix(x)
            return mixed, injection

    decode = _inputs_by_first_use(Mix(fp8), torch.zeros(8, BRANCHES * HIDDEN, dtype=torch.bfloat16))
    prefill = _inputs_by_first_use(Mix(fp8), torch.zeros(256, BRANCHES * HIDDEN, dtype=torch.bfloat16))
    assert decode == prefill
    assert any("weight_scale" in n for n in decode)
