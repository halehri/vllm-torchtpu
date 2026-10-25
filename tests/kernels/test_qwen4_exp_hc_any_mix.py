# SPDX-License-Identifier: Apache-2.0
"""HC mix for any row count (the MTP draft's fused mix) against the GatedResidual torch path."""

import numpy as np
import pytest
import torch

from tests.kernels.test_qwen4_exp_hc_decode import (
    BRANCHES,
    EPS,
    HIDDEN,
    _check_bf16,
    _jax,
    _module,
    _weights,
)
from vllm_torchtpu.kernels.qwen4_exp import hc_decode
from vllm_torchtpu.kernels.qwen4_exp.hc_decode import hc_mix_any, hc_mix_reference


@pytest.mark.parametrize("tokens", [4, 136, 1024])
def test_reference_mix_matches_torch_path(tokens):
    module = _module(tokens)
    torch.manual_seed(200 + tokens)
    hyper = torch.randn(tokens, BRANCHES * HIDDEN).to(torch.bfloat16)
    with torch.no_grad():
        expected_mixed, residual = module.mix(hyper)
    weight, down, up, inject = _weights(module)
    mixed, injection = hc_mix_reference(
        _jax(hyper), weight, down, up, inject, branches=BRANCHES, hidden=HIDDEN, eps=EPS
    )
    _check_bf16(mixed, expected_mixed, "mixed")
    np.testing.assert_allclose(
        np.asarray(injection), residual[1].numpy(), rtol=2e-2, atol=1e-6
    )


def test_any_mix_picks_kernel_by_static_rows(monkeypatch):
    calls = []

    def fake_kernel(*args, **kwargs):
        calls.append(args[0].shape[0])
        return hc_mix_reference(*args, **kwargs) + (None,)

    monkeypatch.setattr(hc_decode, "hc_decode_mix", fake_kernel)
    module = _module(0)
    weight, down, up, inject = _weights(module)
    for tokens in (4, 64, 512, 136, 1024):
        hyper = torch.randn(tokens, BRANCHES * HIDDEN).to(torch.bfloat16)
        hc_mix_any(
            _jax(hyper), weight, down, up, inject, branches=BRANCHES, hidden=HIDDEN, eps=EPS
        )
    # 136 has no decode tile and 1024 is above the kernel's limit.
    assert calls == [4, 64, 512]


def test_any_mix_op_registers():
    """The torch custom op must build (jax_op needs annotated arguments)."""
    from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import build_any_mix_op

    assert build_any_mix_op(BRANCHES, HIDDEN, EPS, True) is not None
    assert build_any_mix_op(BRANCHES, HIDDEN, EPS, False) is not None


def test_use_any_mix_traces_one_op_for_every_count(monkeypatch):
    module = _module(5)
    seen = []

    def any_op(x, *weights):
        seen.append(x.shape[0])
        out = hc_mix_reference(
            _jax(x),
            *(_jax(w) for w in weights),
            branches=BRANCHES,
            hidden=HIDDEN,
            eps=EPS,
        )
        return (
            torch.from_numpy(np.asarray(out[0], np.float32)).to(torch.bfloat16),
            torch.from_numpy(np.asarray(out[1])),
        )

    module._any_mix_op = any_op
    module._decode_mix_op = None
    for tokens in (4, 1024):
        hyper = torch.randn(tokens, BRANCHES * HIDDEN).to(torch.bfloat16)
        with torch.no_grad():
            mixed, (_, injection) = module.mix(hyper)
        assert mixed.shape == (tokens, HIDDEN)
        assert injection.shape == (tokens, BRANCHES)
    assert seen == [4, 1024]
