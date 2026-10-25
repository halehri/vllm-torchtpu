# SPDX-License-Identifier: Apache-2.0
"""The grouped HC branch norm against the per-branch reductions."""

import pytest
import torch

from tests.kernels.test_qwen4_exp_hc_decode import BRANCHES, HIDDEN, _module
from vllm_torchtpu import envs


@pytest.mark.parametrize("tokens", [4, 24, 136])
def test_grouped_norm_matches_per_branch(monkeypatch, tokens):
    module = _module(tokens)
    torch.manual_seed(300 + tokens)
    hyper = (torch.randn(tokens, BRANCHES * HIDDEN) * 3).to(torch.bfloat16)
    with torch.no_grad():
        expected = module._normalize(hyper)
        monkeypatch.setattr(envs, "VLLM_TPU_HC_NORM_GROUPED", True)
        got = module._normalize(hyper)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
