# SPDX-License-Identifier: Apache-2.0
"""Debug FP8 round-trip of the HC projection weights."""

import torch

from tests.kernels.test_qwen4_exp_hc_decode import _module


def test_fake_fp8_keeps_dtype_and_bounds_error():
    module = _module(1)
    before = module.input_mix_weight_down.weight.detach().clone()
    errors = module.fake_quantize_fp8()
    after = module.input_mix_weight_down.weight
    assert after.dtype == before.dtype and after.shape == before.shape
    assert len(errors) == 3
    # e4m3 has a 3-bit mantissa: per-channel relative error is a few percent.
    assert 0 < max(errors) < 0.08
    assert not torch.equal(before, after)
