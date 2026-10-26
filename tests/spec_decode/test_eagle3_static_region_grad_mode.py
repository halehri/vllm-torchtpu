# SPDX-License-Identifier: Apache-2.0
"""The draft's static compiled regions must not recompile across grad modes.

Dynamo guards a compiled entry on the global grad mode, and the eagle3 draft
regions are reached both under execute_model's no_grad and outside it
(sample_tokens, the precompile). Served post-warmup recompiles from
eagle3.py's ``_static_shape_args`` wrapper looked exactly like that.
"""

import torch
from torch._dynamo.testing import CompileCounter

from vllm_torchtpu.spec_decode.eagle3 import _static_shape_args


def _calls(fn):
    x = torch.arange(8, dtype=torch.float32)
    fn(x)  # grad enabled, as in sample_tokens / the precompile
    with torch.no_grad():  # as in execute_model
        fn(x)


def test_one_compile_across_grad_modes():
    torch._dynamo.reset()
    counter = CompileCounter()

    @_static_shape_args
    @torch.compile(backend=counter, fullgraph=True, dynamic=False)
    def region(x):
        return x * 2 + 1

    _calls(region)
    assert counter.frame_count == 1


def test_without_the_wrapper_grad_mode_recompiles():
    # The guard this wrapper neutralizes; if this ever stops recompiling, the
    # first test no longer proves anything.
    torch._dynamo.reset()
    counter = CompileCounter()

    @torch.compile(backend=counter, fullgraph=True, dynamic=False)
    def region(x):
        return x * 2 + 1

    _calls(region)
    assert counter.frame_count == 2
