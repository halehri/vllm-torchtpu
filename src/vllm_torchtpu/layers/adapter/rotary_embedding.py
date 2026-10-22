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
"""FP32 rotary embedding for TPU as out-of-tree CustomOps.

vLLM's CUDA rotary kernel (``csrc/pos_encoding_kernels.cu``) loads the
(activation-dtype) cos/sin cache entries and the q/k values, upcasts all of
them to ``float``, rotates, and casts once on store. The PyTorch reference,
``ApplyRotaryEmb.forward_static``, only does that when
``enable_fp32_compute=True``; the default is ``False`` and
``RotaryEmbedding.forward_static`` never passes it, so on TPU the rotation
``x1 * cos - x2 * sin`` runs in BF16. For RL rollouts the resulting drift in
q/k shows up as trainer/sampler log-prob mismatch.

Two registrations restore the CUDA numerics, following the vLLM
device-extension design (``docs/design/custom_op.md``) rather than patching
vLLM internals. ``CustomOp.__new__`` substitutes an out-of-tree class by the
upstream class name, so each upstream class that needs the fix gets one entry:

* ``ApplyRotaryEmb``: constructed with ``enable_fp32_compute=True``. Covers
  every caller that goes through an instance, in particular
  ``MRotaryEmbedding`` (Qwen3.5 / Qwen3-VL) and its subclasses.
* ``RotaryEmbedding`` and the subclasses that inherit its ``forward_native``
  (Llama3, YaRN, NTK, linear, Gemma4 scaling): ``forward_static`` calls
  ``ApplyRotaryEmb.forward_static`` directly, so the instance flag does not
  reach it. The TPU subclass replays ``forward_static`` with
  ``enable_fp32_compute=True``.

``forward_native`` is overridden as well as ``forward_oot``/``forward_tpu`` so
the numerics do not depend on ``CompilationConfig.custom_ops`` enabling the op:
CUDA runs the FP32 kernel in both cases, and so should TPU.
"""

import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.rotary_embedding import (
    DynamicNTKAlphaRotaryEmbedding,
    DynamicNTKScalingRotaryEmbedding,
    Gemma4RotaryEmbedding,
    LinearScalingRotaryEmbedding,
    Llama3RotaryEmbedding,
    NTKScalingRotaryEmbedding,
    RotaryEmbedding,
    YaRNScalingRotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@CustomOp.register_oot(name="ApplyRotaryEmb")
class TpuApplyRotaryEmb(ApplyRotaryEmb):
    """``ApplyRotaryEmb`` that always rotates in FP32, like the CUDA kernel."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["enable_fp32_compute"] = True
        super().__init__(*args, **kwargs)

    def forward_oot(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return self.forward_native(x, cos, sin)

    forward_tpu = forward_oot


def _rotary_forward_fp32(
    self: RotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``RotaryEmbedding.forward_static`` with ``enable_fp32_compute=True``.

    The cos/sin cache is still matched to the query dtype first: the CUDA
    kernel reads the activation-dtype cache and upcasts the loaded entries,
    so the rounded cache values are part of the reference numerics.
    """
    cos_sin_cache = self._match_cos_sin_cache_dtype(query)
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, self.head_size)
    query_rot = query[..., : self.rotary_dim]
    query_pass = query[..., self.rotary_dim :]
    query_rot = ApplyRotaryEmb.forward_static(
        query_rot, cos, sin, self.is_neox_style, enable_fp32_compute=True
    )
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    if key is not None:
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = ApplyRotaryEmb.forward_static(
            key_rot, cos, sin, self.is_neox_style, enable_fp32_compute=True
        )
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


# Upstream RotaryEmbedding classes whose forward is RotaryEmbedding.forward_static.
# Classes with their own forward_native (MRotaryEmbedding family, FoPE,
# DeepSeek scaling) are not listed; the MRoPE family is covered through
# TpuApplyRotaryEmb.
_STATIC_ROPE_CLASSES: tuple[type[RotaryEmbedding], ...] = (
    RotaryEmbedding,
    DynamicNTKAlphaRotaryEmbedding,
    DynamicNTKScalingRotaryEmbedding,
    Gemma4RotaryEmbedding,
    Llama3RotaryEmbedding,
    LinearScalingRotaryEmbedding,
    NTKScalingRotaryEmbedding,
    YaRNScalingRotaryEmbedding,
)


def _make_tpu_rope_class(base: type[RotaryEmbedding]) -> type[RotaryEmbedding]:
    assert "forward_native" not in base.__dict__ or base is RotaryEmbedding, (
        f"{base.__name__} defines its own forward_native; it needs its own TPU "
        "override instead of the shared RotaryEmbedding.forward_static replay."
    )
    tpu_cls = type(
        f"Tpu{base.__name__}",
        (base,),
        {
            "__module__": __name__,
            "__doc__": f"``{base.__name__}`` rotating in FP32 (CUDA numerics).",
            "forward_native": _rotary_forward_fp32,
            "forward_oot": _rotary_forward_fp32,
            "forward_tpu": _rotary_forward_fp32,
        },
    )
    return CustomOp.register_oot(name=base.__name__)(tpu_cls)


TPU_ROPE_CLASSES: dict[str, type[RotaryEmbedding]] = {
    base.__name__: _make_tpu_rope_class(base) for base in _STATIC_ROPE_CLASSES
}
TpuRotaryEmbedding = TPU_ROPE_CLASSES["RotaryEmbedding"]

logger.info(
    "Registered FP32 rotary embedding for TPU: ApplyRotaryEmb, %s.",
    ", ".join(TPU_ROPE_CLASSES),
)
