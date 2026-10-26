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
"""--attention-backend flywheel: the FlyWheel TPU kernels (flywheel_tpu) in the
attention layers of non-MLA models.

Registering the name loads none of flywheel_tpu; its kernels load once the
backend is selected (layers/core/flywheel_attention.py).
"""

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import jax
import numpy as np
import torch
from torch_tpu._internal import pallas
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionLayer, MultipleOf
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.kv_cache_layout import KVCacheLayout as VllmKVCacheLayout

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    PallasAttentionBackendImpl,
)
from vllm_torchtpu.layers.adapter.attention_backend import (
    HostStepInputs,
    TPUAttentionBackend,
    register_attention_backend,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context,
)
from vllm_torchtpu.utils import synchronize_tensors

if TYPE_CHECKING:
    from vllm.config import VllmConfig

FLYWHEEL_BACKEND_NAME = "flywheel"


@dataclass(frozen=True)
class FlywheelMetadata:
    """AttentionMetadata.backend_metadata of one attention KV cache group in
    one step. Every field is a device array, so one executable per token
    bucket serves every step."""

    # i32[3 * bound]: the KV writer's (cache_row_start, new_row_start, length)
    # page pieces, and i32[1]: their count.
    kv_write_slices: torch.Tensor
    kv_write_num_slices: torch.Tensor
    # i32[1]: the step's index into flywheel_attention.step_kernels(bucket).
    kernel_index: torch.Tensor


class FlywheelAttentionBackend(TPUAttentionBackend):
    """The runner side of --attention-backend flywheel."""

    has_step_metadata: ClassVar[bool] = True

    def check_config(self, vllm_config: "VllmConfig") -> None:
        """Reject every configuration this backend cannot serve, listing each
        unsupported feature in one error."""
        from vllm_torchtpu.layers.core.flywheel_attention import (
            DECODE_HEAD_DIMS,
            PAGE_SIZE_MULTIPLE,
        )

        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        problems: list[str] = []
        if (
            cache_config.user_specified_block_size
            and cache_config.block_size % PAGE_SIZE_MULTIPLE
        ):
            problems.append(
                f"block size {cache_config.block_size} is not a multiple of "
                f"{PAGE_SIZE_MULTIPLE}; pass a --block-size that is, or leave it "
                "unset"
            )
        if model_config.use_mla:
            problems.append("MLA models are not supported")
        if model_config.is_hybrid:
            # TODO: run GDN layers on flywheel_tpu's fused conv1d + GDN kernel.
            problems.append("hybrid (linear attention) models are not supported")
        if model_config.get_head_size() not in DECODE_HEAD_DIMS:
            problems.append(
                f"head_dim {model_config.get_head_size()} is not one of the "
                f"decode kernel's {DECODE_HEAD_DIMS}"
            )
        if model_config.dtype != torch.bfloat16:
            problems.append(f"model dtype {model_config.dtype} is not bfloat16")
        if cache_config.cache_dtype not in ("auto", "bfloat16"):
            problems.append(
                f"kv cache dtype {cache_config.cache_dtype!r} is not bfloat16"
            )
        if vllm_config.speculative_config is not None:
            problems.append("speculative decoding is not supported")
        if vllm_config.kv_transfer_config is not None:
            problems.append("KV transfer (disaggregated serving) is not supported")
        if parallel_config.decode_context_parallel_size > 1:
            problems.append("decode context parallelism is not supported")
        if parallel_config.prefill_context_parallel_size > 1:
            problems.append("prefill context parallelism is not supported")
        if parallel_config.pipeline_parallel_size > 1:
            problems.append("pipeline parallelism is not supported")
        if envs.VLLM_TPU_BLOCK_MAJOR_KV:
            problems.append("VLLM_TPU_BLOCK_MAJOR_KV is not supported; unset it")
        if problems:
            raise ValueError(
                "The flywheel attention backend cannot serve this configuration: "
                + "; ".join(problems)
                + "."
            )

    def prepare_step(
        self, step: HostStepInputs, device: torch.device
    ) -> Mapping[int, FlywheelMetadata]:
        """Pick the step's kernel and build each attention group's KV write
        slice table. The decode kernel appends the leading decode rows itself,
        so the writer takes only the rows after them."""
        from vllm_torchtpu.layers.core import flywheel_attention as fw

        kernel_index = fw.plan_step_kernel(
            step.num_decode, step.query_start_loc, step.seq_lens, step.num_tokens
        )
        kernel_index_dev = torch.tensor([kernel_index], dtype=torch.int32).to(device)
        metadata = {}
        for gid, page_size in step.page_sizes.items():
            # Sized by the largest bucket: the table's shape must not change
            # with the step's bucket (HostStepInputs.max_num_tokens).
            bound = fw.kv_write_slice_bound(
                step.max_num_tokens, step.max_num_seqs, page_size
            )
            slices = np.zeros((3, bound), dtype=np.int32)
            num_slices = fw.build_kv_write_slices(
                step.query_start_loc[step.num_decode :],
                step.seq_lens[step.num_decode :],
                step.block_tables[gid][step.num_decode :],
                page_size=page_size,
                out=slices,
            )
            metadata[gid] = FlywheelMetadata(
                kv_write_slices=torch.from_numpy(slices.reshape(-1)).to(device),
                kv_write_num_slices=torch.tensor([num_slices], dtype=torch.int32).to(
                    device
                ),
                kernel_index=kernel_index_dev,
            )
        return metadata

    def warmup_metadata(
        self,
        page_sizes: Mapping[int, int],
        *,
        max_num_seqs: int,
        max_num_tokens: int,
        device: torch.device,
    ) -> Mapping[int, FlywheelMetadata]:
        """All zeros: the writer copies nothing and the kernel index picks the
        decode-only branch."""
        from vllm_torchtpu.layers.core import flywheel_attention as fw

        kernel_index = torch.zeros((1,), dtype=torch.int32).to(device)
        return {
            gid: FlywheelMetadata(
                kv_write_slices=torch.zeros(
                    (
                        3
                        * fw.kv_write_slice_bound(
                            max_num_tokens, max_num_seqs, page_size
                        ),
                    ),
                    dtype=torch.int32,
                ).to(device),
                kv_write_num_slices=torch.zeros((1,), dtype=torch.int32).to(device),
                kernel_index=kernel_index,
            )
            for gid, page_size in page_sizes.items()
        }


def _flywheel_kernel_entry(
    kv_cache: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    kv_write_slices: jax.Array,
    kv_write_num_slices: jax.Array,
    kernel_index: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    sm_scale: float,
) -> tuple[jax.Array, jax.Array]:
    """The jax_op body. Full signature (not `*args, **kwargs`), which
    `pallas.jax_op` introspects."""
    from vllm_torchtpu.layers.core.flywheel_attention import flywheel_attention

    return flywheel_attention(
        kv_cache,
        query,
        key,
        value,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        kv_write_slices,
        kv_write_num_slices,
        kernel_index,
        mesh=mesh,
        sm_scale=sm_scale,
    )


class FlywheelPallasAttentionBackendImpl(PallasAttentionBackendImpl):
    """The attention impl of --attention-backend flywheel."""

    _kernel_op_prefix: ClassVar[str] = "pallas::flywheel_attention"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        unsupported = {
            "attention sinks": self.sinks is not None,
            "sliding window attention": self.sliding_window is not None,
            "a logits soft cap": bool(self.logits_soft_cap),
            "a quantized KV cache": self.kv_cache_quantized_dtype is not None,
            "KV sharing across layers": self.kv_sharing_target_layer_name is not None,
            "non-causal attention": not self.use_causal_mask,
        }
        for feature, is_unsupported in unsupported.items():
            if is_unsupported:
                raise NotImplementedError(
                    f"The flywheel attention backend does not support {feature}."
                )

    def initialize_kernel(self, layer: AttentionLayer) -> None:
        """Build the kernel op before torch.compile traces the model."""
        mesh = get_vllm_model_wrapper_context().mesh
        key = (self._kernel_op_prefix, self.scale, id(mesh))
        kernel = self._kernel_registry.get(key)
        if kernel is None:
            kernel = self._build_flywheel_kernel(mesh)
            self._kernel_registry[key] = kernel
        self.rpa_kernel = kernel

    def _build_flywheel_kernel(self, mesh: jax.sharding.Mesh):
        op_name = f"{self._kernel_op_prefix}_{self._allocate_kernel_instance_id()}"
        # As for RPA: donating the cache is only safe when the op is compiled,
        # not when kernel-iteration mode runs it eagerly.
        op = pallas.jax_op(
            op_name,
            functools.partial(_flywheel_kernel_entry, mesh=mesh, sm_scale=self.scale),
            donate_argnums=None if envs.TPU_KERNEL_ITER_MODE else (0,),
            mesh=None,
            input_partition_specs=None,
        )

        def _fake(kv_cache: torch.Tensor, query: torch.Tensor, *args, **kwargs):
            return torch.empty_like(kv_cache), torch.empty_like(query)

        op.register_fake(_fake)

        def kernel(kv_cache: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
            new_kv_cache, output = op(kv_cache, *args)
            # Donation + copy_ writeback compiles to an in-place pool update.
            kv_cache.copy_(new_kv_cache)
            return output

        return kernel

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "The flywheel attention backend does not support fused output "
                "quantization."
            )
        # The memory profiling pass runs before the KV cache exists.
        if kv_cache.numel() == 0:
            return torch.ones_like(query) if output is None else output
        flywheel_metadata = attn_metadata.backend_metadata
        if not isinstance(flywheel_metadata, FlywheelMetadata):
            raise ValueError(
                "The flywheel attention backend needs its step metadata "
                "(AttentionMetadata.backend_metadata, from "
                "FlywheelAttentionBackend.prepare_step); got "
                f"{type(flywheel_metadata).__name__}."
            )
        num_tokens = query.shape[0]
        query_dim = query.dim()
        query = query.view(num_tokens, self.num_heads, self.head_size)
        key = key.view(num_tokens, self.num_kv_heads, self.head_size)
        value = value.view(num_tokens, self.num_kv_heads, self.head_size)
        assert self.rpa_kernel is not None, "initialize_kernel was not called"
        outputs = self.rpa_kernel(
            kv_cache,
            query,
            key,
            value,
            attn_metadata.seq_lens,
            attn_metadata.block_tables,
            attn_metadata.query_start_loc,
            attn_metadata.request_distribution,
            flywheel_metadata.kv_write_slices,
            flywheel_metadata.kv_write_num_slices,
            flywheel_metadata.kernel_index,
        )
        if not torch.compiler.is_compiling():
            synchronize_tensors(kv_cache, wait=False)
        if query_dim == 2:
            outputs = outputs.reshape(num_tokens, self.num_heads * self.head_size)
        if output is not None:
            output.copy_(outputs)
        return outputs


class FlywheelPallasAttentionBackend(PallasAttentionBackend):
    """The vLLM attention backend of --attention-backend flywheel. Its bf16
    pages are RPA v3's, which the kernels read as FlyWheel's merged K/V rows."""

    supported_kv_cache_dtypes = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return FLYWHEEL_BACKEND_NAME.upper()

    @staticmethod
    def get_impl_cls() -> type[FlywheelPallasAttentionBackendImpl]:
        return FlywheelPallasAttentionBackendImpl

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[VllmKVCacheLayout, ...]:
        return (VllmKVCacheLayout.LBNHC,)

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec: KVCacheSpec | None = None,
    ) -> list[int | MultipleOf]:
        from vllm_torchtpu.layers.core.flywheel_attention import PAGE_SIZE_MULTIPLE

        return [MultipleOf(PAGE_SIZE_MULTIPLE)]

    @staticmethod
    def get_page_size(vllm_config: "VllmConfig") -> int:
        """RPA's page size, rounded up to whole pages the kernels can index."""
        from vllm_torchtpu.layers.core.flywheel_attention import PAGE_SIZE_MULTIPLE

        page_size = PallasAttentionBackend.get_page_size(vllm_config)
        return cdiv(page_size, PAGE_SIZE_MULTIPLE) * PAGE_SIZE_MULTIPLE


def register_flywheel_attention_backend() -> None:
    """Make --attention-backend flywheel run the FlyWheel kernels. Called from
    the register_layers vllm.general_plugins entry point."""
    register_attention_backend(
        FLYWHEEL_BACKEND_NAME,
        attention=FlywheelAttentionBackend,
        vllm_backend=f"{__name__}.{FlywheelPallasAttentionBackend.__qualname__}",
    )
