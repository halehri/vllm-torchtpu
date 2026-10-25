# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-native [token, branch, hidden] HyperConnections.

Reuse the common checkpoint structure with a TPU-specific SiLU precision
boundary; the NVIDIA model owns CUDA kernels and is not imported here.
"""

import torch
import torch.nn.functional as F
from vllm.models.qwen4_exp.common import hyperconnection as common_hc

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp.hc_decode import decode_token_tile

GroupedGemmaRMSNorm = common_hc.GroupedGemmaRMSNorm
HyperConnectionBase = common_hc.HyperConnectionBase
HyperConnectionConfig = common_hc.HyperConnectionConfig


class GatedResidual(common_hc.GatedResidual):
    """Match NVIDIA HC FP32 elementwise math between BF16 boundaries."""

    def __init__(
        self,
        *args,
        prefill_up_tile: int = 0,
        prefill_norm_tile: int = 0,
        prefill_full_tile: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fp8_weights = bool(envs.VLLM_TPU_HC_FP8_WEIGHTS)
        if self.fp8_weights:
            self._make_fp8_weights()
        if prefill_up_tile and (prefill_up_tile < 128 or prefill_up_tile % 128):
            raise ValueError("HC prefill up tile must be a multiple of 128")
        if prefill_norm_tile and (prefill_norm_tile < 64 or prefill_norm_tile % 64):
            raise ValueError("HC prefill norm tile must be a multiple of 64")
        if prefill_full_tile and (prefill_full_tile < 128 or prefill_full_tile % 128):
            raise ValueError("HC prefill full mix tile must be a multiple of 128")
        self.prefill_up_tile = prefill_up_tile
        self.prefill_norm_tile = prefill_norm_tile
        self.prefill_full_tile = prefill_full_tile
        self._prefill_up_op = None
        self._prefill_norm_op = None
        self._prefill_full_op = None
        self._prefill_combine_op = None
        if (
            prefill_full_tile
            and self.hidden_size % 128 == 0
            and hasattr(self, "block_inject_weight")
        ):
            from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import (
                build_gated_residual_op,
            )

            self._prefill_combine_op = build_gated_residual_op(
                self.hc_count, self.hidden_size, prefill_full_tile
            )
        if (
            prefill_full_tile
            and self.config.hc_per_branch_norm
            and hasattr(self, "input_mix_weight_down")
        ):
            from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import build_full_mix_op

            self._prefill_full_op = build_full_mix_op(
                self.hc_count,
                self.hidden_size,
                self.hc_norm.variance_epsilon,
                prefill_full_tile,
            )
        if prefill_norm_tile and self.config.hc_per_branch_norm:
            from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import build_norm_op

            self._prefill_norm_op = build_norm_op(
                self.hc_count,
                self.hidden_size,
                self.hc_norm.variance_epsilon,
                prefill_norm_tile,
            )
        self._decode_mix_op = None
        self._decode_combine_mix_op = None
        if (
            envs.TPU_QWEN4_HC_DECODE_FUSED
            and not self.fp8_weights
            and self.config.hc_per_branch_norm
            and self.hidden_size % 128 == 0
            and hasattr(self, "input_mix_weight_down")
        ):
            from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import (
                build_decode_combine_mix_op,
                build_decode_mix_op,
            )

            inject = hasattr(self, "block_inject_weight")
            eps = self.hc_norm.variance_epsilon
            self._decode_mix_op = build_decode_mix_op(
                self.hc_count, self.hidden_size, eps, inject
            )
            if inject:
                # Used by the preceding HC's combine_mix (attn -> mlp).
                self._decode_combine_mix_op = build_decode_combine_mix_op(
                    self.hc_count, self.hidden_size, eps
                )
        if prefill_up_tile and hasattr(self, "input_mix_weight_up"):
            # Register before Dynamo traces forward; registration itself uses
            # Python inspect and cannot execute inside a compiled graph.
            from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import (
                build_up_epilogue_op,
            )

            self._prefill_up_op = build_up_epilogue_op(
                self.hc_count, self.hidden_size, prefill_up_tile
            )

    _PROJECTIONS = ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")

    def _make_fp8_weights(self):
        """VLLM_TPU_HC_FP8_WEIGHTS: store the projection weights as FP8 e4m3 with
        an FP32 per-output-channel scale, quantized by the weight loader.

        Created at construction (no dtype change after loading). Every read
        goes through ``_hc_weight`` / ``_hc_linear``, so each graph takes
        (weight, scale) in the same first-use order on every token bucket.
        """
        for name in self._PROJECTIONS:
            linear = getattr(self, name, None)
            if linear is None:
                continue
            weight = torch.nn.Parameter(
                torch.empty_like(linear.weight, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            linear.register_buffer(
                "weight_scale",
                torch.ones(linear.weight.shape[0], 1, dtype=torch.float32,
                           device=linear.weight.device),
                persistent=False,
            )

            def loader(param, loaded, linear=linear):
                w = loaded.detach().to("cpu", torch.float32)
                scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
                q = (w / scale).to(torch.float8_e4m3fn)
                param.data.copy_(q.to(param.device))
                linear.weight_scale.copy_(scale.to(linear.weight_scale.device))

            weight.weight_loader = loader
            linear.weight = weight

    def _hc_weight(self, name):
        """A projection weight in BF16 (dequantized under FP8 weights)."""
        linear = getattr(self, name)
        scale = getattr(linear, "weight_scale", None)
        if scale is None:
            return linear.weight
        return (linear.weight.to(torch.float32) * scale).to(torch.bfloat16)

    def _hc_linear(self, x, name):
        """``F.linear(x, weight)``; under FP8 weights the e4m3 -> BF16 convert
        feeds the dot (exact: e4m3 values are BF16 values) and the per-output
        scale is applied to the result."""
        linear = getattr(self, name)
        scale = getattr(linear, "weight_scale", None)
        if scale is None:
            return F.linear(x, linear.weight)
        out = F.linear(x, linear.weight.to(x.dtype))
        return (out.float() * scale.reshape(-1)).to(x.dtype)

    def _normalize(self, hyper_input):
        if not self.config.hc_per_branch_norm:
            return super()._normalize(hyper_input)
        if (
            self._prefill_norm_op is not None
            and hyper_input.shape[0] >= self.prefill_norm_tile
            and hyper_input.shape[0] % self.prefill_norm_tile == 0
            and hyper_input.dtype == torch.bfloat16
            and self.hc_norm.weight.dtype == torch.bfloat16
        ):
            return self._prefill_norm_op(hyper_input, self.hc_norm.weight)

        # At prefill sizes, reshaping FP32 [tokens, HC*H] to [tokens, HC, H]
        # changes the TPU layout. Reducing contiguous branches separately
        # keeps the projection-native flat layout and fuses their variance.
        flat = hyper_input.float()
        normalized = []
        for branch in range(self.hc_count):
            start = branch * self.hidden_size
            values = flat[..., start : start + self.hidden_size]
            variance = values.square().mean(dim=-1, keepdim=True)
            normalized.append(
                values * torch.rsqrt(variance + self.hc_norm.variance_epsilon)
            )
        output = torch.cat(normalized, dim=-1)
        return (output * (1.0 + self.hc_norm.weight.float())).to(hyper_input.dtype)

    def disable_decode_fused(self):
        """Keep the unfused mix at every token count.

        For a module traced once for every token bucket (the MTP draft): the
        fused path is chosen by the token count while tracing, so that one
        graph could not serve buckets on the other side of the choice.
        """
        self._decode_mix_op = None
        self._decode_combine_mix_op = None

    def _decode_weights(self):
        weights = (
            self.hc_norm.weight,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
        )
        if hasattr(self, "block_inject_weight"):
            weights += (self.block_inject_weight.weight,)
        return weights

    def _decode_fused(self, hyper_input) -> bool:
        """TPU_QWEN4_HC_DECODE_FUSED: one Pallas call for <= 512 BF16 rows."""
        return (
            self._decode_mix_op is not None
            and hyper_input.dim() == 2
            and decode_token_tile(hyper_input.shape[0]) is not None
            and hyper_input.dtype == torch.bfloat16
            and all(w.dtype == torch.bfloat16 for w in self._decode_weights())
        )

    def _residual(self, hyper_input, norm):
        """``mix``'s residual for ``combine``, with the injection gate computed
        here as the fused decode mix computes it.

        This keeps ``block_inject_weight``'s first use inside ``mix`` on every
        path. Piecewise compilation orders a subgraph's inputs by first use, so
        a fused decode bucket and an unfused bucket of the same model then take
        the same inputs in the same order; the shape-variant hand-over refuses
        a bucket whose inputs moved (h21 j-b1210ac2).
        """
        if not hasattr(self, "block_inject_weight"):
            return hyper_input, norm
        injection = self._hc_linear(norm, "block_inject_weight")
        return hyper_input, 2.0 * torch.sigmoid(injection.float() / self.hc_count)

    def mix(self, hyper_input):
        """Returns ``mixed, (hyper_input, injection)``: the combine's [T, hc] FP32
        injection gate, computed from the normalized input (``normed`` in its
        place for a hyper-connection without a combine); ``combine`` accepts
        either form."""
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if not hasattr(self, "input_mix_weight_down"):
            raise RuntimeError("mix was disabled for this hyper-connection")
        if self._decode_fused(hyper_input):
            out = self._decode_mix_op(hyper_input, *self._decode_weights())
            if hasattr(self, "block_inject_weight"):
                mixed, injection = out
                return mixed, (hyper_input, injection)
            return out, (hyper_input, None)
        if (
            self._prefill_full_op is not None
            and hyper_input.shape[0] >= 2 * self.prefill_full_tile
            and hyper_input.shape[0] % (2 * self.prefill_full_tile) == 0
            and hyper_input.dtype == torch.bfloat16
            and self.hc_norm.weight.dtype == torch.bfloat16
            and (
                self.fp8_weights
                or (
                    self.input_mix_weight_down.weight.dtype == torch.bfloat16
                    and self.input_mix_weight_up.weight.dtype == torch.bfloat16
                )
            )
        ):
            norm_weight = self.hc_norm.weight
            if self.fp8_weights:
                # Use the norm weight before the dequantizing multiplies, so
                # this subgraph takes its inputs in the decode buckets' order.
                norm_weight = norm_weight.reshape(-1)
            mixed, norm = self._prefill_full_op(
                hyper_input,
                norm_weight,
                self._hc_weight("input_mix_weight_down"),
                self._hc_weight("input_mix_weight_up"),
            )
            return mixed, self._residual(hyper_input, norm)
        norm = self._normalize(hyper_input)
        gate = self._hc_linear(norm, "input_mix_weight_down") / self.hc_count
        # TorchTPU can lower BF16 SiLU through BF16 logistic/tanh, rounding
        # negative tails to zero. Compute the activation in FP32 and round
        # once, matching NVIDIA hc_silu's BF16 result boundary.
        gate = F.silu(gate.float()).to(gate.dtype)
        if (
            self._prefill_up_op is not None
            and gate.shape[0] >= self.prefill_up_tile
            and gate.shape[0] % self.prefill_up_tile == 0
            and gate.dtype == torch.bfloat16
            and norm.dtype == torch.bfloat16
            and (
                self.fp8_weights
                or self.input_mix_weight_up.weight.dtype == torch.bfloat16
            )
        ):
            mixed = self._prefill_up_op(gate, norm, self._hc_weight("input_mix_weight_up"))
            return mixed, self._residual(hyper_input, norm)
        gate = self._hc_linear(gate, "input_mix_weight_up")
        gate = torch.sigmoid(gate.float()).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        # Reduce the HC streams in the flat matmul-output layout. Unflattening
        # to [tokens, hc_count, hidden_size] forces an F32 layout conversion at
        # prefill sizes, while these fixed branch slices fuse into one reduction.
        weighted = gate.flatten(-2) * norm.float()
        mixed = weighted[..., : self.hidden_size]
        for branch in range(1, self.hc_count):
            start = branch * self.hidden_size
            mixed = mixed + weighted[..., start : start + self.hidden_size]
        mixed = mixed / self.hc_count
        return mixed.to(hyper_input.dtype), self._residual(hyper_input, norm)

    def combine(self, block_output, residuals):
        if not hasattr(self, "block_inject_weight"):
            raise RuntimeError("combine was disabled for this hyper-connection")
        hyper_input, hyper_input_normed = residuals
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size

        if hyper_input_normed.shape[-1] == self.hc_count:
            # The fused decode mix already produced the injection gate.
            injection = hyper_input_normed
        else:
            injection = self._hc_linear(hyper_input_normed, "block_inject_weight")
            injection = 2.0 * torch.sigmoid(injection.float() / self.hc_count)
        if (
            self._prefill_combine_op is not None
            and hyper_input.shape[0] >= 2 * self.prefill_full_tile
            and hyper_input.shape[0] % (2 * self.prefill_full_tile) == 0
            and hyper_input.dtype == torch.bfloat16
            and block_output.dtype == torch.bfloat16
        ):
            return self._prefill_combine_op(hyper_input, block_output, injection)
        # Keep each HC branch in the projection-native flat layout. The
        # three-dimensional FP32 broadcast otherwise requires a full-layout
        # copy before the next consumer at prefill sizes.
        residual = hyper_input.float()
        block = block_output.float()
        output = torch.cat(
            [
                residual[
                    ..., branch * self.hidden_size : (branch + 1) * self.hidden_size
                ]
                + block * injection[..., branch : branch + 1]
                for branch in range(self.hc_count)
            ],
            dim=-1,
        )
        return output.to(hyper_input.dtype)

    def combine_mix(self, block_output, residuals, next_hc):
        """``next_hc.mix(self.combine(block_output, residuals))``.

        After a fused decode mix, this combine runs as the prologue of
        ``next_hc``'s fused mix (one Pallas call; the combined hyper input is
        written once and returned as the next residual).
        """
        hyper_input, gate = residuals
        if (
            next_hc._decode_combine_mix_op is not None
            and gate is not None
            and gate.shape[-1] == self.hc_count
            and gate.dtype == torch.float32
            and block_output.dtype == torch.bfloat16
            and next_hc._decode_fused(hyper_input)
        ):
            mixed, injection, combined = next_hc._decode_combine_mix_op(
                hyper_input, block_output, gate, *next_hc._decode_weights()
            )
            return mixed, (combined, injection)
        return next_hc.mix(self.combine(block_output, residuals))


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
]
