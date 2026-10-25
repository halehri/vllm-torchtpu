# SPDX-License-Identifier: Apache-2.0
"""An MTP draft's loop steps 1..K-1 as one compiled program.

Each loop step used to be two host dispatches -- the draft forward and the
lm-head + argmax -- plus a metadata build, so K=3 paid for four programs and
two builds between step 0 and the verify. The steps differ only in values the
device can derive itself: positions and attention seq_lens advance by one, and
the input ids are the previous step's argmax. This program unrolls the steps
and advances both on device, so the host builds step 1's metadata once and
dispatches once.

seq_lens reaches the attention kernel through the forward context, not as an
argument, so the program is handed the same tensor the step-1 metadata holds
and bumps it in place between steps; the traced attention then reads the
bumped value. The caller must pass a tensor built for this call alone.

This lives outside eagle3.py because that module uses postponed annotations,
which ``support_torch_compile`` cannot read.
"""

import inspect
from collections.abc import Callable
from typing import Any, Protocol

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors

from vllm_torchtpu import envs


class MtpDraftModel(Protocol):
    """What the loop calls on the MTP draft. The draft is an nn.Module, but
    nn.Module types its attributes as Tensor | Module, so name the calls."""

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...


def sharded_greedy_argmax(
    local_logits: torch.Tensor,
    vocab_offset: torch.Tensor,
    org_vocab_size: int,
    tp_size: int,
    all_gather: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """argmax over a vocab-parallel head's full logits, from this rank's shard.

    `local_logits` is [rows, vocab/tp] for the vocab rows starting at
    `vocab_offset` (int32 [1], a runtime value so every rank traces the same
    program). Padding past `org_vocab_size` never wins. Ties resolve to the
    lowest vocab index, as torch.argmax over the gathered logits would: the
    first max within a shard, then the first rank holding it.
    """
    vocab = torch.arange(local_logits.shape[-1], device=local_logits.device)
    vocab = vocab.to(torch.int32) + vocab_offset
    logits = torch.where(vocab < org_vocab_size, local_logits, -float("inf"))
    values, index = logits.max(dim=-1)
    index = index.to(torch.int32) + vocab_offset
    if tp_size == 1:
        return index
    # float32 holds every vocab index exactly (< 2**24).
    pair = torch.stack([values.float(), index.float()], dim=-1)
    gathered = all_gather(pair).view(local_logits.shape[0], tp_size, 2)
    best = gathered[..., 0].argmax(dim=-1, keepdim=True)
    return gathered[..., 1].gather(-1, best).squeeze(-1).to(torch.int32)


class _MtpFusedLoop(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        mtp: MtpDraftModel,
        num_steps: int,
    ) -> None:
        super().__init__()
        # Own compile-cache folder, as in MtpLoopStepModel.
        self.prefix = prefix
        self.mtp = mtp
        self.num_steps = num_steps
        self.sharded_argmax = self._sharded_argmax_head() is not None
        if self.sharded_argmax:
            head = mtp.lm_head
            # A buffer, not a Python int: the offset differs per rank and the
            # program must not (peer replay, compilation/tpu_compiler.py).
            self.register_buffer(
                "vocab_offset",
                torch.tensor(
                    [head.shard_indices.org_vocab_start_index],
                    dtype=torch.int32,
                    device=head.weight.device,
                ),
                persistent=False,
            )

    def _sharded_argmax_head(self):
        """The draft's vocab-parallel head, when the sharded argmax applies."""
        # Opt-in per draft: a draft whose compute_logits picks a head per
        # step (spec_step_idx) must keep the full-logits path.
        if not envs.VLLM_TPU_MTP_SHARDED_ARGMAX or not getattr(
            type(self.mtp), "sharded_argmax_head", False
        ):
            return None
        head = getattr(self.mtp, "lm_head", None)
        processor = getattr(self.mtp, "logits_processor", None)
        if head is None or processor is None or getattr(head, "tp_size", 1) <= 1:
            return None
        if processor.logits_as_input or processor.scale <= 0.0:
            return None
        return head

    def _greedy_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.sharded_argmax:
            return self.mtp.compute_logits(hidden_states).argmax(dim=-1).to(torch.int32)
        from vllm.distributed import tensor_model_parallel_all_gather

        head = self.mtp.lm_head
        processor = self.mtp.logits_processor
        logits = processor._apply_head(head, hidden_states, None)
        if processor.soft_cap is not None:
            logits = torch.tanh(logits / processor.soft_cap) * processor.soft_cap
        if processor.scale != 1.0:
            logits = logits * processor.scale
        return sharded_greedy_argmax(
            logits,
            self.vocab_offset,
            processor.org_vocab_size,
            head.tp_size,
            lambda x: tensor_model_parallel_all_gather(x, dim=-1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        seq_lens: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """`positions` and `seq_lens` are step 1's. Returns the K-1 draft
        tokens, one [p] int32 tensor per step."""
        tokens = []
        for i in range(self.num_steps):
            if i > 0:
                positions = positions + 1
                seq_lens.add_(1)
            out = type(self.mtp).forward(
                self.mtp,
                input_ids,
                positions,
                hidden_states,
                intermediate_tensors,
                inputs_embeds,
                spec_step_idx=i + 1,
            )
            # Same normalisation as EagleProposer._unwrap_model_out.
            if isinstance(out, (list, tuple)) and len(out) == 1:
                out = out[0]
            if isinstance(out, (list, tuple)):
                last_hidden, hidden_states = out
            else:
                last_hidden = hidden_states = out
            input_ids = self._greedy_tokens(last_hidden)
            tokens.append(input_ids)
        return tuple(tokens)


def build_mtp_fused_loop(
    *,
    vllm_config: VllmConfig,
    prefix: str,
    mtp: MtpDraftModel,
    num_steps: int,
    dynamic_arg_dims: dict,
) -> nn.Module:
    """Compile `_MtpFusedLoop` with the draft's own dynamic dims (Qwen3.5's
    MTP marks positions -1, DeepSeek's 0), so both programs specialise the
    token dimension the same way."""
    # Keep only the draft dims the loop's forward takes: a draft-only
    # argument (e.g. Qwen4ExpMTP's `draft_rows`) makes support_torch_compile
    # reject the loop, which then silently falls back to per-step dispatch.
    loop_args = inspect.signature(_MtpFusedLoop.forward).parameters
    dims = {k: v for k, v in dict(dynamic_arg_dims).items() if k in loop_args}
    cls = support_torch_compile(dynamic_arg_dims=dims)(
        type("MtpFusedLoop", (_MtpFusedLoop,), {})
    )
    return cls(vllm_config=vllm_config, prefix=prefix, mtp=mtp, num_steps=num_steps)
