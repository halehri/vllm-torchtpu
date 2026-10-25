# SPDX-License-Identifier: Apache-2.0
"""The fused draft loop's vocab-parallel greedy argmax (VLLM_TPU_MTP_SHARDED_ARGMAX).

Each simulated TP rank holds one vocab shard; the all-gather exchanges the
[rows, 2] (value, index) pairs between threads. The result must match
torch.argmax over the gathered, unpadded logits row for row.
"""

import threading
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.spec_decode import mtp_fused_loop
from vllm_torchtpu.spec_decode.mtp_fused_loop import sharded_greedy_argmax

ORG_VOCAB, TP, ROWS = 37, 4, 6
SHARD = -(-ORG_VOCAB // TP)  # padded vocab rows per rank


def _run_ranks(full):
    """Every rank's sharded argmax over `full` [rows, TP * SHARD]."""
    barrier = threading.Barrier(TP)
    slots, results = [None] * TP, [None] * TP

    def body(rank):
        def all_gather(pair):
            barrier.wait()
            slots[rank] = pair
            barrier.wait()
            return torch.cat(slots, dim=-1)

        local = full[:, rank * SHARD : (rank + 1) * SHARD]
        offset = torch.tensor([rank * SHARD], dtype=torch.int32)
        results[rank] = sharded_greedy_argmax(local, offset, ORG_VOCAB, TP, all_gather)

    threads = [threading.Thread(target=body, args=(r,)) for r in range(TP)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def _logits(seed=0):
    torch.manual_seed(seed)
    full = torch.randn(ROWS, TP * SHARD).bfloat16()
    full[:, ORG_VOCAB:] = 100.0  # padding must never win
    full[1, [3, 25]] = 50.0  # a tie across ranks: the lower index wins
    full[2, [11, 12]] = 50.0  # a tie inside one shard
    full[3, ORG_VOCAB - 1] = 60.0  # the last real token
    return full


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sharded_argmax_matches_argmax_over_gathered_logits(seed):
    full = _logits(seed)
    expected = full[:, :ORG_VOCAB].argmax(dim=-1).to(torch.int32)
    for got in _run_ranks(full):
        assert torch.equal(got, expected), (got, expected)
    assert expected[1] == 3 and expected[2] == 11 and expected[3] == ORG_VOCAB - 1


def test_a_single_rank_needs_no_gather():
    full = _logits()[:, :SHARD]

    def never(_):
        raise AssertionError("tp_size 1 must not gather")

    got = sharded_greedy_argmax(
        full, torch.tensor([0], dtype=torch.int32), ORG_VOCAB, 1, never
    )
    assert torch.equal(got, full.argmax(dim=-1).to(torch.int32))


def test_only_a_draft_declaring_one_head_takes_it(monkeypatch):
    monkeypatch.setattr(mtp_fused_loop.envs, "VLLM_TPU_MTP_SHARDED_ARGMAX", True)
    head = SimpleNamespace(tp_size=TP)
    processor = SimpleNamespace(logits_as_input=False, scale=1.0)

    class PerStepHeads:  # e.g. a draft choosing its head by spec_step_idx
        lm_head, logits_processor = head, processor

    class OneHead(PerStepHeads):
        sharded_argmax_head = True

    loop = mtp_fused_loop._MtpFusedLoop.__new__(mtp_fused_loop._MtpFusedLoop)
    for draft, expected in ((PerStepHeads(), None), (OneHead(), head)):
        object.__setattr__(loop, "mtp", draft)
        assert loop._sharded_argmax_head() is expected
