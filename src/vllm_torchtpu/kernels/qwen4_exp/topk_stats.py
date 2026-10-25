# SPDX-License-Identifier: Apache-2.0
"""Debug-only QSA top-k statistics from a host callback (`VLLM_TPU_QSA_TOPK_STATS`).

Per selection: the mean run length of each row's selected compressed groups
(consecutive group ids form one run), and for requests with exactly four rows
(MTP K=3 verify) the overlap of adjacent rows and the union size relative to the
sum of the rows' selections (what fetching a request's union once would save).
"""

import os

import jax
import numpy as np

_acc = {"calls": 0, "rows": 0, "runs": 0, "picks": 0, "pairs": 0, "shared": 0, "union": 0, "sum4": 0}


# Prefill chunks: per 512-row query block (the served SHARD_PREFILL_TILES block)
# the 1024-token KV tiles it visits vs the unique compressed groups its rows
# select, by context bucket. With VLLM_TPU_QSA_TOPK_DUMP=N the first N steps
# of at least 2048 rows are also saved for replay (blocks, lengths, starts).
_BLOCK, _TILE, _RATIO = 512, 1024, 4
_prefill = {}
_dumped = [0]


def _ctx_bucket(ctx):
    for edge in (16000, 32000, 64000, 128000):
        if ctx < edge:
            return f"<{edge // 1000}K"
    return ">=128K"


def _record_prefill(blocks, lengths, starts):
    for r in range(len(starts) - 1):
        lo, hi = int(starts[r]), int(starts[r + 1])
        if hi - lo < _BLOCK or hi > len(blocks):
            continue
        key = _ctx_bucket(int(lengths[hi - 1]))
        acc = _prefill.setdefault(key, [0, 0, 0, 0])  # blocks, tiles, union, picks
        for b in range(lo, hi, _BLOCK):
            rows = blocks[b : min(b + _BLOCK, hi)]
            g = rows[rows >= 0]
            acc[0] += 1
            acc[1] += len(np.unique(g * _RATIO // _TILE))
            acc[2] += len(np.unique(g))
            acc[3] += len(g)


def _dump(blocks, lengths, starts):
    limit = int(os.environ.get("VLLM_TPU_QSA_TOPK_DUMP", "0") or 0)
    if _dumped[0] >= limit or len(blocks) < 2048:
        return
    out = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
    path = os.path.join(out, f"qsa_topk_{os.getpid()}_{_dumped[0]}.npz")
    np.savez_compressed(path, blocks=blocks, lengths=lengths, starts=starts)
    _dumped[0] += 1
    print(f"qsa topk dump {path} rows={len(blocks)}", flush=True)


def _record(blocks, lengths, starts, every):
    blocks, lengths, starts = np.asarray(blocks), np.asarray(lengths), np.asarray(starts)
    _record_prefill(blocks, lengths, starts)
    _dump(blocks, lengths, starts)
    a = _acc
    a["calls"] += 1
    sets = []
    for row, length in zip(blocks, lengths):
        g = np.unique(row[row >= 0]) if length > 0 else np.empty(0, row.dtype)
        sets.append(g)
        if len(g):
            a["rows"] += 1
            a["picks"] += len(g)
            a["runs"] += 1 + int(np.count_nonzero(np.diff(g) != 1))
    for r in range(len(starts) - 1):
        lo, hi = int(starts[r]), int(starts[r + 1])
        if hi - lo != 4 or hi > len(sets) or not all(len(sets[i]) for i in range(lo, hi)):
            continue
        for i in range(lo, hi - 1):
            a["pairs"] += len(sets[i])
            a["shared"] += len(np.intersect1d(sets[i], sets[i + 1]))
        a["union"] += len(np.unique(np.concatenate(sets[lo:hi])))
        a["sum4"] += sum(len(sets[i]) for i in range(lo, hi))
    if a["calls"] % every == 0 and a["rows"]:
        print(
            "qsa topk stats calls=%d rows=%d mean_run=%.2f adj_overlap=%.3f union_over_sum=%.3f"
            % (
                a["calls"],
                a["rows"],
                a["picks"] / max(a["runs"], 1),
                a["shared"] / max(a["pairs"], 1),
                a["union"] / max(a["sum4"], 1),
            ),
            flush=True,
        )
        for key, (nb, tiles, union, picks) in sorted(_prefill.items()):
            print(
                "qsa topk prefill ctx%s blocks=%d tiles_per_block=%.1f "
                "union_groups_per_block=%.1f picks_per_block=%.1f "
                "tile_tokens_over_union_tokens=%.1f"
                % (key, nb, tiles / nb, union / nb, picks / nb,
                   tiles * _TILE / max(union * _RATIO, 1)),
                flush=True,
            )


def record(blocks, lengths, starts, every: int) -> None:
    jax.debug.callback(lambda b, l, s: _record(b, l, s, every), blocks, lengths, starts)
