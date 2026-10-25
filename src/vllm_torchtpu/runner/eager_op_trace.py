# SPDX-License-Identifier: Apache-2.0
"""Census of the eager (uncompiled) torch ops a served step issues, with their source lines.

Every eager op on a TPU tensor becomes its own small device program (`tt_jit_where`,
`tt_jit_gather`, ... in the Perfetto trace) whose name carries no source location. With
`VLLM_TPU_TRACE_EAGER_OPS=K:M`, rank 0 records, for engine steps `K .. K+M-1` (counted on
non-empty `execute_model` calls), every aten op on a device tensor that reaches the Python
dispatcher during
`execute_model` / `sample_tokens` together with the innermost vLLM / vllm_torchtpu frame
that issued it, then logs the per-step counts once. Compiled programs (the model forward, the
drafter loop, jitted helpers) run as single calls after warm-up, so their internals are not
counted. Diagnostic only: the dispatch mode and stack walks slow the traced steps down.
"""

from __future__ import annotations

import collections
import contextlib
import os
import sys
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_SRC_MARKERS = ("/vllm_torchtpu/", "/vllm/")
_SKIP = ("eager_op_trace.py", "/torch/", "/contextlib.py")


def _parse(spec: str) -> tuple[int, int] | None:
    if not spec:
        return None
    k, _, m = spec.partition(":")
    return int(k), int(m or 100)


def _on_device(obj: Any) -> bool:
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.device.type != "cpu"
    if isinstance(obj, (list, tuple)):
        return any(_on_device(o) for o in obj)
    if isinstance(obj, dict):
        return any(_on_device(o) for o in obj.values())
    return False


def _site() -> str:
    f = sys._getframe(2)
    while f is not None:
        name = f.f_code.co_filename
        if any(m in name for m in _SRC_MARKERS) and not any(s in name for s in _SKIP):
            short = name.split("/vllm_torchtpu/")[-1] if "/vllm_torchtpu/" in name else name.split("/vllm/")[-1]
            return f"{short}:{f.f_lineno} {f.f_code.co_name}"
        f = f.f_back
    return "?"


class EagerOpTrace:
    """Per-runner tracer; `step()` wraps one execute_model / sample_tokens call."""

    def __init__(self) -> None:
        self.window = _parse(os.environ.get("VLLM_TPU_TRACE_EAGER_OPS", ""))
        self.steps = 0
        self.counts: collections.Counter[tuple[str, str]] = collections.Counter()
        self.done = self.window is None
        self.active = False  # execute_model may call sample_tokens itself: count once
        if not self.done:
            try:
                import torch.distributed as dist

                if dist.is_initialized() and dist.get_rank() != 0:
                    self.done = True
            except Exception:
                pass

    def count_step(self) -> None:
        if not self.done:
            self.steps += 1

    @contextlib.contextmanager
    def step(self) -> Any:
        if self.active or self.done or not (
            self.window[0] <= self.steps < self.window[0] + self.window[1]
        ):
            if not self.done and self.steps >= self.window[0] + self.window[1]:
                self._report()
            yield
            return
        from torch.utils._python_dispatch import TorchDispatchMode

        counts = self.counts

        class _Mode(TorchDispatchMode):
            # Reported as an infra mode so torch.compile'd helpers called inside the step
            # (e.g. _staged_walk_products_compiled, fullgraph=True) still run their compiled
            # code: with a non-infra mode present Dynamo skips every frame and raises
            # (g01 j-3b5c3f5c: "non-infra torch dispatch mode present").
            @classmethod
            def is_infra_mode(cls) -> bool:
                return True

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
                out = func(*args, **(kwargs or {}))
                # Only ops touching a device tensor become device programs; host-side
                # bookkeeping on CPU tensors (block-table mirrors, staging) is skipped.
                if _on_device(args) or _on_device(kwargs) or _on_device(out):
                    counts[(str(func.overloadpacket.__name__), _site())] += 1
                return out

        self.active = True
        try:
            with _Mode():
                yield
        finally:
            self.active = False

    def _report(self) -> None:
        self.done = True
        n = max(1, self.window[1])
        rows = sorted(self.counts.items(), key=lambda kv: -kv[1])
        logger.info("eager op census: %d steps, %d op sites, %.1f ops/step", n, len(rows),
                    sum(self.counts.values()) / n)
        for (op, site), c in rows[:80]:
            logger.info("eager op census: %6.2f/step %-24s %s", c / n, op, site)
