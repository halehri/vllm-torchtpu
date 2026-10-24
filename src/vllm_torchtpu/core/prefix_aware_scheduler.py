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
"""Prefix-aware admission scheduler for prefix-caching workloads.

Standard vLLM admits ready requests in FCFS order, which is oblivious to
whether a request's prompt prefix is already cached in HBM. In multi-turn
agentic workloads, admitting a long uncached prompt at queue[0] blocks ready
follow-up requests whose prefixes are already cached, and under high HBM
utilization can evict those cached prefixes before they are reused.

PrefixAwareScheduler inspects the ready queue within a lookahead window and
prioritizes requests with the fewest uncached prompt tokens. It modifies only
the admission ordering; memory management, eviction, and preemption policies
remain untouched.

Enable with:
    --scheduler-cls vllm_torchtpu.core.prefix_aware_scheduler.PrefixAwareScheduler
"""

from itertools import islice
from typing import cast

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    RequestQueue,
    SchedulingPolicy,
)
from vllm.v1.request import Request

from vllm_torchtpu import envs

logger = init_logger("vllm." + __name__)

_LOG_EVERY = 1000

# Priority tiers: lower values admit earlier.
_TIER_PREEMPTED = 1
_TIER_NORMAL = 2


class PrefixAwareScheduler(AsyncScheduler):
    """Admit ready requests prioritizing fewest uncached prompt tokens."""

    #: Lookahead window of ready requests to evaluate for cached prefix length.
    WINDOW = 64

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._paw_n = 0
        self._paw_cold_tokens = 0
        self._paw_prompt_tokens = 0
        self._paw_scoring_broken = False

        logger.info("PrefixAwareScheduler ACTIVE (window=%d)", self.WINDOW)

    def _prefix_warmth(self, request: Request) -> int:
        """Prompt tokens of ``request`` resident in the KV cache."""
        kvm = self.kv_cache_manager
        if not kvm.enable_caching or request.skip_reading_prefix_cache:
            return 0
        if request.num_computed_tokens > 0:
            return request.num_computed_tokens
        try:
            result = kvm.coordinator.find_longest_cache_hit(
                request.block_hashes, max(request.num_tokens - 1, 0)
            )
            return int(result[1])
        except Exception:
            if not self._paw_scoring_broken:
                self._paw_scoring_broken = True
                logger.exception(
                    "PrefixAwareScheduler: warmth scoring failed; falling back to FCFS."
                )
            return 0

    def _rank_and_warmth(
        self, request: Request
    ) -> tuple[tuple[int, float, float], int]:
        """Return (sort_key, warmth). Lower sort_key is admitted first."""
        warmth = self._prefix_warmth(request)
        cold_tokens = float(request.num_tokens - warmth)
        tier = _TIER_PREEMPTED if request.num_preemptions > 0 else _TIER_NORMAL
        return (tier, cold_tokens, request.arrival_time), warmth

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        queue = getattr(self, "kv_holding_waiting", None) or getattr(
            self, "waiting", None
        )
        if queue is not None:
            self._promote_warmest(queue)
        return queue

    def schedule(self, throttle_prefills: bool = False):
        queue = getattr(self, "kv_holding_waiting", None) or getattr(
            self, "waiting", None
        )
        if queue is not None:
            self._promote_warmest(queue)
        cap = envs.VLLM_TPU_PREFILL_STEP_CAP
        if cap > 0:
            return self._schedule_with_prefill_cap(throttle_prefills, cap)
        return super().schedule(throttle_prefills)

    def _schedule_with_prefill_cap(self, throttle_prefills: bool, cap: int):
        """Cap a step's prefill tokens at `cap` while any decode is running
        (VLLM_TPU_PREFILL_STEP_CAP).

        A full prefill chunk makes every running decode wait out its whole
        forward; capped, the same prefill runs over more, shorter steps and
        the decodes advance on each of them. The step budget becomes `cap`
        plus the running decodes' tokens, decodes are scheduled first so
        they always fit, and each request's chunk is capped at `cap`, so
        the prefill still progresses every step. Without a running decode
        the full budget applies.
        """
        decodes = sum(1 for r in self.running if not r.is_prefill_chunk)
        if decodes == 0:
            return super().schedule(throttle_prefills)
        spec = self.vllm_config.speculative_config
        per_decode = 1 + (spec.num_speculative_tokens if spec is not None else 0)
        budget = cap + decodes * per_decode
        if budget >= self.max_num_scheduled_tokens:
            return super().schedule(throttle_prefills)
        # Decodes first (stable), so the prefill chunks take what is left.
        self.running.sort(key=lambda r: r.is_prefill_chunk)
        sched_config = self.scheduler_config
        saved = (
            self.max_num_scheduled_tokens,
            sched_config.long_prefill_token_threshold,
            getattr(self, "adaptive_long_prefill_threshold", False),
        )
        self.max_num_scheduled_tokens = budget
        sched_config.long_prefill_token_threshold = cap
        self.adaptive_long_prefill_threshold = False
        try:
            return super().schedule(throttle_prefills)
        finally:
            (
                self.max_num_scheduled_tokens,
                sched_config.long_prefill_token_threshold,
                self.adaptive_long_prefill_threshold,
            ) = saved

    def _promote_warmest(self, queue: RequestQueue) -> None:
        """Promote the request with the fewest uncached tokens within WINDOW."""
        if self.policy != SchedulingPolicy.FCFS or len(queue) < 2:
            return

        window = list(islice(queue, self.WINDOW))
        head = window[0]
        scored = [(self._rank_and_warmth(r), r) for r in window]
        (best_rank, best_warmth), best = min(scored, key=lambda t: t[0][0])

        if best is not head and best_rank < scored[0][0][0]:
            # Only FCFS reaches here (see the early return above).
            cast(FCFSRequestQueue, queue).remove(best)
            queue.prepend_request(best)

        self._paw_n += 1
        self._paw_cold_tokens += max(best.num_tokens - best_warmth, 0)
        self._paw_prompt_tokens += best.num_tokens

        if self._paw_n % _LOG_EVERY == 0:
            logger.info(
                "warmest-first: n=%d mean_cold_tokens=%.0f "
                "cold_frac=%.3f queue_depth=%d",
                self._paw_n,
                self._paw_cold_tokens / self._paw_n,
                self._paw_cold_tokens / max(self._paw_prompt_tokens, 1),
                len(queue),
            )
