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
"""Measured-vs-floor gap decomposition and sub-floor guardrail."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .chips import ChipSpec
from .step import StepFloor

_NAME_ALIASES: dict[str, str] = {
    "attention": "gqa",
    "attn": "gqa",
    "full_attention": "gqa",
    "rpa": "gqa",
    "linear_attention": "gdn",
    "mamba": "gdn",
    "routed_mlp": "moe",
    "fused_moe": "moe",
    "mlp": "mlp",
    "dense_mlp": "mlp",
    "ffn": "mlp",
    "head": "lm_head",
    "Step time": "step_ms",
    "step_time_ms": "step_ms",
}


@dataclass(frozen=True)
class ModuleGap:
    """Per-module analytical floor versus measured latency."""

    name: str
    hbm_ms: float
    mm_ms: float
    ici_ms: float
    floor_ms: float
    measured_ms: float
    gap_ms: float
    achieved_util: float
    gap_share: float
    status: str


@dataclass(frozen=True)
class StepGapReport:
    """Step-level and module-level comparison of measured latency against StepFloor."""

    modules: tuple[ModuleGap, ...]
    hard_floor_ms: float
    serial_floor_ms: float
    measured_step_ms: float
    unattributed_ms: float
    total_gap_ms: float
    achieved_step_util: float
    status: str


def _canonical_key(key: str) -> str:
    k = key.strip()
    return _NAME_ALIASES.get(k, _NAME_ALIASES.get(k.lower(), k.lower()))


def normalize_measured(
    raw: Mapping[str, object],
) -> tuple[dict[str, float], float | None]:
    """Extract canonical per-module latencies (ms) and optional total step_ms."""
    source: Mapping[str, object] = raw
    if "decode" in raw and isinstance(raw["decode"], Mapping):
        source = raw["decode"]
    elif "modules" in raw and isinstance(raw["modules"], Mapping):
        source = raw["modules"]

    mod_ms: dict[str, float] = {}
    step_ms: float | None = None
    for k, v in raw.items():
        if isinstance(v, (int, float)) and _canonical_key(str(k)) == "step_ms":
            step_ms = float(v)

    for k, v in source.items():
        if not isinstance(v, (int, float)):
            continue
        canon = _canonical_key(str(k))
        if canon == "step_ms":
            step_ms = float(v)
        else:
            mod_ms[canon] = mod_ms.get(canon, 0.0) + float(v)
    return mod_ms, step_ms


def compare_measured(
    sf: StepFloor,
    chip: ChipSpec,
    measured: Mapping[str, object],
    sub_floor_tol_ratio: float = 0.02,
) -> StepGapReport:
    """Compare a StepFloor against measured module/step latencies (ms).

    Flags any module or total step with
    ``measured_ms < floor_ms * (1 - sub_floor_tol_ratio)``
    as ``SUB_FLOOR_MODEL_BUG`` so under-counted bytes/FLOPs or mismatched
    parallelism cannot masquerade as >100% hardware efficiency.
    """
    mod_map, explicit_step_ms = normalize_measured(measured)
    raw_rows: list[tuple[str, float, float, float, float, float]] = []
    positive_gap_sum = 0.0
    sum_measured_mods = 0.0

    for m in sf.modules:
        hbm_ms = (m.hbm_bytes / chip.hbm_bw) * 1e3
        mm_ms = (m.fp8_flops / chip.fp8_flops + m.bf16_flops / chip.bf16_flops) * 1e3
        ici_ms = (
            (m.ici_bytes / chip.link_bw) * 1e3
            if chip.link_bw is not None and chip.link_bw > 0
            else 0.0
        )
        floor_ms = max(hbm_ms, mm_ms, ici_ms)
        meas_ms = mod_map.get(m.name, 0.0)
        sum_measured_mods += meas_ms
        gap_ms = meas_ms - floor_ms
        if meas_ms > 0.0 and gap_ms > 0.0:
            positive_gap_sum += gap_ms
        raw_rows.append((m.name, hbm_ms, mm_ms, ici_ms, floor_ms, meas_ms))

    measured_step_ms = (
        explicit_step_ms if explicit_step_ms is not None else sum_measured_mods
    )
    unattributed_ms = measured_step_ms - sum_measured_mods
    if unattributed_ms > 0.0:
        positive_gap_sum += unattributed_ms

    tol_factor = 1.0 - sub_floor_tol_ratio
    out_mods: list[ModuleGap] = []
    any_sub_floor = False
    for name, hbm_ms, mm_ms, ici_ms, floor_ms, meas_ms in raw_rows:
        gap_ms = meas_ms - floor_ms
        if meas_ms <= 0.0:
            status = "UNMEASURED"
            util = 0.0
            share = 0.0
        else:
            util = floor_ms / meas_ms if meas_ms > 0.0 else 0.0
            share = (
                max(0.0, gap_ms) / positive_gap_sum if positive_gap_sum > 0.0 else 0.0
            )
            if meas_ms < floor_ms * tol_factor:
                status = "SUB_FLOOR_MODEL_BUG"
                any_sub_floor = True
            else:
                status = "OK"
        out_mods.append(
            ModuleGap(
                name=name,
                hbm_ms=hbm_ms,
                mm_ms=mm_ms,
                ici_ms=ici_ms,
                floor_ms=floor_ms,
                measured_ms=meas_ms,
                gap_ms=gap_ms,
                achieved_util=util,
                gap_share=share,
                status=status,
            )
        )

    hard_floor_ms = sf.hard_floor_s * 1e3
    serial_floor_ms = sf.serial_floor_s * 1e3
    total_gap_ms = measured_step_ms - hard_floor_ms
    step_util = hard_floor_ms / measured_step_ms if measured_step_ms > 0.0 else 0.0
    if measured_step_ms > 0.0 and measured_step_ms < hard_floor_ms * tol_factor:
        any_sub_floor = True
    step_status = "SUB_FLOOR_MODEL_BUG" if any_sub_floor else "OK"
    return StepGapReport(
        modules=tuple(out_mods),
        hard_floor_ms=hard_floor_ms,
        serial_floor_ms=serial_floor_ms,
        measured_step_ms=measured_step_ms,
        unattributed_ms=unattributed_ms,
        total_gap_ms=total_gap_ms,
        achieved_step_util=step_util,
        status=step_status,
    )
