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
"""Wire bytes, serial hops, and off-chip share of decode collectives."""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Kind = Literal["ar", "ag", "rs", "push"]
Algo = Literal["one_shot", "two_shot", "ring"]


@dataclass(frozen=True)
class Collective:
    """One collective as seen by one rank."""

    name: str
    group: int
    wire_bytes: float  # bytes this rank sends
    hops: int  # serial steps this collective adds to the dependency chain
    offchip: float  # share of wire_bytes that leaves the chip
    kind: Kind
    payload: float  # AR/RS input, AG output, or push wire, in bytes


def offchip_share(group: int, ranks_per_chip: int) -> float:
    """Share of peers outside this rank's chip, assuming adjacent ranks share a chip."""
    if group <= 1:
        return 0.0
    return max(0, group - min(group, ranks_per_chip)) / (group - 1)


def make(
    name: str,
    kind: Kind,
    payload: float,
    group: int,
    algo: Algo,
    ranks_per_chip: int,
) -> Collective:
    """Price one collective.

    Payload is the AR/RS input, AG output, or the push wire.
    """
    if group <= 1:
        return Collective(name, group, 0.0, 0, 0.0, kind, payload)
    n = group
    if kind == "push":
        wire, hops = payload, 1
    elif kind == "ar":
        if algo == "one_shot":
            wire, hops = (n - 1) * payload, 1
        elif algo == "two_shot":
            wire, hops = 2 * (n - 1) / n * payload, 2
        else:
            wire, hops = 2 * (n - 1) / n * payload, 2 * (n - 1)
    else:
        wire = (n - 1) / n * payload
        hops = n - 1 if algo == "ring" else 1
    return Collective(
        name, n, wire, hops, offchip_share(n, ranks_per_chip), kind, payload
    )


@dataclass(frozen=True)
class CollectiveTable:
    """Measured per-call time by (name or kind, group) and payload.

    Each payload keeps its raw samples; lookup uses their mean and interpolates
    linearly between payloads. Outside the measured range it returns the
    nearest payload's mean and source "extrapolated".
    """

    # key -> ((payload_B, (seconds, ...)), ...), payloads ascending and unique
    rows: Mapping[tuple[str, int], tuple[tuple[float, tuple[float, ...]], ...]]

    def time_s(self, c: Collective) -> tuple[float, str]:
        pts = self.rows.get((c.name, c.group)) or self.rows.get((c.kind, c.group))
        if not pts:
            return 0.0, "missing"
        xs = [p for p, _ in pts]
        ys = [sum(s) / len(s) for _, s in pts]
        if c.payload < xs[0] or c.payload > xs[-1]:
            return (ys[0] if c.payload < xs[0] else ys[-1]), "extrapolated"
        i = bisect.bisect_left(xs, c.payload)
        if xs[i] == c.payload:
            return ys[i], "measured"
        x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
        return y0 + (y1 - y0) * (c.payload - x0) / (x1 - x0), "interpolated"


def padded_rows(rows: int, tp: int, residual: str) -> int:
    """Rows a collective carries: sequence parallelism pads to a multiple of tp."""
    return rows if residual == "replicated" else math.ceil(rows / tp) * tp
