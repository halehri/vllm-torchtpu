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
"""Measured per-call collective times on TPU v7x, 32 ranks over 4 hosts.

Each sample is one per-call device time in seconds, read on rank 0 from decode
XProf traces of a TP32 + DCP8 + EP32 engine (MTP off, FP8 KV) at concurrency
4, 8, and 32 (512-token prompts), 64 (32K) and 256 (1K). All-gather payloads
cover 32 ranks x pow2(max(1, batch / 32)) rows; the all-reduce carries
max(16, batch) rows.
"""

from __future__ import annotations

from .collectives import CollectiveTable

_KIB = 1024

TPU7X_32R = CollectiveTable(
    rows={
        # FP8 token all-gather before the MoE kernel, 32 x rows x 8192 x 1 B.
        ("dispatch_ag", 32): (
            (256 * _KIB, (32.65e-6, 37.17e-6, 35.39e-6)),  # c=4, 8, 32
            (512 * _KIB, (34.64e-6,)),  # c=64
            (2048 * _KIB, (28.16e-6,)),  # c=256
        ),
        # Routing-metadata all-gather, s32[32, rows, 768].
        ("route_ag", 32): (
            (96 * _KIB, (29.11e-6, 33.58e-6, 32.34e-6)),  # c=4, 8, 32
            (192 * _KIB, (30.10e-6,)),  # c=64
            (768 * _KIB, (36.56e-6,)),  # c=256
        ),
        # Restore all-gather after the MoE kernel, bf16[32, rows, 8192].
        ("restore_ag", 32): (
            (512 * _KIB, (31.55e-6, 33.67e-6, 34.13e-6)),  # c=4, 8, 32
            (1024 * _KIB, (31.15e-6,)),  # c=64
            (4096 * _KIB, (42.19e-6,)),  # c=256
        ),
        # Residual all-reduce, bf16[max(16, batch), 8192]; prices any 32-rank AR.
        ("ar", 32): (
            (256 * _KIB, (29.41e-6, 25.51e-6)),  # c=4, 8
            (512 * _KIB, (30.77e-6,)),  # c=32
            (1024 * _KIB, (38.63e-6,)),  # c=64
            (4096 * _KIB, (95.09e-6,)),  # c=256
        ),
    }
)
