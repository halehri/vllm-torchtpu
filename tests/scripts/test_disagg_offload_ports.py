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
"""Script-text checks for disaggregated offload controller port defaults."""

from __future__ import annotations

import pathlib
import re
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LINUX_EPHEMERAL_LOW = 32768
LINUX_EPHEMERAL_HIGH = 60999


class TestDisaggOffloadPorts(unittest.TestCase):
    def test_p4d2_launcher_offload_controller_ports_outside_ephemeral_range(
        self,
    ) -> None:
        launcher_path = (
            REPO_ROOT / "examples" / "disagg" / "launch_qwen35_p4d2_v2_baseline.sh"
        )
        launcher = launcher_path.read_text(encoding="utf-8")

        defaults = {
            match.group(1): int(match.group(2))
            for match in re.finditer(
                r'^([A-Z0-9_]+)="\$\{\1:-([0-9]+)\}"$',
                launcher,
                flags=re.MULTILINE,
            )
        }

        prefill_dp = defaults.get("PREFILL_DP", 1)
        decode_dp = defaults.get("DECODE_DP", 1)
        prefill_sb_workers = (
            defaults["PREFILL_PP"]
            * defaults["PREFILL_TP"]
            * defaults["PREFILL_PCP"]
            * prefill_dp
        )
        decode_sb_workers = (
            defaults.get("DECODE_PP", 1)
            * defaults["DECODE_TP"]
            * defaults.get("DECODE_PCP", 1)
            * decode_dp
        )
        self.assertEqual(prefill_sb_workers, 4)
        self.assertEqual(decode_sb_workers, 2)

        other_ports: dict[int, str] = {}
        for var_name in (
            "PROXY_PORT",
            "PREFILL_PORT",
            "DECODE_PORT",
            "KV_PORT",
            "PREFILL_TPU_KV_TRANSFER_PORT",
            "DECODE_TPU_KV_TRANSFER_PORT",
            "TPU_SIDE_CHANNEL_PORT",
            "PREFILL_CTRL_PORT",
            "DECODE_CTRL_PORT",
            "DECODE_DP_MASTER_PORT",
        ):
            other_ports[defaults[var_name]] = var_name

        for idx in range(prefill_sb_workers):
            other_ports[defaults["PREFILL_SB_PORT_BASE"] + idx] = (
                f"PREFILL_SB_PORT_BASE+{idx}"
            )
        for idx in range(decode_sb_workers):
            other_ports[defaults["DECODE_SB_PORT_BASE"] + idx] = (
                f"DECODE_SB_PORT_BASE+{idx}"
            )

        # Raiden offsets raiden_controller_port by dp_rank_local, and the
        # launcher's managed_ports loop reserves offsets 0..3 per instance.
        prefill_offsets = range(max(prefill_dp, 4))
        decode_offsets = range(max(decode_dp, 4))

        offload_ports: dict[int, str] = {}
        for var_name, offsets in (
            ("PREFILL_OFFLOAD_CONTROLLER_PORT", prefill_offsets),
            ("DECODE_OFFLOAD_CONTROLLER_PORT", decode_offsets),
        ):
            base_port = defaults[var_name]
            for offset in offsets:
                port = base_port + offset
                owner = f"{var_name}+{offset}"
                self.assertGreaterEqual(
                    port, 1024, f"{owner} port {port} is in privileged range"
                )
                self.assertFalse(
                    LINUX_EPHEMERAL_LOW <= port <= LINUX_EPHEMERAL_HIGH,
                    f"{owner} port {port} falls inside Linux ephemeral range "
                    f"{LINUX_EPHEMERAL_LOW}..{LINUX_EPHEMERAL_HIGH}",
                )
                self.assertNotIn(
                    port,
                    other_ports,
                    f"{owner} port {port} collides with {other_ports.get(port)}",
                )
                self.assertNotIn(
                    port,
                    offload_ports,
                    f"{owner} port {port} collides with {offload_ports.get(port)}",
                )
                offload_ports[port] = owner

    def test_kimi_k3_1p1d_config_controller_port_outside_ephemeral_range(
        self,
    ) -> None:
        config_path = (
            REPO_ROOT
            / "scripts"
            / "vllm"
            / "benchmarking"
            / "configs"
            / "kimi-k3-tp8-dp4-ep-1p1d.sh"
        )
        config_text = config_path.read_text(encoding="utf-8")
        ports = [
            int(match.group(1))
            for match in re.finditer(
                r'\\"raiden_controller_port\\":([0-9]+)', config_text
            )
        ]
        self.assertTrue(ports, "Expected raiden_controller_port in kimi-k3 config")
        # kimi-k3-tp8-dp4-ep-1p1d.sh runs with DP=4, so dp_rank_local spans 0..3.
        for base_port in ports:
            for dp_offset in range(4):
                port = base_port + dp_offset
                self.assertGreaterEqual(port, 1024)
                self.assertFalse(
                    LINUX_EPHEMERAL_LOW <= port <= LINUX_EPHEMERAL_HIGH,
                    f"raiden_controller_port+{dp_offset}={port} falls inside "
                    f"{LINUX_EPHEMERAL_LOW}..{LINUX_EPHEMERAL_HIGH}",
                )


if __name__ == "__main__":
    unittest.main()
