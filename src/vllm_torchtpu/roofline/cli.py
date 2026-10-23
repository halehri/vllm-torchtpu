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
"""Command-line entrypoint for decode roofline planning and module breakdown."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from .chips import CHIPS
from .geometry import from_hf_config
from .modules import Dtypes, Parallel, StepShape
from .serving import plan
from .step import spec_cycle_floor, step_floor

_WEIGHT_BYTES: dict[str, float] = {"bf16": 2.0, "fp8": 1.0, "fp4": 0.5}
_KV_BYTES: dict[str, float] = {"bf16": 2.0, "fp8": 1.0}
_STATE_BYTES: dict[str, float] = {"fp32": 4.0, "bf16": 2.0}


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="Path to HF config.json")
    p.add_argument("--chip", default="tpu7x", choices=sorted(CHIPS))
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--context", type=int, default=0)
    p.add_argument("--path", default="standard", choices=("standard", "fused"))
    p.add_argument("--weight-dtype", default="fp8", choices=("bf16", "fp8", "fp4"))
    p.add_argument("--kv-dtype", default="bf16", choices=("bf16", "fp8"))
    p.add_argument("--state-dtype", default="fp32", choices=("fp32", "bf16"))
    p.add_argument("--link-bw-gbps", type=float, default=None)
    p.add_argument("--json", action="store_true")


def main(argv: list[str] | None = None) -> int:
    """Run the roofline plan or explain subcommand."""
    parser = argparse.ArgumentParser(prog="python -m vllm_torchtpu.roofline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan")
    _add_common_args(p_plan)
    p_plan.add_argument("--concurrency", default="1,4,16,64")
    p_plan.add_argument("--spec-window", type=int, default=0)
    p_plan.add_argument("--acceptance-length", type=float, default=1.0)
    p_plan.add_argument("--collective-latency-us", type=float, default=None)

    p_exp = sub.add_parser("explain")
    _add_common_args(p_exp)
    p_exp.add_argument("--batch", type=int, default=1)
    p_exp.add_argument("--rows-per-seq", type=int, default=1)
    p_exp.add_argument("--spec-window", type=int, default=0)

    args = parser.parse_args(argv)
    if args.tp <= 0:
        raise ValueError(f"--tp must be >= 1, got {args.tp}")
    cfg = json.loads(Path(args.config).read_text())
    geom = from_hf_config(cfg)
    chip = CHIPS[args.chip]
    if args.link_bw_gbps is not None:
        chip = replace(chip, link_bw=args.link_bw_gbps * 1e9)
    par = Parallel(tp=args.tp, kv_path=args.path)
    dt = Dtypes(
        weight=_WEIGHT_BYTES[args.weight_dtype],
        kv=_KV_BYTES[args.kv_dtype],
        state=_STATE_BYTES[args.state_dtype],
    )

    if args.cmd == "plan":
        concs = [int(x) for x in args.concurrency.split(",") if x.strip()]
        rows = plan(
            geom=geom,
            chip=chip,
            par=par,
            dt=dt,
            context=args.context,
            concurrencies=concs,
            spec_window=args.spec_window,
            acceptance_length=args.acceptance_length,
            collective_latency_us=args.collective_latency_us,
        )
        if args.json:
            sys.stdout.write(json.dumps([asdict(r) for r in rows]) + "\n")
            return 0
        hdr = (
            f"{'c':>4}  {'rows':>6}  {'binding':<7}  "
            f"{'hard_floor_ms':>13}  {'serial_floor_ms':>15}  "
            f"{'tpot_floor_ms':>13}  {'interactivity_ceil_tok_s':>24}  "
            f"{'output_tput_ceil_per_chip':>25}  {'flag':<7}\n"
        )
        sys.stdout.write(hdr)
        for r in rows:
            line = (
                f"{r.c:>4}  {r.rows:>6}  {r.binding:<7}  "
                f"{r.hard_floor_ms:>13.3f}  {r.serial_floor_ms:>15.3f}  "
                f"{r.tpot_floor_ms:>13.3f}  {r.interactivity_ceil_tok_s:>24.1f}  "
                f"{r.output_tput_ceil_per_chip:>25.1f}  {r.flag:<7}\n"
            )
            sys.stdout.write(line)
        return 0

    if args.spec_window > 0:
        sf = spec_cycle_floor(
            geom, chip, par, args.batch, args.context, args.spec_window, dt
        )
    else:
        sf = step_floor(
            geom=geom,
            chip=chip,
            par=par,
            shape=StepShape(
                batch=args.batch,
                rows_per_seq=args.rows_per_seq,
                context=args.context,
            ),
            dt=dt,
        )
    if args.json:
        sys.stdout.write(json.dumps([asdict(m) for m in sf.modules]) + "\n")
        return 0
    sys.stdout.write(
        f"{'module':<10}  {'hbm_mb':>10}  {'fp8_gflops':>10}  "
        f"{'bf16_gflops':>11}  {'collectives':>11}\n"
    )
    for m in sf.modules:
        sys.stdout.write(
            f"{m.name:<10}  {m.hbm_bytes / 1e6:>10.2f}  "
            f"{m.fp8_flops / 1e9:>10.2f}  {m.bf16_flops / 1e9:>11.2f}  "
            f"{m.n_collectives:>11}\n"
        )
    return 0
