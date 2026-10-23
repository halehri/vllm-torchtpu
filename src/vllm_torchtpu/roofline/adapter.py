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
"""Megakernel Plan and Profile adapter for the decode roofline library."""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any

from .chips import CHIPS, ChipSpec
from .geometry import ModelGeometry
from .modules import Dtypes, ModuleCost, Parallel, StepShape, experts_touched
from .step import StepFloor, _combine_modules, step_floor

DEFAULT_DTYPES = Dtypes()


@dataclass(frozen=True)
class PlanReconciliation:
    """Per-module byte and FLOP delta between Plan floor and standalone step_floor."""

    module: str
    plan_hbm_bytes: float
    standalone_hbm_bytes: float
    delta_hbm_bytes: float
    plan_flops: float
    standalone_flops: float
    delta_flops: float


def _block_collectives(spec: Any, n_layers: int, r_val: int) -> int:
    if r_val > 1 and spec is not None and spec.reduction == "allreduce":
        return n_layers
    return 0


def _dtype_bytes(dt: Any) -> float:
    """Return element size in bytes from a JAX/NumPy dtype, class, or string."""
    if isinstance(dt, (int, float)):
        return float(dt)
    name = getattr(dt, "__name__", str(dt)).lower()
    if "float4" in name or "fp4" in name or "int4" in name:
        return 0.5
    if "float8" in name or "fp8" in name or "int8" in name or "uint8" in name:
        return 1.0
    if "bfloat16" in name or "bf16" in name or "float16" in name or "fp16" in name:
        return 2.0
    if "float32" in name or "fp32" in name or "int32" in name:
        return 4.0
    itemsize = getattr(dt, "itemsize", None)
    if itemsize is not None:
        return float(itemsize)
    dtype_attr = getattr(dt, "dtype", None)
    if dtype_attr is not None and hasattr(dtype_attr, "itemsize"):
        return float(dtype_attr.itemsize)
    raise ValueError(f"Unrecognized dtype in Megakernel Plan: {dt!r}")


def floor_from_plan(
    p: Any,
    ex: Any = None,
    *,
    B: int | None = None,
    S: int | None = None,
    T: int | None = None,
    R: int | None = None,
    state_writes: int | None = None,
    chip: ChipSpec = CHIPS["tpu7x"],
    kv_path: str = "fused",
    q_heads_global: int | None = None,
    num_kv_heads_global: int | None = None,
    num_experts_global: int | None = None,
    top_k: int = 10,
    vocab: int = 0,
    lm_head_weight_bytes: float = 1.0,
    include_plan_extras: bool = True,
) -> StepFloor:
    """Derive a StepFloor directly from a megakernel.planner.Plan and ExecSpec.

    Args:
        p: A ``megakernel.planner.Plan`` (or duck-typed equivalent).
        ex: Optional ``megakernel.exec_spec.ExecSpec`` providing ``B``,
            ``T``, ``R``, ``ctx``.
        B: Sequence batch size (if ``ex`` is provided and ``B`` is omitted,
            ``ex.B // ex.T``).
        S: Active KV context length per sequence (defaults to ``ex.ctx`` or ``0``).
        T: Tokens/rows per sequence (defaults to ``ex.T`` or ``1``).
        R: Tensor/expert parallel rank count (defaults to ``ex.R`` or ``1``).
        state_writes: Recurrent state write count per sequence (defaults to ``T``).
        chip: Target ``ChipSpec`` (defaults to ``CHIPS["tpu7x"]``).
        kv_path: ``"fused"`` (split KV pass within KV group) or ``"standard"``.
        q_heads_global: Total query attention heads in the model.
        num_kv_heads_global: Total KV attention heads in the model (for KV group size).
        num_experts_global: Total routed experts across all ranks.
        top_k: Active routed experts per token.
        vocab: Optional vocabulary size to append an ``lm_head`` ``ModuleCost`` row.
        lm_head_weight_bytes: Element size in bytes for ``lm_head`` weights.
        include_plan_extras: When ``True`` (default), includes Megakernel-specific
            ``BA_PAD=128`` columns in ``lin_in``, the ``es`` F32 scale table in ``moe``,
            and a ``smalls`` ``ModuleCost`` row. When ``False``, strips those three
            plan-only extras and restores unpadded GDN ``conv`` weights to match
            standalone ``step_floor`` byte-for-byte.
    """
    t_val = int(T if T is not None else getattr(ex, "T", 1))
    r_val = int(R if R is not None else getattr(ex, "R", 1))
    if B is not None:
        b_val = int(B)
    elif ex is not None and hasattr(ex, "B"):
        b_val = max(1, int(ex.B) // max(1, t_val))
    else:
        b_val = 1
    s_val = int(S if S is not None else getattr(ex, "ctx", 0))
    sw_val = int(state_writes if state_writes is not None else t_val)
    m_rows = b_val * t_val

    block_counts: dict[str, int] = dict(getattr(p, "block_layer_counts", {}))
    raw_specs = getattr(p, "block_specs", None)
    if raw_specs is not None:
        block_specs: dict[str, Any] = {bs.name: bs for bs in raw_specs}
    else:
        block_specs = {}
        bp_dict = getattr(p, "block_params", {})
        sd_dict = getattr(p, "state_decls", {})
        if "gqa" in bp_dict or "gqa" in block_counts:
            bp_gqa = bp_dict.get("gqa")
            kc_state = sd_dict.get("kc")
            page_dtype = (
                getattr(kc_state, "dtype", "bfloat16") if kc_state else "bfloat16"
            )
            block_specs["gqa"] = types.SimpleNamespace(
                name="gqa",
                reduction="allreduce",
                cache_decl=types.SimpleNamespace(
                    hd=getattr(bp_gqa, "HD", 256) if bp_gqa else 256,
                    kvh=getattr(bp_gqa, "NKV", 1) if bp_gqa else 1,
                    page_size=getattr(bp_gqa, "BLK", 128) if bp_gqa else 128,
                    page_dtype=page_dtype,
                ),
            )
        if "gdn" in bp_dict or "gdn" in block_counts:
            bp_gdn = bp_dict.get("gdn")
            rec_state = sd_dict.get("rec")
            rec_dtype = (
                getattr(rec_state, "dtype", "bfloat16") if rec_state else "bfloat16"
            )
            conv_state = sd_dict.get("conv")
            conv_dtype = (
                getattr(conv_state, "dtype", "float32") if conv_state else "float32"
            )
            block_specs["gdn"] = types.SimpleNamespace(
                name="gdn",
                reduction="allreduce",
                cache_decl=types.SimpleNamespace(
                    kh=getattr(bp_gdn, "kh", 2) if bp_gdn else 2,
                    vh=getattr(bp_gdn, "vh", 8) if bp_gdn else 8,
                    KD=getattr(bp_gdn, "KD", 128) if bp_gdn else 128,
                    VD=getattr(bp_gdn, "VD", 128) if bp_gdn else 128,
                    rec_dtype=rec_dtype,
                    conv_taps=3,
                    conv_dtype=conv_dtype,
                ),
            )
        if "moe" in bp_dict or "moe" in block_counts:
            block_specs["moe"] = types.SimpleNamespace(
                name="moe",
                reduction="allreduce",
                cache_decl=None,
            )

    raw_weights = getattr(p, "weight_decls", ())
    if isinstance(raw_weights, dict):
        weight_decls = tuple(raw_weights.values())
    else:
        weight_decls = tuple(raw_weights)

    # Infer hidden dimension from static projection K (e.g. att_in, lin_in, router)
    hidden = 0
    for wd in weight_decls:
        if wd.name in ("att_in", "lin_in", "router"):
            hidden = int(wd.K)
            break
    ar_payload = float(m_rows * hidden * 2)

    # Group weight declarations by owner block
    weight_owners: dict[str, str] = dict(getattr(p, "weight_owners", {}))
    by_owner: dict[str, list[Any]] = {}
    for wd in weight_decls:
        owner = getattr(wd, "owner", None) or weight_owners.get(wd.name, "unknown")
        by_owner.setdefault(owner, []).append(wd)

    modules: list[ModuleCost] = []

    # 1. GQA block
    n_gqa = block_counts.get("gqa", 0)
    if n_gqa > 0:
        gqa_hbm = 0.0
        gqa_fp8 = 0.0
        gqa_bf16 = 0.0
        for wd in by_owner.get("gqa", ()):
            db = _dtype_bytes(wd.dtype)
            elems = float(wd.K) * float(wd.N)
            gqa_hbm += n_gqa * elems * db
            flops = n_gqa * 2.0 * m_rows * elems
            if db <= 1.0:
                gqa_fp8 += flops
            else:
                gqa_bf16 += flops

        gqa_spec = block_specs.get("gqa")
        cd = getattr(gqa_spec, "cache_decl", None)
        if cd is not None:
            kvh = int(cd.kvh)
            hd = int(cd.hd)
            kv_b = _dtype_bytes(cd.page_dtype)
            att_out_decl = next(
                (w for w in by_owner.get("gqa", ()) if w.name == "att_out"), None
            )
            if q_heads_global is not None:
                q_per_rank = max(1, q_heads_global // r_val)
            elif att_out_decl is not None and hd > 0:
                q_per_rank = max(1, int(att_out_decl.K) // hd)
            else:
                q_per_rank = max(1, 32 // r_val)
            kv_heads_glob = (
                num_kv_heads_global if num_kv_heads_global is not None else 2
            )
            kv_group = (
                max(1, r_val // kv_heads_glob) if kv_heads_glob > 0 else 1
            )
            kv_div = kv_group if kv_path == "fused" else 1
            kv_bytes = float(b_val * kvh * hd * 2 * s_val) * kv_b / kv_div
            gqa_hbm += n_gqa * kv_bytes
            gqa_bf16 += n_gqa * (4.0 * m_rows * q_per_rank * s_val * hd)

        n_coll_gqa = _block_collectives(gqa_spec, n_gqa, r_val)
        ici_gqa = ar_payload * n_coll_gqa if n_coll_gqa > 0 else 0.0
        modules.append(
            ModuleCost(
                name="gqa",
                hbm_bytes=gqa_hbm,
                fp8_flops=gqa_fp8,
                bf16_flops=gqa_bf16,
                ici_bytes=ici_gqa,
                n_collectives=n_coll_gqa,
            )
        )

    # 2. GDN block
    n_gdn = block_counts.get("gdn", 0)
    if n_gdn > 0:
        gdn_hbm = 0.0
        gdn_fp8 = 0.0
        gdn_bf16 = 0.0
        gdn_spec = block_specs.get("gdn")
        cd = getattr(gdn_spec, "cache_decl", None)
        for wd in by_owner.get("gdn", ()):
            db = _dtype_bytes(wd.dtype)
            n_cols = float(wd.N)
            if not include_plan_extras and wd.name == "lin_in" and cd is not None:
                vh = int(cd.vh)
                ba_pad = max(2 * vh, 128)
                n_cols -= float(ba_pad - 2 * vh)
            elems = float(wd.K) * n_cols
            gdn_hbm += n_gdn * elems * db
            flops = n_gdn * 2.0 * m_rows * elems
            if db <= 1.0:
                gdn_fp8 += flops
            else:
                gdn_bf16 += flops

        if cd is not None:
            kh = int(cd.kh)
            vh = int(cd.vh)
            kd = int(cd.KD)
            vd = int(cd.VD)
            state_b = _dtype_bytes(cd.rec_dtype)
            conv_b = _dtype_bytes(getattr(cd, "conv_dtype", cd.rec_dtype))
            conv_taps = int(cd.conv_taps)
            conv_k = conv_taps + 1
            qkv_r = 2 * kh * kd + vh * vd
            if not include_plan_extras:
                # Restore conv weight + FLOPs to gdn to match standalone step_floor
                lin_in_decl = next(
                    (w for w in by_owner.get("gdn", ()) if w.name == "lin_in"),
                    None,
                )
                w_b = (
                    _dtype_bytes(lin_in_decl.dtype) if lin_in_decl is not None else 1.0
                )
                conv_elems = float(conv_k * qkv_r)
                gdn_hbm += n_gdn * conv_elems * w_b
                if w_b <= 1.0:
                    gdn_fp8 += n_gdn * 2.0 * m_rows * conv_elems
                else:
                    gdn_bf16 += n_gdn * 2.0 * m_rows * conv_elems
            state_elems = float(vh * kd * vd)
            conv_state_elems = float(qkv_r * conv_taps)
            gdn_hbm += (
                n_gdn
                * float(b_val)
                * (1.0 + sw_val)
                * (state_elems * state_b + conv_state_elems * conv_b)
            )

        n_coll_gdn = _block_collectives(gdn_spec, n_gdn, r_val)
        ici_gdn = ar_payload * n_coll_gdn if n_coll_gdn > 0 else 0.0
        modules.append(
            ModuleCost(
                name="gdn",
                hbm_bytes=gdn_hbm,
                fp8_flops=gdn_fp8,
                bf16_flops=gdn_bf16,
                ici_bytes=ici_gdn,
                n_collectives=n_coll_gdn,
            )
        )

    # 3. MoE block
    n_moe = block_counts.get("moe", 0)
    if n_moe > 0:
        moe_hbm = 0.0
        moe_fp8 = 0.0
        moe_bf16 = 0.0
        moe_wds = by_owner.get("moe", ())
        router_decl = next((w for w in moe_wds if w.name == "router"), None)
        lead_decl = next((w for w in moe_wds if w.lead_dims), None)
        if num_experts_global is not None:
            e_global = num_experts_global
        elif router_decl is not None:
            e_global = int(router_decl.N)
        elif lead_decl is not None:
            e_global = int(lead_decl.lead_dims[0])
        else:
            e_global = 512
        e_d_global = experts_touched(e_global, top_k, m_rows)
        for wd in by_owner.get("moe", ()):
            if wd.name == "es" and not include_plan_extras:
                continue
            db = _dtype_bytes(wd.dtype)
            elems = float(wd.K) * float(wd.N)
            lead = getattr(wd, "lead_dims", ())
            if not lead:
                # Static replicated/TP-sharded weight (router, sh13, sh2)
                moe_hbm += n_moe * elems * db
                flops = n_moe * 2.0 * m_rows * elems
                if db <= 1.0:
                    moe_fp8 += flops
                else:
                    moe_bf16 += flops
            else:
                e_rank = int(lead[0])
                scale = float(e_rank) / float(max(1, e_global))
                e_d_rank = e_d_global * scale
                moe_hbm += n_moe * e_d_rank * elems * db
                if wd.name in ("e13", "e2"):
                    active_rows = float(m_rows * top_k) * scale
                    flops = n_moe * 2.0 * active_rows * elems
                    if db <= 1.0:
                        moe_fp8 += flops
                    else:
                        moe_bf16 += flops

        moe_spec = block_specs.get("moe")
        n_coll_moe = _block_collectives(moe_spec, n_moe, r_val)
        ici_moe = ar_payload * n_coll_moe if n_coll_moe > 0 else 0.0
        modules.append(
            ModuleCost(
                name="moe",
                hbm_bytes=moe_hbm,
                fp8_flops=moe_fp8,
                bf16_flops=moe_bf16,
                ici_bytes=ici_moe,
                n_collectives=n_coll_moe,
            )
        )

    # 4. Smalls per-layer F32 table (when include_plan_extras=True)
    if include_plan_extras:
        smalls_shape = getattr(p, "smalls_shape", None)
        if smalls_shape is not None and len(smalls_shape) == 2:
            dims, s_dt = smalls_shape
            smalls_elems = float(math.prod(int(d) for d in dims))
            smalls_b = smalls_elems * _dtype_bytes(s_dt)
            if smalls_b > 0:
                modules.append(
                    ModuleCost(
                        name="smalls",
                        hbm_bytes=smalls_b,
                        fp8_flops=0.0,
                        bf16_flops=0.0,
                        ici_bytes=0.0,
                        n_collectives=0,
                    )
                )
        elif hasattr(p, "smalls_width") and p.smalls_width:
            num_layers = (
                sum(block_counts.values()) or len(getattr(p, "slot_ar_index", ())) or 60
            )
            smalls_b = float(num_layers * p.smalls_width * 4)
            if smalls_b > 0:
                modules.append(
                    ModuleCost(
                        name="smalls",
                        hbm_bytes=smalls_b,
                        fp8_flops=0.0,
                        bf16_flops=0.0,
                        ici_bytes=0.0,
                        n_collectives=0,
                    )
                )

    # 5. Optional LM head
    if vocab > 0 and hidden > 0:
        lm_elems = float(vocab * hidden // r_val)
        lm_flops = 2.0 * m_rows * lm_elems
        modules.append(
            ModuleCost(
                name="lm_head",
                hbm_bytes=lm_elems * lm_head_weight_bytes,
                fp8_flops=lm_flops if lm_head_weight_bytes <= 1.0 else 0.0,
                bf16_flops=0.0 if lm_head_weight_bytes <= 1.0 else lm_flops,
                ici_bytes=0.0,
                n_collectives=0,
            )
        )

    return _combine_modules(tuple(modules), chip)


def floor_for_profile(
    model_spec: Any,
    profile: Any,
    *,
    B: int,
    S: int,
    T: int = 1,
    R: int | None = None,
    state_writes: int | None = None,
    chip: ChipSpec = CHIPS["tpu7x"],
    kv_path: str = "fused",
    q_heads_global: int | None = None,
    num_kv_heads_global: int | None = None,
    num_experts_global: int | None = None,
    top_k: int = 10,
    vocab: int = 0,
    include_plan_extras: bool = True,
) -> StepFloor:
    """Build ExecSpec + Plan from Megakernel ModelSpec and Profile, then StepFloor."""
    import importlib

    exec_spec_mod = importlib.import_module("vllm_torchtpu.megakernel.exec_spec")
    ExecSpec = exec_spec_mod.ExecSpec
    planner_mod = importlib.import_module("vllm_torchtpu.megakernel.planner")
    mk_plan = planner_mod.plan

    r_val = int(R if R is not None else getattr(profile, "R", 8) or 8)
    # GQABlock.plan_params requires ex.ctx % kv_blk == 0 (default kv_blk=2048).
    # Align the planning ctx to a multiple of 2048, then price the exact runtime S.
    plan_ctx = max(2048, ((max(1, int(S)) + 2047) // 2048) * 2048)
    base = ExecSpec(R=r_val, ctx=plan_ctx)
    ex = profile.exec_for(base, B=int(B) * int(T), T=int(T))
    p = mk_plan(model_spec, ex, fit_vmem=False)
    return floor_from_plan(
        p,
        ex,
        B=B,
        S=S,
        T=T,
        R=r_val,
        state_writes=state_writes,
        chip=chip,
        kv_path=kv_path,
        q_heads_global=q_heads_global,
        num_kv_heads_global=num_kv_heads_global,
        num_experts_global=num_experts_global,
        top_k=top_k,
        vocab=vocab,
        include_plan_extras=include_plan_extras,
    )


def reconcile_plan_vs_standalone(
    p: Any,
    ex: Any,
    geom: ModelGeometry,
    par: Parallel,
    shape: StepShape,
    dt: Dtypes = DEFAULT_DTYPES,
    chip: ChipSpec = CHIPS["tpu7x"],
    include_plan_extras: bool = False,
) -> tuple[PlanReconciliation, ...]:
    """Compare floor_from_plan against standalone step_floor module-by-module."""
    plan_sf = floor_from_plan(
        p,
        ex,
        B=shape.batch,
        S=shape.context,
        T=shape.rows_per_seq,
        R=par.tp,
        state_writes=shape.state_writes,
        chip=chip,
        kv_path=par.kv_path,
        q_heads_global=geom.num_heads,
        num_kv_heads_global=geom.num_kv_heads,
        num_experts_global=geom.num_experts,
        top_k=geom.top_k,
        vocab=0,
        include_plan_extras=include_plan_extras,
    )
    std_sf = step_floor(geom, chip, par, shape, dt)
    plan_map = {m.name: m for m in plan_sf.modules}
    std_map = {m.name: m for m in std_sf.modules if m.name != "lm_head"}
    names = list(dict.fromkeys([*std_map.keys(), *plan_map.keys()]))
    rows: list[PlanReconciliation] = []
    for name in names:
        pm = plan_map.get(name)
        sm = std_map.get(name)
        pb = pm.hbm_bytes if pm is not None else 0.0
        sb = sm.hbm_bytes if sm is not None else 0.0
        pf = (pm.fp8_flops + pm.bf16_flops) if pm is not None else 0.0
        sf = (sm.fp8_flops + sm.bf16_flops) if sm is not None else 0.0
        rows.append(
            PlanReconciliation(
                module=name,
                plan_hbm_bytes=pb,
                standalone_hbm_bytes=sb,
                delta_hbm_bytes=pb - sb,
                plan_flops=pf,
                standalone_flops=sf,
                delta_flops=pf - sf,
            )
        )
    return tuple(rows)
