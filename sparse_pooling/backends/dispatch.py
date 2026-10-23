"""Plan resolution: a requested ``PhysicalPlan`` + device capabilities + tuning profile -> the
concrete, immutable execution choices of one call (``ResolvedPlan``), with reasons.

Precedence (tests/test_dispatch.py):
  1. numerical requirements: a strict deterministic request (explicit, or the PyTorch global flag)
     admits only reductions with a fixed owner and order; unordered candidates raise, never fall back;
  2. legality: ISA features of the kernel family, compiled instantiations, the device's actual
     opt-in shared memory per block (plus the kernel's static shared memory), the workspace caps;
  3. valid explicit PhysicalPlan overrides (an explicit request that is illegal raises);
  4. the tuning profile's measured choice (backends/profiles);
  5. a conservative eligible fallback (recorded in the reasons / ``grouped_fallback``).

Static facts (capabilities, profile, bucket statistics, work lists) are resolved once per runtime;
the per-call resolution is host-only dictionary work and is cached per (batch, reduction mode,
plan field values), so the PyTorch deterministic flag is re-read on every call and never cached.
"""
from __future__ import annotations

import dataclasses
import math
from types import MappingProxyType
from typing import Dict, Optional, Tuple

from ..config import DET_WORKSPACE_DEFAULT, DX_PLANS, PhysicalPlan
from .capabilities import NATIVE_FAMILY_FEATURES, DeviceCapabilities
from .cuda import variants as V
from .profiles import ProfileSelection, TuningProfile

TOPK_STATIC_SMEM = 256  # fwd_topk_kernel static shared memory (cuobjdump -res-usage: 128 / 256 B)


class UnsupportedError(ValueError):
    """An explicit request that the device / kernels cannot honour."""


# ------------------------------------------------------------------------- helpers
def smem_limit(caps: DeviceCapabilities, static: int = 0) -> int:
    """Largest dynamic shared memory a block may request (opt-in limit minus static use)."""
    return caps.smem_per_block_optin - static


def native_family_reason(caps: DeviceCapabilities) -> str:
    """Why the native CUDA kernel family cannot run on a device ("" if it can)."""
    if caps.backend != "cuda":
        return f"not a CUDA device ({caps.backend})"
    miss = caps.missing(NATIVE_FAMILY_FEATURES)
    if miss:
        return (f"sm_{caps.cc[0]}{caps.cc[1]} lacks {', '.join(miss)} (the native kernels need sm_80+: BF16 "
                f"mma.sync, cp.async)")
    if caps.warp_size != V.WARP_SIZE:
        return f"warp size {caps.warp_size} != 32 (fragment layouts)"
    return ""


def p_int(level) -> int:
    if level.type == "power_mean" and float(level.p).is_integer() and int(level.p) in (1, 2, 4, 8):
        return int(level.p)
    return 0


def epilogue_choice(profile: TuningProfile, requested: str, lv, grad_precision: str = "split",
                    kernel: str = "dx") -> str:
    """Effective dZ epilogue: "auto" asks the profile; a variant the operator does not have falls
    back to the next simpler one (mask -> fast -> generic; all are bitwise identical to generic)."""
    if requested == "auto":
        requested = profile.epilogue(kernel, lv.type, lv.activation, grad_precision)
    act_ok = lv.activation in ("relu", "identity")
    if requested == "mask" and lv.type == "mean" and act_ok:
        return "mask"
    if requested in ("fast", "mask") and ((lv.type in ("mean", "logmeanexp") and act_ok)
                                         or (lv.type == "power_mean" and p_int(lv) != 0)):
        return "fast"
    return "generic"


def implied_epilogue(plan: PhysicalPlan, kernel: str) -> str:
    """The epilogue an explicit option needs when the epilogue itself is "auto" (precedence: a valid
    explicit override constrains automatic fields; the epilogues are bitwise equal to generic):
    the compact dW projection needs the fast dW epilogue, the fused backward mask dX / fast dW."""
    if plan.cuda_grouped_backward in ("fused", "fused_recompute") and plan.grad_precision == "split":
        return "mask" if kernel == "dx" else "fast"  # (the fused kernel exists for split gradients only)
    if kernel == "dw" and plan.cuda_grouped_dw_projection == "compact":
        return "fast"
    return "auto"


def plan_key(plan: PhysicalPlan) -> Tuple:
    return tuple(getattr(plan, f.name) for f in dataclasses.fields(plan))


# ---------------------------------------------------------------------- resolver
class Resolver:
    """Decision functions over (capabilities, profile).  ``rt`` provides layout, level, plan, sg
    (supergroup execution metadata or None), units and the current reduction mode ``det``."""

    def __init__(self, caps: DeviceCapabilities, selection: ProfileSelection):
        self.caps = caps
        self.selection = selection
        self.profile: TuningProfile = selection.profile
        self.limit = smem_limit(caps)

    # ------------------------------------------------------------- epilogues
    def epilogue(self, requested: str, lv, grad_precision: str = "split", kernel: str = "dx", plan=None) -> str:
        if requested == "auto" and plan is not None:
            requested = implied_epilogue(plan, kernel)
        return epilogue_choice(self.profile, requested, lv, grad_precision, kernel)

    # ------------------------------------------------------- generic kernels
    def fwd_cfg(self, plan: PhysicalPlan):
        mt = V.FWD_TILES.get(plan.cuda_block_m)
        if mt is None:
            raise ValueError("cuda_block_m must be 16 or 32")
        return mt, plan.cuda_warps

    def supports_forward(self, rt, track_ids: bool = False) -> bool:
        return self.forward_reason(rt) == ""

    def forward_reason(self, rt) -> str:
        lv, L = rt.level, rt.layout
        fam = native_family_reason(self.caps)
        if fam:
            return fam
        if not L.shared_indices:
            return "Mode A (independent indices) runs on Triton"
        if lv.type == "topk_mean":  # ranks raw z; any activation is applied at finalize
            # k = 32 lists (2 x 32 value/id pairs per lane) exceed the register budget and the
            # insertion cost grows ~k: measured faster on Triton
            if not (L.packed and lv.k <= 16 and L.input_dim % 8 == 0):
                return "Top-K needs packed storage, k <= 16, D % 8 == 0"
            if L.input_dim * 32 * 2 > smem_limit(self.caps, TOPK_STATIC_SMEM):
                return f"Top-K slab of D = {L.input_dim} exceeds the device's shared memory"
            return ""
        if lv.type not in V.CUDA_POOLS:
            return f"no native forward for {lv.type}"
        if lv.activation not in ("identity", "relu"):
            return f"activation {lv.activation} runs on Triton"
        mt, _ = self.fwd_cfg(rt.plan)
        if L.input_dim % 8:
            return "needs D % 8 == 0"
        if L.input_dim * 16 * mt * 2 > self.limit:
            return f"forward slab of D = {L.input_dim} x {16 * mt} tokens exceeds the device's shared memory"
        return ""

    def supports_backward(self, rt) -> bool:
        return self.backward_reason(rt) == ""

    def backward_reason(self, rt) -> str:
        lv, L = rt.level, rt.layout
        fam = native_family_reason(self.caps)
        if fam:
            return fam
        if not (L.packed and lv.type in V.CUDA_BWD_POOLS and lv.activation in ("identity", "relu")):
            return "native backward: packed storage, Mean / LME / power / softmax-weighted, ReLU / identity"
        if L.input_dim % 8:
            return "needs D % 8 == 0"
        if V.dx_smem(L.input_dim) > self.limit:  # also for grouped layers: k_dx is the native dX fallback
            return f"k_dx slab of D = {L.input_dim} exceeds the device's shared memory"
        return ""

    def dw_staging_reason(self, L, sxb) -> str:
        """Why indexed k_dw staging cannot be used ("" if it can): compiled for packed C = 64 Mode B
        with a feature-major input operand (cp.async ring of index rows)."""
        if not (L.packed and L.shared_indices):
            return "needs packed Mode B parameters"
        if L.block not in V.INDEXED_DW_C:
            return f"implemented for C = 64 only (C = {L.block})"
        if sxb != 1:
            return "needs the feature-major input operand (input_layout='feature_major' and B % 8 == 0)"
        if not self.caps.has("cp.async"):
            return "needs cp.async (sm_80+)"
        return ""

    def dw_staging(self, plan: PhysicalPlan, L, sxb) -> Tuple[str, str]:
        """(staging, fallback reason of an explicit 'indexed' request or ""): an explicit legal request
        is honoured; "auto" takes the profile's choice where indexed staging is legal."""
        if plan.cuda_dw_staging == "full":
            return "full", ""
        reason = self.dw_staging_reason(L, sxb)
        if plan.cuda_dw_staging == "indexed":
            return ("full", reason) if reason else ("indexed", "")  # explicit: never fall back silently
        if not reason and self.profile.dw_staging() == "indexed":
            return "indexed", ""
        return "full", ""

    def generic_dw_cfg(self, plan: PhysicalPlan, L, lv, staging: str) -> int:
        if plan.cuda_dw_cfg >= 0:
            cfg = plan.cuda_dw_cfg
        else:
            cfg = self.profile.generic_dw_cfg(L.block, lv.type, staging)
        if cfg == 3 and staging == "full":
            cfg = 0  # same tiling; the register-capped variant exists only for indexed staging
        return cfg

    def dx_warps(self, plan: PhysicalPlan, L, units) -> int:
        return plan.cuda_dx_warps or self.profile.dx_warps(L.block, units.BPS, L.unit_blocks)

    # -------------------------------------------- input-supergroup (grouped) kernels
    def sg_reason(self, rt, phase: str, B: int) -> str:
        """Why the grouped kernel of a phase cannot run for this layer / batch ("" if it can)."""
        sg = getattr(rt, "sg", None)
        lv, L, plan = rt.level, rt.layout, rt.plan
        if sg is None:
            return "layer has no input-supergroup connectivity"
        fam = native_family_reason(self.caps)
        if fam:
            return fam
        if not (L.packed and L.shared_indices):
            return "needs packed Mode B parameters"
        if L.block not in V.SG_C:
            return f"implemented for C = 8 and 64 (C = {L.block})"
        if lv.type != "mean" or lv.activation not in ("relu", "identity"):
            return f"implemented for Mean(ReLU / identity) only (pooling {lv.describe()})"
        if rt.units.S != 1:
            return "semantic level-1 shards are not supported"
        if plan.input_layout != "feature_major" or B % 8:
            return "needs the feature-major input operand (input_layout='feature_major' and B % 8 == 0)"
        if L.input_dim % 8 or L.input_dim > V.SG_MAX_INPUT_DIM:
            return f"needs D % 8 == 0 and D <= {V.SG_MAX_INPUT_DIM} (shared-memory feature map)"
        S = sg.S
        if phase == "fwd":
            T, nw, _ = self.sg_fwd_cfg(rt)
            smem = V.sg_smem("fwd", L.input_dim, S, T, nw)
        elif phase == "dx":
            epi = self.epilogue(plan.cuda_dx_epilogue, lv, plan.grad_precision, "dx", plan)
            if epi == "fast":
                return "the grouped dX implements the generic and mask epilogues only (requested fast)"
            if epi == "mask" and plan.grad_precision != "split":
                return "the grouped dX mask epilogue is compiled for split gradients only"
            T, nw, wg, _ = self.sg_dx_cfg(rt, B)
            if nw % wg:
                return f"cuda_grouped_dx_reduction={wg} must divide the {nw} warps"
            smem = V.sg_smem("dx", L.input_dim, S, T, nw, wg)
        else:
            epi = self.epilogue(plan.cuda_dw_epilogue, lv, plan.grad_precision, "dw", plan)
            if epi == "mask" and L.block != 8:
                return "the grouped dW mask epilogue is compiled for C = 8 only (requested mask)"
            smem = V.sg_smem("dw", L.input_dim, S, dw_cfg=self.sg_dw_cfg(rt))
        if smem > self.limit:
            return (f"{phase} configuration needs {smem / 1024:.1f} KiB of shared memory > "
                    f"{self.limit / 1024:.1f} KiB (device opt-in limit)")
        return ""

    def sg_use(self, rt, phase: str, B: int, fallbacks: Optional[Dict] = None) -> bool:
        """Run the grouped kernel for this phase?  Explicit "grouped" requests that cannot run
        raise; "auto" falls back visibly (``fallbacks[phase]``)."""
        if getattr(rt, "sg", None) is None:
            return False
        req = rt.plan.grouped_request(phase)
        if req == "generic":
            return False
        reason = self.sg_reason(rt, phase, B)
        if reason:
            if req == "grouped":
                raise UnsupportedError(f"grouped {phase} requested but unavailable: {reason}")
            if fallbacks is not None:
                fallbacks[phase] = reason
            return False
        if req == "auto":
            if phase == "dx" and getattr(rt, "det", False):
                return True  # deterministic: the generic k_dx flushes with FP32 atomics
            return self.sg_auto_phase(rt, phase) is not None
        return True

    def sg_auto_phase(self, rt, phase: str):
        """Profile config of a phase (None: the generic kernel is preferred)."""
        return self.profile.sg_phase(rt.layout.block, rt.sg.S, phase, rt.sg.mean_bucket_units(),
                                     rt.plan.grad_precision)

    def sg_fwd_cfg(self, rt):
        """(tokens, warps, chunk units) of the grouped forward: explicit fields, else the profile's,
        with the token tile halved until it fits the device."""
        p = rt.plan
        T0, nw0 = self.sg_auto_phase(rt, "fwd") or self.profile.sg_default("fwd")
        T = p.cuda_grouped_fwd_tokens or T0
        nw = p.cuda_grouped_fwd_warps or nw0
        if not p.cuda_grouped_fwd_tokens:
            while T > 16 and V.sg_smem("fwd", rt.layout.input_dim, rt.sg.S, T, nw) > self.limit:
                T //= 2
        return T, nw, p.cuda_grouped_fwd_chunk_units or self.profile.sg_chunk_units("fwd")

    def sg_dx_plan(self, rt, B: int):
        """(plan, reason) of the grouped dX: a DX_PLANS key or "tiles" (per-field settings)."""
        p = rt.plan
        det = getattr(rt, "det", False)
        if det and p.cuda_grouped_dx_plan == "shared_t64":
            raise UnsupportedError("cuda_grouped_dx_plan='shared_t64' reduces into one shared FP32 copy with CAS "
                                   "(unordered): unsupported with reduction='deterministic' (use 'private_t32' or 'auto')")
        if p.cuda_grouped_dx_plan != "auto":
            return p.cuda_grouped_dx_plan, "explicit cuda_grouped_dx_plan"
        explicit = [f for f in ("tokens", "warps", "reduction", "chunk_units") if getattr(p, f"cuda_grouped_dx_{f}")]
        if p.cuda_grouped_dx_scatter != "auto":
            explicit.append("scatter")
        if explicit:
            return "tiles", "explicit grouped dX fields take precedence (" + ", ".join(explicit) + ")"
        C, S, prec, act = rt.layout.block, rt.sg.S, p.grad_precision, rt.level.activation
        if det:  # numerical requirement first: only warp-private candidates, no fast-mode thresholds
            name = self.profile.dx_det_plan(C, S)
            if name in DX_PLANS and DX_PLANS[name]["reduction"] != 1:
                raise RuntimeError(f"profile {self.profile.name} proposed {name!r} for deterministic mode (shared copies)")
            if name != "tiles":
                return name, (f"deterministic: warp-private copies ({name}, profile {self.profile.name}; the fast "
                              "S = 64 shared plan uses CAS)")
            return "tiles", "deterministic: the per-field tiles with warp-private copies"
        return self.profile.dx_scale_plan(C, S, prec, act, B, rt.sg.bucket_stats)

    def sg_dx_cfg(self, rt, B: int = 0):
        """(tokens, warps, warps per FP32 slab copy, chunk units) of the grouped dX.  A named plan
        fixes the first three; otherwise unset fields come from the profile; if that does not fit
        the device, copies are shared (fast mode only), then tiles halved."""
        p, S, D = rt.plan, rt.sg.S, rt.layout.input_dim
        cu = p.cuda_grouped_dx_chunk_units or self.profile.sg_chunk_units("dx")
        name, _ = self.sg_dx_plan(rt, B)
        if name in DX_PLANS:
            d = DX_PLANS[name]
            return d["tokens"], d["warps"], d["reduction"], cu
        T0, nw0, wg0 = self.sg_auto_phase(rt, "dx") or self.profile.sg_default("dx")
        T = p.cuda_grouped_dx_tokens or T0
        nw = p.cuda_grouped_dx_warps or nw0
        wg = p.cuda_grouped_dx_reduction
        if getattr(rt, "det", False):
            if wg > 1:
                raise UnsupportedError(f"cuda_grouped_dx_reduction={wg} shares FP32 slab copies through CAS "
                                       "(unordered): unsupported with reduction='deterministic' (use 1 or 0 = auto)")
            wg = 1
            if not p.cuda_grouped_dx_tokens:
                while T > 16 and V.sg_smem("dx", D, S, T, nw, wg) > self.limit:
                    T //= 2
        if not wg:
            wg = min(wg0, nw)
            while wg < nw and V.sg_smem("dx", D, S, T, nw, wg) > self.limit:
                wg *= 2
            if not p.cuda_grouped_dx_tokens:
                while T > 16 and V.sg_smem("dx", D, S, T, nw, wg) > self.limit:
                    T //= 2
        return T, nw, wg, cu

    def sg_dx_scatter(self, rt, T: int, nw: int, wg: int, plan: str = "tiles"):
        """(variant, fallback reason) of the grouped dX local accumulation.  An explicit
        unsupported request (or a named plan's scatter) raises; auto keeps the baseline outside the
        profile's measured cases."""
        req = DX_PLANS[plan]["scatter"] if plan in DX_PLANS else rt.plan.cuda_grouped_dx_scatter
        name = req
        if req == "auto":
            name = self.profile.dx_scatter((rt.layout.block, rt.sg.S, T, nw, wg, rt.plan.grad_precision,
                                            rt.level.activation)) or "baseline"
        why = ""
        if name != "baseline":
            if rt.layout.block != 8:
                why = f"implemented for C = 8 only (C = {rt.layout.block})"
            elif T < 32:
                why = f"needs >= 2 m16 token tiles (cuda_grouped_dx_tokens >= 32, got {T})"
            elif name == "swizzle4_capped" and (nw != 8 or T not in (64, 128)):
                why = "swizzle4_capped is compiled for 8 warps and 64 / 128 tokens"
            elif not name.startswith("swizzle4") and wg != 1:
                why = f"{name} needs warp-private copies (cuda_grouped_dx_reduction = 1, got {wg})"
            elif name == "feature_owner" and rt.sg.S > 32:
                why = f"feature_owner keeps S x T accumulators in registers: S <= 32 (S = {rt.sg.S})"
        if why:
            if req != "auto":
                raise UnsupportedError(f"cuda_grouped_dx_scatter={req!r} unavailable: {why}")
            return "baseline", why
        return name, ""

    def sg_dw_cfg(self, rt) -> int:
        p = rt.plan
        if p.cuda_grouped_dw_cfg >= 0:
            return p.cuda_grouped_dw_cfg
        cfg = self.sg_auto_phase(rt, "dw")
        cfg = self.profile.sg_default("dw") if cfg is None else cfg
        if V.sg_smem("dw", rt.layout.input_dim, rt.sg.S, dw_cfg=cfg) > self.limit:
            cfg = 0  # one 16-token tile per stage: the smallest ring (3 S x 16 slabs)
        return cfg

    def sg_dw_projection(self, rt, dw_epi: str):
        """(projection, reason) of the grouped dW: "compact" (dw_block_c8_compact) or "padded"."""
        req = rt.plan.cuda_grouped_dw_projection
        why = ""
        if rt.layout.block != 8:
            why = f"the compact projection is a C = 8 variant (C = {rt.layout.block})"
        elif rt.level.activation != "relu":
            why = f"needs Mean(ReLU) ({rt.level.describe()}: no projection to share)"
        elif dw_epi != "fast":
            why = f"needs the fast dW epilogue (got {dw_epi})"
        if req == "compact":
            if why:
                raise UnsupportedError(f"cuda_grouped_dw_projection='compact' unavailable: {why}")
            return "compact", "explicit"
        if req == "padded":
            return "padded", "explicit"
        if why:
            return "padded", why
        return self.profile.dw_projection(), f"auto (profile {self.profile.name})"

    def sg_fused_reason(self, rt, B: int, dx_epi: str, dw_epi: str) -> str:
        """Why the experimental fused shared-projection backward cannot run ("" if it can)."""
        L, lv, p = rt.layout, rt.level, rt.plan
        if L.block != 8:
            return f"implemented for C = 8 only (C = {L.block})"
        if lv.type != "mean" or lv.activation != "relu":
            return f"implemented for Mean(ReLU) only ({lv.describe()}); Mean(identity) has no ReLU mask to share"
        if p.grad_precision != "split":
            return "implemented for split gradients only"
        if dx_epi != "mask" or dw_epi != "fast":
            return f"needs the mask dX / fast dW epilogues (got {dx_epi} / {dw_epi})"
        for ph in ("dx", "dw"):
            why = self.sg_reason(rt, ph, B)
            if why:
                return f"grouped {ph} unavailable: {why}"
        smem = V.sgf_smem(L.input_dim, rt.sg.S, p.cuda_grouped_fused_cfg)
        if smem > self.limit:
            return (f"fused cfg {p.cuda_grouped_fused_cfg} needs {smem / 1024:.1f} KiB of shared memory > "
                    f"{self.limit / 1024:.1f} KiB")
        return ""

    # ------------------------------------------------------ deterministic reductions
    def det_reason(self, rt, B: int) -> str:
        """Why reduction='deterministic' cannot run for this layer / batch ("" if it can).
        Supported: input-supergroup layers, packed C = 8 / 64 Mode B, Mean(ReLU / identity), split
        or BF16 gradients, feature-major input (B % 8 == 0), grouped dX (forced; warp-private
        copies), grouped or generic forward and dW, separate backward."""
        lv, p = rt.level, rt.plan
        if getattr(rt, "sg", None) is None:
            return "needs input-supergroup connectivity (the generic dX kernel k_dx flushes with FP32 atomics)"
        if lv.type != "mean" or lv.activation not in ("relu", "identity"):
            return f"implemented for Mean(ReLU / identity) (pooling {lv.describe()})"
        if not self.supports_forward(rt, False) or not self.supports_backward(rt):
            return "the CUDA kernels do not support this layer (the Triton kernels use FP32 atomics)"
        if p.grouped_request("dx") == "generic":
            return "explicit generic dX (k_dx flushes with FP32 atomics)"
        if p.cuda_grouped_backward in ("fused", "fused_recompute"):
            return f"cuda_grouped_backward={p.cuda_grouped_backward!r} flushes dX with FP32 atomics"
        why = self.sg_reason(rt, "dx", B)
        if why:
            return f"grouped dX unavailable: {why}"
        return ""  # forward: grouped (ordered partials) or generic (no atomics for Mean); dW: exclusive owners

    @staticmethod
    def det_window(rt, rows: int, T: int, B: int, what: str) -> int:
        """Tokens per window of a deterministic partial stage: the largest multiple of the token
        tile T whose [rows, window] FP32 workspace fits the cap (all of B if it fits)."""
        cap = rt.plan.deterministic_workspace_bytes or DET_WORKSPACE_DEFAULT
        full = math.ceil(B / T) * T
        tw = min(full, (cap // (4 * rows)) // T * T)
        if tw < T:
            raise RuntimeError(f"deterministic {what} workspace: one {T}-token window needs {rows * T * 4} bytes "
                               f"> deterministic_workspace_bytes = {cap} (raise the cap; no atomic fallback)")
        return tw

    # ------------------------------------------------------------ whole plan
    def resolve(self, rt, B: int, part: str) -> "ResolvedPlan":
        """Every backend choice of one call's forward (``part="fwd"``) or backward (``"bwd"``) as an
        immutable record.  The parts are resolved separately, when each runs: an illegal explicit
        backward option raises in backward, as before the refactor."""
        fb: Dict[str, str] = {}
        plan, L, lv = rt.plan, rt.layout, rt.level
        det = bool(getattr(rt, "det", False))
        has_sg = getattr(rt, "sg", None) is not None
        fwd: Dict = {}
        bwd: Dict = {}
        dx: Dict = {}
        dw: Dict = {}
        if part == "fwd":
            reason = self.forward_reason(rt)
            if reason and has_sg and plan.grouped_request("fwd") == "grouped":
                raise UnsupportedError(f"grouped fwd requested but the native forward is unavailable: {reason}")
            fwd.update(backend="cuda" if not reason else "triton", reason=reason)
            if not reason:
                fwd["kernel"] = "generic"
                if has_sg and self.sg_use(rt, "fwd", B, fb):
                    T, nw, cu = self.sg_fwd_cfg(rt)
                    fwd.update(kernel="grouped", tokens=T, warps=nw, chunk_units=cu,
                               smem=V.sg_smem("fwd", L.input_dim, rt.sg.S, T, nw))
                elif lv.type != "topk_mean":
                    mt, nw = self.fwd_cfg(plan)
                    fwd.update(tokens=16 * mt, warps=nw, smem=V.fwd_smem(L.input_dim, mt, nw))
            else:
                fwd["kernel"] = "triton"
        else:
            reason = self.backward_reason(rt)
            for ph in ("dx", "dw"):  # an explicit grouped request is never dropped silently
                if reason and has_sg and plan.grouped_request(ph) == "grouped":
                    raise UnsupportedError(f"grouped {ph} requested but the native backward is unavailable: {reason}")
            bwd.update(backend="cuda" if not reason else "triton", reason=reason)
            if not reason:
                sxb = 1 if (plan.input_layout == "feature_major" and B % 8 == 0) else L.input_dim
                use_dx = self.sg_use(rt, "dx", B, fb)
                use_dw = self.sg_use(rt, "dw", B, fb)
                staging, staging_fb = self.dw_staging(plan, L, sxb)
                dx_epi = self.epilogue(plan.cuda_dx_epilogue, lv, plan.grad_precision, "dx", plan)
                dw_epi = (self.epilogue(plan.cuda_dw_epilogue, lv, plan.grad_precision, "dw", plan)
                          if (staging == "indexed" or use_dw) else "generic")
                dx.update(kernel="grouped" if use_dx else "generic", epilogue=dx_epi,
                          generic_warps=self.dx_warps(plan, L, rt.units))
                dw.update(kernel="grouped" if use_dw else "generic", epilogue=dw_epi, staging=staging,
                          staging_fallback=staging_fb, generic_cfg=self.generic_dw_cfg(plan, L, lv, staging),
                          stages=self.profile.dw_indexed_stages() if staging == "indexed" else None)
                if use_dx:
                    name, why = self.sg_dx_plan(rt, B)
                    T, nw, wg, cu = self.sg_dx_cfg(rt, B)
                    sc, sc_why = self.sg_dx_scatter(rt, T, nw, wg, name)
                    dx.update(plan=name, plan_reason=why, tokens=T, warps=nw, group_warps=wg, chunk_units=cu,
                              scatter=sc, scatter_fallback=sc_why, scatter_status=V.SG_DX_SCATTERS[sc]["status"],
                              smem=V.sg_smem("dx", L.input_dim, rt.sg.S, T, nw, wg))
                else:
                    dx.update(smem=V.dx_smem(L.input_dim))
                if use_dw:
                    cfg = self.sg_dw_cfg(rt)
                    proj, pwhy = self.sg_dw_projection(rt, dw_epi)
                    dw.update(cfg=cfg, projection=proj, projection_reason=pwhy,
                              smem=V.sg_smem("dw", L.input_dim, rt.sg.S, dw_cfg=cfg))
                org = plan.cuda_grouped_backward
                if org in ("fused", "fused_recompute") and has_sg:
                    why = self.sg_fused_reason(rt, B, dx_epi, dw_epi)
                    if why:
                        raise UnsupportedError(f"cuda_grouped_backward={org!r} unavailable: {why}")
                    if not (use_dx and use_dw):
                        raise UnsupportedError(f"cuda_grouped_backward={org!r} needs grouped dX and dW (requested generic)")
                    bwd["organization"] = org
                else:
                    bwd["organization"] = "separate"
        return ResolvedPlan(
            part=part, batch=B, reduction="deterministic" if det else "fast", profile=self.profile.name,
            fwd=MappingProxyType(fwd), bwd=MappingProxyType(bwd), dx=MappingProxyType(dx), dw=MappingProxyType(dw),
            fallbacks=MappingProxyType(fb))


@dataclasses.dataclass(frozen=True)
class ResolvedPlan:
    """Concrete choices of one call.  ``as_dict`` is the machine-readable form recorded in
    ``last_info["execution"]``."""

    part: str
    batch: int
    reduction: str
    profile: str
    fwd: MappingProxyType
    bwd: MappingProxyType
    dx: MappingProxyType
    dw: MappingProxyType
    fallbacks: MappingProxyType

    def as_dict(self) -> Dict:
        if self.part == "fwd":
            return dict(batch=self.batch, reduction=self.reduction, profile=self.profile, **dict(self.fwd),
                        fallbacks=dict(self.fallbacks))
        return dict(batch=self.batch, reduction=self.reduction, profile=self.profile, **dict(self.bwd),
                    dx=dict(self.dx), dw=dict(self.dw), fallbacks=dict(self.fallbacks))


# ------------------------------------------- EXPERIMENTAL generalized fan-in (kgen)
def kgen_reason(caps: DeviceCapabilities, K: int, C: int, S: int, D: int, B: int, level, plan: PhysicalPlan,
                det: bool, cfg=None) -> str:
    """Why the experimental generalized-fan-in kernels (variants.KGEN_*; sparse_pooling.experimental)
    cannot run a call ("" if they can).  Legality only: these kernels have one compiled tiling per K and
    no tuning profile."""
    fam = native_family_reason(caps)
    if fam:
        return fam
    if K not in V.KGEN_K:
        return f"fan-in K = {K} is not compiled (kgen: K in {V.KGEN_K})"
    if C % 8 or not 8 <= C <= V.KGEN_C_MAX:
        return f"C = {C}: kgen needs C % 8 == 0 and 8 <= C <= {V.KGEN_C_MAX}"
    if not (V.KGEN_S_RANGE[0] <= S <= V.KGEN_S_RANGE[1]) or S % 8 or K > S:
        return f"S = {S}: kgen needs S % 8 == 0, {V.KGEN_S_RANGE[0]} <= S <= {V.KGEN_S_RANGE[1]} and K <= S"
    if level.type != "mean" or level.activation not in ("relu", "identity"):
        return f"kgen implements Mean(ReLU / identity) only (pooling {level.describe()})"
    if plan.grad_precision not in ("split", "bf16"):
        return f"grad_precision {plan.grad_precision!r}"
    if det:
        return "kgen reduces with FP32 atomics: reduction='deterministic' is not implemented"
    if plan.input_layout != "feature_major" or B % 8:
        return "kgen needs the feature-major input operand (input_layout='feature_major' and B % 8 == 0)"
    if D % 8 or D > V.SG_MAX_INPUT_DIM:
        return f"needs D % 8 == 0 and D <= {V.SG_MAX_INPUT_DIM} (shared-memory feature map)"
    if cfg is not None:
        lim = smem_limit(caps)
        for kind, gw in (("fwd", 1), ("dx", cfg[3]), ("dw", 1)):
            if V.kgen_smem(kind, D, S, cfg, gw) > lim:
                return f"kgen {kind} needs {V.kgen_smem(kind, D, S, cfg, gw) / 1024:.1f} KiB of shared memory > {lim / 1024:.1f} KiB"
    return ""


def kgen_dx_group_warps(caps: DeviceCapabilities, D: int, S: int, cfg) -> int:
    """Warps per FP32 dX slab copy of kgen: warp-private copies where they fit the device, else shared
    (64-bit CAS) copies, doubling the group until the launch fits."""
    wg, nw, lim = 1, cfg[3], smem_limit(caps)
    while wg < nw and V.kgen_smem("dx", D, S, cfg, wg) > lim:
        wg *= 2
    return wg


# ------------------------------------------------------------------- matrix adapter
def _call(fn, *a):
    try:
        v = fn(*a)
        return list(v) if isinstance(v, tuple) else v
    except UnsupportedError:
        return {"raises": "ValueError"}
    except Exception as e:  # noqa: BLE001
        return {"raises": type(e).__name__}


def matrix_decisions(rt, B: int, caps: Optional[DeviceCapabilities] = None, profile: Optional[str] = None) -> Dict:
    """The decision record of benchmarks/plan_matrix.py (same keys as its legacy adapter)."""
    from .capabilities import synthetic
    from .profiles import select

    caps = caps or synthetic("NVIDIA GeForce RTX 3090", (8, 6), sm_count=82)
    r = Resolver(caps, select(caps, profile or "auto"))
    out = dict(supports_forward=_call(r.supports_forward, rt, False), supports_backward=_call(r.supports_backward, rt))
    sxb = 1 if (rt.plan.input_layout == "feature_major" and B % 8 == 0) else rt.layout.input_dim
    staging = _call(lambda: r.dw_staging(rt.plan, rt.layout, sxb)[0])
    out["dw_staging"] = staging
    if rt.sg is not None:
        out["det_reason_empty"] = _call(lambda: r.det_reason(rt, B) == "")
        for ph in ("fwd", "dx", "dw"):
            out[f"use_{ph}"] = _call(r.sg_use, rt, ph, B)
        out["fwd_cfg"] = _call(r.sg_fwd_cfg, rt)
        out["dx_plan"] = _call(lambda: r.sg_dx_plan(rt, B)[0])
        out["dx_cfg"] = _call(r.sg_dx_cfg, rt, B)
        if isinstance(out["dx_cfg"], list) and not isinstance(out["dx_plan"], dict):
            T, nw, wg, _ = out["dx_cfg"]
            out["dx_scatter"] = _call(lambda: r.sg_dx_scatter(rt, T, nw, wg, out["dx_plan"])[0])
        out["dw_cfg"] = _call(r.sg_dw_cfg, rt)
    sg_dw = out.get("use_dw") is True
    sg_any = sg_dw or out.get("use_dx") is True
    lv, plan = rt.level, rt.plan
    dx_epi = _call(r.epilogue, plan.cuda_dx_epilogue, lv, plan.grad_precision, "dx", plan)
    dw_epi = (_call(r.epilogue, plan.cuda_dw_epilogue, lv, plan.grad_precision, "dw", plan)
              if (staging == "indexed" or sg_dw) else "generic")
    out.update(dx_epilogue=dx_epi, dw_epilogue=dw_epi)
    if rt.sg is not None and sg_dw and isinstance(dw_epi, str):
        out["dw_projection"] = _call(lambda: r.sg_dw_projection(rt, dw_epi)[0])
    if rt.sg is not None and plan.cuda_grouped_backward in ("fused", "fused_recompute") and sg_any:
        out["fused_ok"] = _call(lambda: r.sg_fused_reason(rt, B, dx_epi, dw_epi) == "")
    out["generic_dw_cfg"] = r.generic_dw_cfg(plan, rt.layout, lv, staging)
    out["dx_warps"] = r.dx_warps(plan, rt.layout, rt.units)
    if rt.sg is not None and getattr(rt, "det", False) and out.get("det_reason_empty") is True:
        out["det_window"] = _call(r.det_window, rt, 1024, 64, B, "dx")
    return out
