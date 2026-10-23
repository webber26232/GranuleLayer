"""ctypes runtime of the native CUDA kernels: library registry, ABI checks and launches.

Tensors are passed as raw device pointers with the stream of their device, so no PyTorch C++
extension (and no exact nvcc / PyTorch CUDA version match) is needed; the CUDA driver must still
support the toolkit that compiled the library (README "Requirements").

* ``backend_for(device)`` returns the ``CudaBackend`` of the device's build target: libraries are
  shared between devices of the same target and build identity (immutable code), never between
  different targets or kinds.  Device-specific launch state (shared-memory opt-in, kernel
  attributes) lives per device in the native launch guard (kernels/cuda/launch.cuh).
* Every launch runs under ``torch.cuda.device(<tensor device>)`` on that device's current stream.
* Decisions come from the resolver (backends/dispatch.py) through ``rt.resolve(B)``; this module
  only executes a resolved plan.
"""
from __future__ import annotations

import ctypes
import math
import threading
import warnings
from pathlib import Path
from typing import Dict, Optional

import torch

from ...config import DX_PLANS
from ...kernels import ops
from ..capabilities import DeviceCapabilities, query
from ..dispatch import UnsupportedError
from . import build as B_
from . import variants as V

ABI_VERSION = 3
_ERRORS = {-1: "no compiled instantiation for these parameters", -2: "unsupported variant / layout for this entry point",
           -3: "shared memory above the device's opt-in limit", -4: "invalid warps per slab copy",
           -5: "variant constraint violated (S too large)", -6: "block larger than the kernel / device allows",
           -7: "no current CUDA device (or ordinal >= SP_MAX_DEVICES)", -8: "CUDA function attribute query/set failed",
           -9: "the library has no code for this device (build target mismatch)"}


def _check(err: int, what: str):
    if err != 0:
        msg = _ERRORS.get(err, f"CUDA error {err}" if err > 0 else "error")
        raise RuntimeError(f"{what} failed (code {err}: {msg})")


# --------------------------------------------------------------------------- ABI
class FwdArgs(ctypes.Structure):
    _fields_ = [
        ("X", ctypes.c_void_p), ("sxf", ctypes.c_longlong), ("sxb", ctypes.c_longlong),
        ("idx", ctypes.c_void_p), ("W", ctypes.c_void_p), ("bias", ctypes.c_void_p),
        ("counts", ctypes.c_void_p),
        ("st_m", ctypes.c_void_p), ("st_s", ctypes.c_void_p), ("st_t", ctypes.c_void_p),
        ("st_id", ctypes.c_void_p),
        ("B", ctypes.c_int), ("U", ctypes.c_int), ("S", ctypes.c_int), ("NB", ctypes.c_int),
        ("BPS", ctypes.c_int), ("BPSP", ctypes.c_int), ("D", ctypes.c_int),
        ("scale", ctypes.c_float), ("p", ctypes.c_float), ("p_int", ctypes.c_int),
        ("packed", ctypes.c_int),
    ]


class TopkArgs(ctypes.Structure):
    _fields_ = [("f", FwdArgs), ("st_keys", ctypes.c_void_p), ("st_pre", ctypes.c_void_p),
                ("st_slot", ctypes.c_void_p), ("act", ctypes.c_int)]


class BwdArgs(ctypes.Structure):
    _fields_ = [
        ("X", ctypes.c_void_p), ("sxf", ctypes.c_longlong), ("sxb", ctypes.c_longlong),
        ("idx", ctypes.c_void_p), ("W", ctypes.c_void_p), ("bias", ctypes.c_void_p),
        ("counts", ctypes.c_void_p),
        ("c1", ctypes.c_void_p), ("c2", ctypes.c_void_p), ("c3", ctypes.c_void_p),
        ("dX", ctypes.c_void_p), ("sdf", ctypes.c_longlong), ("sdb", ctypes.c_longlong),
        ("dW", ctypes.c_void_p), ("dB", ctypes.c_void_p),
        ("B", ctypes.c_int), ("U", ctypes.c_int), ("S", ctypes.c_int), ("NB", ctypes.c_int),
        ("BPS", ctypes.c_int), ("D", ctypes.c_int),
        ("scale", ctypes.c_float), ("p", ctypes.c_float), ("p_int", ctypes.c_int),
        ("units_per_cta", ctypes.c_int), ("wunits_per_cta", ctypes.c_int), ("flags", ctypes.c_int),
        ("nonfinite", ctypes.c_void_p),
    ]


class SgArgs(ctypes.Structure):
    _fields_ = [
        ("X", ctypes.c_void_p), ("sxf", ctypes.c_longlong),
        ("idx", ctypes.c_void_p), ("W", ctypes.c_void_p), ("bias", ctypes.c_void_p),
        ("counts", ctypes.c_void_p), ("feat", ctypes.c_void_p), ("items", ctypes.c_void_p),
        ("chunks", ctypes.c_void_p), ("units", ctypes.c_void_p), ("ysum", ctypes.c_void_p),
        ("c1T", ctypes.c_void_p), ("dXT", ctypes.c_void_p), ("dW", ctypes.c_void_p), ("dB", ctypes.c_void_p),
        ("B", ctypes.c_int), ("G", ctypes.c_int), ("D", ctypes.c_int), ("S", ctypes.c_int),
        ("NB", ctypes.c_int), ("NU", ctypes.c_int), ("dx_group_warps", ctypes.c_int), ("flags", ctypes.c_int),
        ("nonfinite", ctypes.c_void_p),
        ("part", ctypes.c_void_p), ("part_t0", ctypes.c_int), ("part_tw", ctypes.c_int),
    ]


STRUCTS = (FwdArgs, TopkArgs, BwdArgs, SgArgs)  # sp_struct_layout ids 0..3
BWD_FAST_DX_EPI, BWD_FAST_DW_EPI, BWD_MASK_DX_EPI, BWD_MASK_DW_EPI, BWD_FALLBACK = 1, 2, 4, 8, 16  # BwdArgs.flags
_EPI_BITS = {"dx": {"fast": BWD_FAST_DX_EPI, "mask": BWD_MASK_DX_EPI},
             "dw": {"fast": BWD_FAST_DW_EPI, "mask": BWD_MASK_DW_EPI}}


def check_abi(lib, bundle: str, kind: str, path: Path):
    """Refuse a library whose ABI version, bundle, kind or struct layouts differ from this bridge."""
    lib.sp_abi_version.restype = ctypes.c_int
    v = lib.sp_abi_version()
    if v != ABI_VERSION:
        raise RuntimeError(f"{path}: native ABI version {v} != {ABI_VERSION} (stale library; clear the build cache)")
    lib.sp_bundle_id.restype = lib.sp_build_kind.restype = ctypes.c_int
    if lib.sp_bundle_id() != B_.BUNDLE_ID[bundle] or lib.sp_build_kind() != B_.KIND_ID[kind]:
        raise RuntimeError(f"{path}: library is bundle {lib.sp_bundle_id()} kind {lib.sp_build_kind()}, "
                           f"expected {bundle} / {kind}")
    f = lib.sp_struct_layout
    f.restype, f.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ctypes.c_longlong), ctypes.c_int]
    for which, st in enumerate(STRUCTS):
        buf = (ctypes.c_longlong * 64)()
        n = f(which, buf, 64)
        want = [ctypes.sizeof(st)] + [getattr(st, name).offset for name, _ in st._fields_]
        if n != len(want) or list(buf[:n]) != want:
            raise RuntimeError(f"{path}: {st.__name__} layout mismatch (native {list(buf[:max(n, 0)])}, ctypes {want})")


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr()) if t is not None else ctypes.c_void_p(0)


def _stream(dev) -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)


# ---------------------------------------------------------------- instrumentation
# tests only: fill value of every deterministic partial workspace before use (None = no fill);
# a NaN poison shows any partial row that is read without having been written
DET_WORKSPACE_FILL = None
# Optional per-phase instrumentation (benchmarks/bench_kernels.py, bench_supergroups.py): when set to
# a list, every phase boundary records a CUDA event on the current stream.  The computation is
# identical with and without it.
PHASE_EVENTS = None
# with PHASE_EVENTS: (name, start event, end event) of repeated sub-phases (the token windows of the
# deterministic stages), summed per name by the benchmarks
PHASE_INTERVALS = None


def _mark(name):
    if PHASE_EVENTS is not None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        PHASE_EVENTS.append((name, e))


_warned = set()  # fallback reasons already reported (one warning each)


# ------------------------------------------------------------------------ libraries
class NativeLibrary:
    """One loaded bundle (immutable code; shareable between devices of its target)."""

    def __init__(self, path: Path, ident: Dict):
        self.path, self.ident = path, ident
        self.lib = ctypes.CDLL(str(path))
        check_abi(self.lib, ident["bundle"], ident["kind"], path)

    def fn(self, name, argtypes):
        f = getattr(self.lib, name)
        f.restype, f.argtypes = ctypes.c_int, argtypes
        return f


_lock = threading.Lock()
_libraries: Dict[tuple, NativeLibrary] = {}
_backends: Dict[tuple, "CudaBackend"] = {}


def load_bundle(bundle: str, target: str) -> NativeLibrary:
    spec = B_.make_spec(bundle, target)
    with _lock:
        ident = spec.identity()
        key = (bundle, target, ident["kind"], ident["library_hash"])
        if key not in _libraries:
            if spec.kind == "ablation":
                warnings.warn(f"loading an ABLATION build ({' '.join(spec.flags)}): results are deliberately wrong; "
                              "for attribution benchmarks only", stacklevel=3)
            path, ident = B_.build(spec)
            _libraries[key] = NativeLibrary(path, ident)
        return _libraries[key]


def backend_for(device) -> "CudaBackend":
    """The CUDA backend of a device (built / loaded on first use; cached per build identity)."""
    caps = query(device)
    target, why = B_.resolve_target(caps)
    lib = load_bundle("production", target)
    with _lock:
        key = (target, lib.ident["kind"], lib.ident["library_hash"])
        if key not in _backends:
            _backends[key] = CudaBackend(lib, target, why)
        return _backends[key]


def build_info(device=None) -> Dict:
    """Build identity of the production library for a device (default: the current device)."""
    be = backend_for(device if device is not None else "cuda")
    return be.build_info()


# ------------------------------------------------------------------------- backend
class CudaBackend:
    _instance = None  # compatibility: CudaBackend.get() returns the current device's backend

    def __init__(self, lib: NativeLibrary, target: str, target_reason: str = ""):
        self.native = lib
        self.lib = lib.lib
        self.target, self.target_reason = target, target_reason
        self._exp: Optional[NativeLibrary] = None
        self._diag: Optional[NativeLibrary] = None
        P, I, VP = ctypes.POINTER, ctypes.c_int, ctypes.c_void_p
        self.f_fwd = lib.fn("sp_fwd", [P(FwdArgs)] + [I] * 8 + [VP])
        self.f_topk = lib.fn("sp_fwd_topk", [P(TopkArgs)] + [I] * 6 + [VP])
        self.f_bwd = lib.fn("sp_bwd", [P(BwdArgs)] + [I] * 8 + [VP])
        self.sg_fns = {}
        for C in V.SG_C:
            self.sg_fns[("fwd", C)] = lib.fn(f"sp_sg_fwd_c{C}", [P(SgArgs)] + [I] * 5 + [VP])
            for act, an in ((0, "a_id"), (1, "a_relu")):
                self.sg_fns[("dx", C, act)] = lib.fn(f"sp_sg_dx_c{C}_{an}", [P(SgArgs)] + [I] * 6 + [VP])
            self.sg_fns[("dw", C)] = lib.fn(f"sp_sg_dw_c{C}", [P(SgArgs)] + [I] * 6 + [VP])
        for act, an in ((0, "a_id"), (1, "a_relu")):  # production scatter layouts (sc 9, 10)
            self.sg_fns[("dxs", 8, act)] = lib.fn(f"sp_sgx_dx_c8_{an}", [P(SgArgs)] + [I] * 7 + [VP])
        self.f_reduce = lib.fn("sp_sg_reduce", [VP, ctypes.c_longlong, VP, VP, I, VP, ctypes.c_longlong, I, I, VP])

    # ---- registry / identity
    @classmethod
    def get(cls, strict: bool = False):
        """Backend of the current CUDA device (compatibility entry point; layers use backend_for)."""
        return backend_for(torch.device("cuda", torch.cuda.current_device()))

    def experimental(self) -> NativeLibrary:
        """Experimental bundle (explicit-only variants), built on first use."""
        if self._exp is None:
            lib = load_bundle("experimental", self.target)
            P, I, VP = ctypes.POINTER, ctypes.c_int, ctypes.c_void_p
            fns = {}
            for act, an in ((0, "a_id"), (1, "a_relu")):
                fns[("dxs", act)] = lib.fn(f"sp_sgxe_dx_c8_{an}", [P(SgArgs)] + [I] * 7 + [VP])
                fns[("dxo", act)] = lib.fn(f"sp_sgo_dx_c8_{an}", [P(SgArgs)] + [I] * 6 + [VP])
            fns["dw"] = lib.fn("sp_sg_dw_c8_exp", [P(SgArgs)] + [I] * 6 + [VP])
            fns["fused"] = lib.fn("sp_sgf_c8", [P(SgArgs)] + [I] * 3 + [VP])
            lib.fns = fns
            self._exp = lib
        return self._exp

    def diagnostics(self) -> NativeLibrary:
        if self._diag is None:
            self._diag = load_bundle("diagnostics", self.target)
        return self._diag

    def build_info(self) -> Dict:
        bundles = {"production": str(self.native.path)}
        if self._exp is not None:
            bundles["experimental"] = str(self._exp.path)
        if self._diag is not None:
            bundles["diagnostics"] = str(self._diag.path)
        i = self.native.ident
        return dict(target=self.target, target_reason=self.target_reason, kind=i["kind"], library=str(self.native.path),
                    library_hash=i["library_hash"], flags=i["flags"], nvcc=i["nvcc"], nvcc_version=i["nvcc_version"],
                    host_compiler=i["host_compiler"], schema=i["schema"], abi=ABI_VERSION, bundles=bundles)

    # ---- compatibility: decision functions now live in the resolver (backends/dispatch.py)
    @staticmethod
    def _resolver(rt):
        """The runtime's resolver; for a stand-in runtime without one (tests / tools), a resolver of
        the device its indices live on under the automatically selected profile."""
        r = getattr(rt, "resolver", None)
        if r is None:
            from ..dispatch import Resolver
            from ..profiles import select

            caps = query(rt.idx.device)
            r = Resolver(caps, select(caps, getattr(rt.plan, "tuning_profile", "auto")))
        return r

    def supports_forward(self, rt, track_ids: bool) -> bool:
        return self._resolver(rt).supports_forward(rt, track_ids)

    def supports_backward(self, rt) -> bool:
        return self._resolver(rt).supports_backward(rt)

    def sg_reason(self, rt, phase: str, B: int) -> str:
        return self._resolver(rt).sg_reason(rt, phase, B)

    def det_reason(self, rt, B: int) -> str:
        return self._resolver(rt).det_reason(rt, B)

    def sg_dx_plan(self, rt, B: int):
        return self._resolver(rt).sg_dx_plan(rt, B)

    def sg_dx_cfg(self, rt, B: int = 0):
        return self._resolver(rt).sg_dx_cfg(rt, B)

    # ----------------------------------------------------------------- forward
    def _topk_forward(self, rt, xop, sxf, sxb, w, b, track_ids, B):
        lv, L, plan = rt.level, rt.layout, rt.plan
        KP = lv.k
        mt, _ = rt.resolver.fwd_cfg(plan)
        if KP >= 16:  # per-lane lists: 2 rows x KP (value, id) pairs in registers
            mt = 1
        nw = 8 if mt == 1 else 16  # compiled tile/warp pairs
        BM = 16 * mt
        U = rt.units.U
        cplan = type(plan)(phys_splits=plan.phys_splits, block_m=BM)
        PS = ops.choose_splits(B, rt.units, lv, cplan, track_ids, rt.workspace_budget, rt.caps.sm_count)
        bpsp = (math.ceil(rt.units.BPS / PS) + 3) // 4 * 4
        dev = xop.device
        st_pre = st_slot = st_keys = None
        if PS == 1:
            st_pre = torch.empty((B, U), device=dev, dtype=torch.float32)
            if track_ids:
                st_slot = torch.empty((B, U, KP), device=dev, dtype=torch.int32)
        else:
            st_keys = torch.empty((PS, B, U, KP), device=dev, dtype=torch.int64)
        f = FwdArgs(
            _ptr(xop), sxf, sxb, _ptr(rt.idx), _ptr(w), _ptr(b), _ptr(rt.counts_i32),
            0, 0, 0, 0, B, U, rt.units.S, L.blocks_per_group, rt.units.BPS, bpsp, L.input_dim,
            1.0, 0.0, 0, int(L.packed),
        )
        ta = TopkArgs(f, _ptr(st_keys), _ptr(st_pre), _ptr(st_slot), lv.act_id)
        _check(self.f_topk(ctypes.byref(ta), KP, L.block, mt, math.ceil(B / BM), PS, nw, _stream(dev)), "sp_fwd_topk")
        if PS == 1:
            rt.last_info.update(phys_splits=1, workspace_bytes=0)
            out = {"pre": st_pre}
            if st_slot is not None:
                out["topk_slot"] = st_slot  # int32: [B, U, k] can be GBs for sharded models
            return out
        rt.last_info.update(phys_splits=PS, workspace_bytes=st_keys.numel() * 8)
        out = ops.merge_finalize({"keys": st_keys}, lv, rt.units.counts)
        if not track_ids:
            out.pop("topk_slot", None)
        return out

    def forward(self, rt, x, w, b, track_ids: bool):
        with torch.cuda.device(x.device):
            return self._forward(rt, x, w, b, track_ids)

    def _forward(self, rt, x, w, b, track_ids: bool):
        lv, L, plan = rt.level, rt.layout, rt.plan
        assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
        B = x.shape[0]
        rp = rt.resolve(B, "fwd")
        rt.note_fallback("fwd", rp)
        if rp.fwd["kernel"] == "grouped":
            return self.sg_forward(rt, x, w, b, rp)
        rt.last_info["forward_kernel"] = "generic"
        if lv.type == "topk_mean":
            if plan.input_layout == "feature_major" and B % 8 == 0:
                xop = x.t().contiguous()
                return self._topk_forward(rt, xop, xop.stride(0), xop.stride(1), w, b, track_ids, B)
            xop = x.contiguous()
            return self._topk_forward(rt, xop, xop.stride(1), xop.stride(0), w, b, track_ids, B)
        mt, nw = rt.resolver.fwd_cfg(plan)
        BM = 16 * mt
        U = rt.units.U
        _mark("fwd.begin")
        # workspace-checked number of physical splits (reuse the Triton policy)
        cplan = type(plan)(phys_splits=plan.phys_splits, block_m=BM)
        PS = ops.choose_splits(B, rt.units, lv, cplan, track_ids, rt.workspace_budget, rt.caps.sm_count)
        bpsp = (math.ceil(rt.units.BPS / PS) + 3) // 4 * 4  # whole work units per split
        if plan.input_layout == "feature_major" and B % 8 == 0:
            xop = x.t().contiguous()
            sxf, sxb = xop.stride(0), xop.stride(1)
        else:
            xop = x.contiguous()
            sxf, sxb = xop.stride(1), xop.stride(0)
        _mark("fwd.layout")
        f32 = dict(device=x.device, dtype=torch.float32)
        st_m = torch.empty((PS, B, U), **f32)
        st_s = torch.empty((PS, B, U), **f32) if lv.type != "max" else None
        st_t = torch.empty((PS, B, U), **f32) if lv.type == "softmax_weighted" else None
        st_id = torch.empty((PS, B, U), device=x.device, dtype=torch.int32) if track_ids else None
        a = FwdArgs(
            _ptr(xop), sxf, sxb, _ptr(rt.idx), _ptr(w), _ptr(b), _ptr(rt.counts_i32),
            _ptr(st_m), _ptr(st_s), _ptr(st_t), _ptr(st_id),
            B, U, rt.units.S, L.blocks_per_group, rt.units.BPS, bpsp, L.input_dim,
            ops.kernel_scale(lv), float(lv.p), ops._p_int(lv), int(L.packed),
        )
        n_tiles = math.ceil(B / BM)
        _check(self.f_fwd(ctypes.byref(a), V.CUDA_POOLS[lv.type], lv.act_id, int(track_ids), L.block, mt,
                          n_tiles, PS, nw, _stream(x.device)), "sp_fwd")
        _mark("fwd.kernel")
        st = {"m": st_m}
        if st_s is not None:
            st["s"] = st_s
        if st_t is not None:
            st["t"] = st_t
        if st_id is not None:
            st["id"] = st_id
        rt.last_info.update(phys_splits=PS, workspace_bytes=sum(t.numel() * t.element_size() for t in st.values()))
        out = ops.merge_finalize(st, lv, rt.units.counts)
        _mark("fwd.finalize")
        return out

    # ------------------------------------------- input-supergroup (grouped) kernels
    def sgx_layout(self, sc: int, tokens: int, S: int) -> torch.Tensor:
        """Device offsets sgx_off<sc, tokens / 16>(f, t) [S, tokens] of a dX slab layout (diagnostics)."""
        f = self.diagnostics().fn("sp_sgx_layout", [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2)
        out = torch.empty((S, tokens), dtype=torch.int32, device="cuda")
        _check(f(sc, tokens // 16, S, _ptr(out), _stream(out.device)), "sp_sgx_layout")
        return out

    def _sg_args(self, rt, xop, w, b, B):
        L, sg = rt.layout, rt.sg
        return SgArgs(_ptr(xop), xop.stride(0), _ptr(rt.idx), _ptr(w), _ptr(b), _ptr(rt.counts_i32),
                      _ptr(sg.feat), 0, 0, 0, 0, 0, 0, 0, 0,
                      B, L.groups, L.input_dim, sg.S, L.blocks_per_group, L.blocks_per_group // L.unit_blocks,
                      1, 0, 0)

    def sg_forward(self, rt, x, w, b, rp=None):
        """Grouped Mean forward: returns {"pre": [B, G]}."""
        lv, L = rt.level, rt.layout
        B = x.shape[0]
        rp = rp or rt.resolve(B, "fwd")
        T, nw, cu = rp.fwd["tokens"], rp.fwd["warps"], rp.fwd["chunk_units"]
        _mark("fwd.begin")
        xop = x.t().contiguous()  # [D, B] (same transpose as the generic path)
        _mark("fwd.layout")
        det = getattr(rt, "det", False)
        ysum = (torch.empty if det else torch.zeros)((L.groups, B), device=x.device, dtype=torch.float32)
        _mark("fwd.zero")
        items, ch = rt.sg.item_chunks(cu)
        a = self._sg_args(rt, xop, w, b, B)
        a.items, a.chunks, a.ysum = _ptr(items), _ptr(ch), _ptr(ysum)
        stream = _stream(x.device)
        fn = self.sg_fns[("fwd", L.block)]
        if det:  # per-item partials, then the fixed-order sum over each output's items
            ws = self._sg_det_run(rt, a, "fwd", cu, items.shape[0], T, B, ysum, L.groups,
                                  lambda gy: fn(ctypes.byref(a), lv.act_id, T // 16, nw, ch.shape[0], gy, stream))
        else:
            _check(fn(ctypes.byref(a), lv.act_id, T // 16, nw, ch.shape[0], math.ceil(B / T), stream), "sp_sg_fwd")
        _mark("fwd.kernel")
        pre = (ysum / rt.units.counts[:, None]).t().contiguous()
        _mark("fwd.finalize")
        rt.last_info.update(phys_splits=1, workspace_bytes=ysum.numel() * 4, forward_kernel="grouped",
                            grouped_fwd=dict(tokens=T, warps=nw, chunk_units=cu, ctas=ch.shape[0] * math.ceil(B / T),
                                             chunks=ch.shape[0], items=items.shape[0], smem=rp.fwd["smem"],
                                             work_list_bytes=items.numel() * items.element_size()
                                             + ch.numel() * ch.element_size(),
                                             reduction=ws if det else "fp32_atomics"))
        return {"pre": pre}

    # ------------------------------------------------ deterministic reductions
    def _sg_det_run(self, rt, a, kind, cu, rows, T, B, out, out_rows, launch):
        """Token-windowed deterministic stage: per window, the first-stage kernel stores the
        partials of its CTAs into exclusive rows of ``part``; sp_sg_reduce then writes
        out[r, window] = fixed-order sum of row r's contributors (SgExec.det_rows)."""
        off, rec = rt.sg.det_rows(kind, cu)
        tw = rt.resolver.det_window(rt, rows, T, B, kind)
        part = torch.empty((rows, tw), device=out.device, dtype=torch.float32)
        if DET_WORKSPACE_FILL is not None:  # tests: poison the workspace (every read row must be written)
            part.fill_(DET_WORKSPACE_FILL)
        stream = _stream(out.device)
        a.part, a.part_tw = _ptr(part), tw
        nwin = 0
        timed = PHASE_EVENTS is not None and PHASE_INTERVALS is not None
        for w0 in range(0, B, tw):
            ntok = min(tw, B - w0)
            a.part_t0 = w0
            if timed:
                ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                ev[0].record()
            _check(launch(math.ceil(ntok / T)), f"deterministic {kind} stage 1")
            if timed:
                ev[1].record()
            _check(self.f_reduce(_ptr(part), tw, _ptr(off), _ptr(rec), out_rows, _ptr(out), out.stride(0), w0, ntok,
                                 stream), f"sp_sg_reduce ({kind})")
            if timed:
                ev[2].record()
                PHASE_INTERVALS.extend([(f"det_{kind}_stage1", ev[0], ev[1]), (f"det_{kind}_reduce", ev[1], ev[2])])
            nwin += 1
        a.part, a.part_t0, a.part_tw = 0, 0, 0
        return dict(mode="ordered_partials", workspace_bytes=part.numel() * 4, window_tokens=tw, windows=nwin,
                    launches=2 * nwin, partial_rows=rows, contributors=rec.numel(),
                    list_bytes=(off.numel() + rec.numel()) * 4)

    def _sg_dx(self, rt, a, dxt, c1T, B, split, dx_epi, stream, rp):
        L = rt.layout
        d = rp.dx
        plan, plan_why = d["plan"], d["plan_reason"]
        T, nw, wg, cu = d["tokens"], d["warps"], d["group_warps"], d["chunk_units"]
        items, ch = rt.sg.item_chunks(cu)
        a.items, a.chunks, a.c1T, a.dXT, a.dx_group_warps = _ptr(items), _ptr(ch), _ptr(c1T), _ptr(dxt), wg
        epi = {"generic": 0, "mask": 2}[dx_epi]
        scatter, why = d["scatter"], d["scatter_fallback"]
        sc = V.SG_DX_SCATTERS[scatter]["sc"]
        act = rt.level.act_id
        gx, gy = ch.shape[0], math.ceil(B / T)
        if sc == 1 and wg == 1:
            sc = 6  # SC_SWZ4P: the swizzle4 layout compiled for warp-private copies
        if sc == 0:
            kernel = f"sg_dx_kernel<{act}, {L.block}, {T // 16}, {nw}, {split}, {epi}>"
            launch = lambda gy_: self.sg_fns[("dx", L.block, act)](ctypes.byref(a), T // 16, nw, split, epi, gx, gy_,
                                                                    stream)
        elif sc == 4:
            ns = 1 if rt.sg.S <= 8 else (2 if rt.sg.S <= 16 else 4)
            kernel = f"sgx_dx_kernel<{act}, {L.block}, {T // 16}, {nw}, {split}, {epi}, 4, {ns}>"
            f = self.experimental().fns[("dxo", act)]
            launch = lambda gy_: f(ctypes.byref(a), T // 16, nw, split, epi, gx, gy_, stream)
        else:
            kernel = f"sgx_dx_kernel<{act}, {L.block}, {T // 16}, {nw}, {split}, {epi}, {sc}, 1>"
            f = self.sg_fns[("dxs", 8, act)] if sc in V.PRODUCTION_SC else self.experimental().fns[("dxs", act)]
            launch = lambda gy_: f(ctypes.byref(a), sc, T // 16, nw, split, epi, gx, gy_, stream)
        det = getattr(rt, "det", False)
        if det:  # per-(chunk, local feature) partials, then the fixed-order sum per input feature
            ws = self._sg_det_run(rt, a, "dx", cu, gx * rt.sg.S, T, B, dxt, L.input_dim, launch)
        else:
            _check(launch(gy), f"sp_sg_dx ({scatter})")
        st = rt.sg.item_chunk_stats(cu)
        bs = rt.sg.bucket_stats
        rt.last_info["grouped_dx"] = dict(plan=plan, plan_reason=plan_why,
                                          mean_units_per_nonempty_bucket=bs["mean_units_per_nonempty_bucket"],
                                          mean_units_per_possible_bucket=bs["mean_units_per_possible_bucket"],
                                          bucket_occupancy=bs["bucket_occupancy"], tokens=T, warps=nw, reduction_group_warps=wg,
                                          chunk_units=cu, units_per_chunk_mean=st["units_per_chunk_mean"],
                                          token_tiles=gy, tail_utilization=B / (gy * T),
                                          ctas=gx * gy, chunks=ch.shape[0], smem=d["smem"],
                                          scatter=scatter, scatter_layout=V.SG_DX_SCATTERS[scatter]["layout"],
                                          scatter_order=V.SG_DX_SCATTERS[scatter]["order"], scatter_fallback=why,
                                          scatter_status=V.SG_DX_SCATTERS[scatter]["status"],
                                          kernel=kernel, epilogue=dx_epi,
                                          work_list_bytes=items.numel() * items.element_size()
                                          + ch.numel() * ch.element_size(),
                                          reduction=ws if det else ("fp32_atomics" if wg == 1 else
                                                                    "shared_cas+fp32_atomics"))
        return dxt.t()

    def _sg_dw(self, rt, a, c1T, split, dw_epi, stream, rp):
        L = rt.layout
        cfg = rp.dw["cfg"]
        nw, wpw = V.SG_DW_CFGS[cfg][:2]
        units, ch = rt.sg.unit_chunks(nw * wpw)
        a.units, a.chunks, a.c1T = _ptr(units), _ptr(ch), _ptr(c1T)
        epi = {"generic": 0, "fast": 1, "mask": 2}[dw_epi]  # mask: attribution control (C = 8; bitwise = fast)
        proj, proj_why = rp.dw["projection"], rp.dw["projection_reason"]
        pj = int(proj == "compact")
        experimental = bool(pj or epi == 2)
        fn = self.experimental().fns["dw"] if experimental else self.sg_fns[("dw", L.block)]
        a.flags = 0
        err = fn(ctypes.byref(a), rt.level.act_id, cfg, split, epi, pj, ch.shape[0], stream)
        if err == 0 and epi:  # generic fallback launch: exits at once unless a coefficient is non-finite
            a.flags = BWD_FALLBACK
            err = self.sg_fns[("dw", L.block)](ctypes.byref(a), rt.level.act_id, cfg, split, 0, 0, ch.shape[0], stream)
        _check(err, "sp_sg_dw")
        nw_, wpw_, ts_, st_, minb_ = V.SG_DW_CFGS[cfg]
        rt.last_info["grouped_dw"] = dict(cfg=cfg, ctas=ch.shape[0], smem=rp.dw["smem"],
                                          projection=proj, projection_reason=proj_why,
                                          status="experimental" if experimental else "production",
                                          kernel=f"sg_dw_kernel<{rt.level.act_id}, {L.block}, {nw_}, {wpw_}, {ts_}, "
                                                 f"{st_}, {str(bool(split)).lower()}, {minb_}, {epi}, {pj}>",
                                          fallback="sg_dw_kernel generic epilogue (BWD_FALLBACK, padded projection)"
                                          if epi else None,
                                          work_list_bytes=units.numel() * units.element_size()
                                          + ch.numel() * ch.element_size())

    def _sg_fused(self, rt, a, dxt, c1T, B, stream, share):
        """Experimental fused backward + the guarded generic fallbacks (run only for non-finite
        coefficients, when the fused kernel exits): exactly one path contributes."""
        L = rt.layout
        cfg = rt.plan.cuda_grouped_fused_cfg
        nw, wpw, ts, _, minb = V.SGF_CFGS[cfg]
        units, ch = rt.sg.unit_chunks(nw * wpw)
        a.units, a.chunks, a.c1T, a.dXT, a.flags = _ptr(units), _ptr(ch), _ptr(c1T), _ptr(dxt), 0
        _check(self.experimental().fns["fused"](ctypes.byref(a), cfg, int(share), ch.shape[0], stream), "sp_sgf_c8")
        _mark("bwd.fused")
        # fallback dX: the original grouped kernel, generic epilogue (private copies where they fit)
        T, nwx, wg = 64, 8, (1 if rt.sg.S <= 32 else 8)
        items, chx = rt.sg.item_chunks(rt.resolver.profile.sg_chunk_units("dx"))
        a.items, a.chunks, a.dx_group_warps, a.flags = _ptr(items), _ptr(chx), wg, BWD_FALLBACK
        _check(self.sg_fns[("dx", L.block, rt.level.act_id)](ctypes.byref(a), T // 16, nwx, 1, 0, chx.shape[0],
                                                              math.ceil(B / T), stream), "sp_sg_dx fallback")
        # fallback dW / db: the grouped dW generic launch (BWD_FALLBACK), cfg with the same ownership
        units_w, chw = rt.sg.unit_chunks(nw * wpw) if (nw, wpw) in ((8, 2), (8, 4)) else rt.sg.unit_chunks(16)
        dwc = {(8, 2): 2, (8, 4): 3}.get((nw, wpw), 2)
        a.units, a.chunks, a.flags = _ptr(units_w), _ptr(chw), BWD_FALLBACK
        _check(self.sg_fns[("dw", L.block)](ctypes.byref(a), rt.level.act_id, dwc, 1, 0, 0, chw.shape[0], stream),
               "sp_sg_dw fallback")
        a.flags = 0
        S = rt.sg.S
        rt.last_info["grouped_fused"] = dict(
            cfg=cfg, share=bool(share), warps=nw, units_per_warp=wpw, window_tokens=16 * ts, min_ctas_per_sm=minb,
            ctas=ch.shape[0], windows=math.ceil(B / (16 * ts)), smem=V.sgf_smem(L.input_dim, S, cfg),
            status="experimental",
            kernel=(f"sgf_kernel<{nw}, {wpw}, {ts}, {V.SGF_CFGS[cfg][3]}, {minb}, {str(bool(share)).lower()}>" if cfg < 6
                    else f"sgf2_kernel<{nw}, {wpw}, {V.SGF_CFGS[cfg][3]}, {minb}, {str(bool(share)).lower()}>"),
            dx_flush_atomics_upper=ch.shape[0] * S * B,
            fallback="sg_dx_kernel generic + sg_dw generic, flags BWD_FALLBACK (run only if coefficients are non-finite)",
            work_list_bytes=units.numel() * units.element_size() + ch.numel() * ch.element_size())
        rt.last_info.pop("grouped_dx", None)
        rt.last_info.pop("grouped_dw", None)
        return dxt.t()

    # ---------------------------------------------------------------- backward
    def backward(self, rt, x, w, b, g_pre, stats, dw_buf, db_buf):
        """Recompute backward: accumulates into dw_buf/db_buf (packed FP32), returns dX [B, D] FP32."""
        with torch.cuda.device(x.device):
            return self._backward(rt, x, w, b, g_pre, stats, dw_buf, db_buf)

    def _backward(self, rt, x, w, b, g_pre, stats, dw_buf, db_buf):
        lv, L, plan = rt.level, rt.layout, rt.plan
        B, D = x.shape
        U = rt.units.U
        rp = rt.resolve(B, "bwd")
        rt.note_fallback("dx", rp)
        rt.note_fallback("dw", rp)
        _mark("bwd.begin")
        c1, c2, c3 = ops.dense_coefficients(lv, g_pre, stats.get("pre"), rt.units.counts, stats.get("L2"))
        c1, c2, c3 = c1.contiguous(), c2.contiguous(), c3.contiguous()
        sg_dx, sg_dw = rp.dx["kernel"] == "grouped", rp.dw["kernel"] == "grouped"
        c1T = c1.t().contiguous() if (sg_dx or sg_dw) else None  # [G, B]: token-contiguous coefficients
        _mark("bwd.coef")
        if plan.input_layout == "feature_major" and B % 8 == 0:
            xop = x.t().contiguous()
            sxf, sxb = xop.stride(0), xop.stride(1)
        else:
            xop = x.contiguous()
            sxf, sxb = xop.stride(1), xop.stride(0)
        _mark("bwd.layout")
        dx = None if sg_dx else torch.zeros((B, D), device=x.device, dtype=torch.float32)
        det = getattr(rt, "det", False)
        if det and not sg_dx:
            raise RuntimeError("deterministic reduction resolved without the grouped dX (internal error)")
        # grouped: [D, B]; the deterministic second stage writes every element (no zeroing)
        dxt = (torch.empty if det else torch.zeros)((D, B), device=x.device, dtype=torch.float32) if sg_dx else None
        _mark("bwd.dx_init")
        n_tiles = math.ceil(B / 16)
        gy = max(1, min(U, math.ceil(rt.resolver.profile.dx_target_ctas() / n_tiles)))  # one dX flush per CTA
        upc = math.ceil(U / gy)
        gy = math.ceil(U / upc)
        staging = rp.dw["staging"]
        rt.last_info.pop("dw_staging_fallback", None)
        if rp.dw["staging_fallback"]:  # explicit request: never fall back silently
            reason = rp.dw["staging_fallback"]
            rt.last_info["dw_staging_fallback"] = reason
            if reason not in _warned:
                _warned.add(reason)
                warnings.warn(f"cuda_dw_staging='indexed' unavailable ({reason}); using the full-slab k_dw", stacklevel=3)
        cfg = rp.dw["generic_cfg"]
        nw, wpw = V.DW_CFGS[cfg]
        ub = L.unit_blocks
        wu_per_cta = nw * wpw
        n_chunks = math.ceil(math.ceil(rt.units.BPS / ub) / wu_per_cta)
        a = BwdArgs(
            _ptr(xop), sxf, sxb, _ptr(rt.idx), _ptr(w), _ptr(b), _ptr(rt.counts_i32),
            _ptr(c1), _ptr(c2), _ptr(c3), _ptr(dx), D if dx is None else dx.stride(1), 1 if dx is None else dx.stride(0),
            _ptr(dw_buf), _ptr(db_buf),
            B, U, rt.units.S, L.blocks_per_group, rt.units.BPS, D,
            ops.kernel_scale(lv), float(lv.p), ops._p_int(lv), upc, wu_per_cta, 0,
        )
        # pooling-specialised dZ epilogues (FastDz in sparse_pool_bwd.cuh): "fast" for Mean and
        # LME (ReLU / identity) and integer-p power mean, "mask" for Mean only; the generic dW
        # variants exist for the indexed kernel only (the grouped dW has generic and fast).
        dx_epi, dw_epi = rp.dx["epilogue"], rp.dw["epilogue"]
        nonfinite = None
        if dx_epi != "generic" or dw_epi != "generic":  # device flag (no host sync): inf/NaN coefficients make the kernels take the generic path
            # scalar inf-norm reductions (NaN propagates and fails the compare): no [B, U] temporaries
            ok = torch.linalg.vector_norm(c1, float("inf")) < 2.0**64
            if lv.type != "mean":  # Mean never reads c2
                ok = ok & (torch.linalg.vector_norm(c2, float("inf")) < float("inf"))
            nonfinite = (~ok).to(torch.uint8)
            a.nonfinite = _ptr(nonfinite)
        split = int(plan.grad_precision == "split")
        stream = _stream(x.device)
        pool = V.CUDA_POOLS[lv.type]
        dx_warps = rp.dx["generic_warps"]

        def launch(kind, cfg_, gx, gy_, flags, what):
            a.flags = flags
            _check(self.f_bwd(ctypes.byref(a), pool, lv.act_id, L.block, split, kind, cfg_, gx, gy_, stream),
                   f"sp_bwd({what})")

        if sg_dx or sg_dw:
            sa = self._sg_args(rt, xop, w, b, B)
            sa.dW, sa.dB, sa.nonfinite = _ptr(dw_buf), _ptr(db_buf), _ptr(nonfinite)
        org = rp.bwd.get("organization", "separate")
        fused = org in ("fused", "fused_recompute")
        rt.last_info["backward_organization"] = org
        if not fused:
            rt.last_info.pop("grouped_fused", None)
        if fused:
            dx = self._sg_fused(rt, sa, dxt, c1T, B, stream, share=(org == "fused"))
            _mark("bwd.dx")
            _mark("bwd.dw")
        elif sg_dx:
            dx = self._sg_dx(rt, sa, dxt, c1T, B, split, dx_epi, stream, rp)
        else:
            dx_flags = _EPI_BITS["dx"].get(dx_epi, 0)
            launch(0, dx_warps, n_tiles, gy, dx_flags, "dx")  # k_dx falls back to the generic body in-kernel
        if not fused:
            _mark("bwd.dx")
        if fused:
            pass
        elif sg_dw:
            self._sg_dw(rt, sa, c1T, split, dw_epi, stream, rp)
        else:
            kind = 1 if staging == "full" else {2: 2, 3: 3}[rt.resolver.profile.dw_indexed_stages()]
            dw_flags = _EPI_BITS["dw"].get(dw_epi, 0)
            launch(kind, cfg, n_chunks, U, dw_flags, "dw")
            if dw_flags:  # generic fallback launch: exits at once unless some coefficient is non-finite
                launch(kind, cfg, n_chunks, U, BWD_FALLBACK, "dw fallback")
        if not fused:
            _mark("bwd.dw")
        rt.last_info.update(dx_kernel="grouped" if sg_dx else "generic", dw_kernel="grouped" if sg_dw else "generic")
        rt.last_info.update(bwd_dx_grid=(n_tiles, gy), bwd_dw_grid=(n_chunks, U), grad_precision=plan.grad_precision,
                            dw_staging=staging, dw_cfg=cfg, dx_warps=dx_warps,
                            dx_epilogue=dx_epi, dw_epilogue=dw_epi)
        return dx
