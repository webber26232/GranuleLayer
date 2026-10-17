"""ctypes bridge to the native CUDA kernels (kernels/cuda/*.cu).

The shared library is compiled with the system nvcc on first use and cached
under kernels/cuda/build/<source-hash>/.  Tensors are passed as raw device
pointers together with the current CUDA stream, so no PyTorch C++ extension
(and no CUDA-version match between nvcc and torch) is required.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from ..config import PhysicalPlan
from . import ops

SRC_DIR = Path(__file__).parent / "cuda"
ARCH = os.environ.get("SPARSE_POOL_ARCH", "sm_86")
NVCC_FLAGS = ["-O3", f"-arch={ARCH}", "--use_fast_math", "-lineinfo", "-std=c++17", "-Xcompiler", "-fPIC"]
# extra flags, e.g. the SP_ABLATE_* attribution ablations in sparse_pool_bwd.cuh (benchmarks only:
# ablated kernels compute wrong gradients); part of the build hash, so they get their own library
NVCC_FLAGS += os.environ.get("SPARSE_POOL_NVCC_EXTRA", "").split()
_lock = threading.Lock()


def _sources():
    return sorted(SRC_DIR.glob("*.cu")) + sorted(SRC_DIR.glob("*.cuh"))


def _nvcc():
    for cand in (os.environ.get("NVCC"), shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"):
        if cand and Path(cand).exists():
            return cand
    raise RuntimeError("nvcc not found (set NVCC=/path/to/nvcc)")


def _include_closure(src: Path, seen=None):
    """Transitive local #include "..." closure of a source file."""
    seen = set() if seen is None else seen
    for line in src.read_text().splitlines():
        line = line.strip()
        if line.startswith('#include "'):
            dep = SRC_DIR / line.split('"')[1]
            if dep.exists() and dep not in seen:
                seen.add(dep)
                _include_closure(dep, seen)
    return seen


def _unit_hash(src: Path) -> str:
    h = hashlib.sha1(" ".join(NVCC_FLAGS).encode())
    for p in [src] + sorted(_include_closure(src)):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def build_library(verbose: bool = False) -> Path:
    """Compile (incrementally: objects are cached per translation unit, keyed by
    the hash of its #include closure) and link libsparse_pool.so."""
    units = sorted(SRC_DIR.glob("*.cu"))
    hashes = {u: _unit_hash(u) for u in units}
    lib_hash = hashlib.sha1("".join(hashes[u] for u in units).encode()).hexdigest()[:16]
    out_dir = SRC_DIR / "build" / lib_hash
    lib = out_dir / "libsparse_pool.so"
    if lib.exists():
        return lib
    obj_dir = SRC_DIR / "build" / "objs"
    obj_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    nvcc = _nvcc()

    def compile_one(src: Path):
        obj = obj_dir / f"{src.stem}-{hashes[src]}.o"
        if obj.exists():
            return obj
        tmp = obj.with_suffix(".o.tmp")
        cmd = [nvcc, *NVCC_FLAGS, "-c", str(src), "-o", str(tmp)]
        if verbose:
            print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc failed for {src.name}:\n{r.stderr}")
        tmp.rename(obj)
        return obj

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        objs = list(ex.map(compile_one, units))
    tmp = out_dir / "libsparse_pool.so.tmp"
    r = subprocess.run([nvcc, "-shared", f"-arch={ARCH}", *map(str, objs), "-o", str(tmp)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"link failed:\n{r.stderr}")
    tmp.rename(lib)
    return lib


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
        ("units_per_cta", ctypes.c_int), ("wunits_per_cta", ctypes.c_int),
    ]


_CUDA_POOLS = {"max": 0, "mean": 2, "logmeanexp": 3, "power_mean": 4, "softmax_weighted": 5}
_CUDA_BWD_POOLS = ("mean", "logmeanexp", "power_mean", "softmax_weighted")
# k_dw tiling per C: cfg -> (warps, work units per warp)
DW_CFGS = {0: (8, 2), 1: (8, 4), 2: (16, 1), 3: (8, 2)}  # 3: indexed staging only, register cap for 2 CTAs/SM
DW_CFG = {8: 1, 16: 1, 32: 1, 64: 1}  # measured at H=100M, B=8192 (8 warps x 4 work units)
DW_INDEXED_STAGES = 2  # cp.async ring depth of the indexed k_dw (2 or 3)
# default indexed k_dw tiling per operator at C = 64 (H = 1M-100M, B = 8192): the
# register-capped 2-CTA/SM cfg 3 wins for Mean / LME; it spills more with the
# softmax-weighted / power epilogues, which keep 8 warps x 4 work units
DW_CFG_INDEXED = {"mean": 3, "logmeanexp": 3, "power_mean": 1, "softmax_weighted": 1}


# Optional per-phase instrumentation (benchmarks/bench_kernels.py): when set to a
# list, every phase boundary records a CUDA event on the current stream.  The
# computation is identical with and without it.
PHASE_EVENTS = None


def _mark(name):
    if PHASE_EVENTS is not None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        PHASE_EVENTS.append((name, e))


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr()) if t is not None else ctypes.c_void_p(0)


class CudaBackend:
    _instance = None

    def __init__(self, lib_path: Path):
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.sp_fwd.restype = ctypes.c_int
        self.lib.sp_fwd.argtypes = [ctypes.POINTER(FwdArgs)] + [ctypes.c_int] * 8 + [ctypes.c_void_p]
        self.lib.sp_fwd_args_size.restype = ctypes.c_int
        assert self.lib.sp_fwd_args_size() == ctypes.sizeof(FwdArgs), "FwdArgs layout mismatch"
        self.lib.sp_fwd_topk.restype = ctypes.c_int
        self.lib.sp_fwd_topk.argtypes = [ctypes.POINTER(TopkArgs)] + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        self.lib.sp_topk_args_size.restype = ctypes.c_int
        assert self.lib.sp_topk_args_size() == ctypes.sizeof(TopkArgs), "TopkArgs layout mismatch"
        self.lib.sp_bwd.restype = ctypes.c_int
        self.lib.sp_bwd.argtypes = [ctypes.POINTER(BwdArgs)] + [ctypes.c_int] * 8 + [ctypes.c_void_p]
        self.lib.sp_bwd_args_size.restype = ctypes.c_int
        assert self.lib.sp_bwd_args_size() == ctypes.sizeof(BwdArgs), "BwdArgs layout mismatch"

    @classmethod
    def get(cls, strict: bool = False):
        with _lock:
            if cls._instance is None:
                cls._instance = CudaBackend(build_library())
            return cls._instance

    # ----------------------------------------------------------------- forward
    @staticmethod
    def _cfg(plan: PhysicalPlan):
        mt = {16: 1, 32: 2}.get(plan.cuda_block_m)
        if mt is None:
            raise ValueError("cuda_block_m must be 16 or 32")
        return mt, plan.cuda_warps

    def supports_forward(self, rt, track_ids: bool) -> bool:
        lv, L = rt.level, rt.layout
        if not L.shared_indices:
            return False
        if lv.type == "topk_mean":  # ranks raw z; any activation is applied at finalize
            # k = 32 lists (2 x 32 value/id pairs per lane) exceed the register
            # budget and the insertion cost grows ~k: measured faster on Triton
            return (L.packed and lv.k <= 16 and L.input_dim % 8 == 0
                    and L.input_dim * 32 * 2 <= 99 * 1024)
        if lv.type not in _CUDA_POOLS:
            return False
        if lv.activation not in ("identity", "relu"):
            return False
        mt, _ = self._cfg(rt.plan)
        return L.input_dim % 8 == 0 and L.input_dim * 16 * mt * 2 <= 99 * 1024

    def _topk_forward(self, rt, xop, sxf, sxb, w, b, track_ids, B):
        lv, L, plan = rt.level, rt.layout, rt.plan
        KP = lv.k
        mt, _ = self._cfg(plan)
        if KP >= 16:  # per-lane lists: 2 rows x KP (value, id) pairs in registers
            mt = 1
        nw = 8 if mt == 1 else 16  # compiled tile/warp pairs
        BM = 16 * mt
        U = rt.units.U
        cplan = PhysicalPlan(phys_splits=plan.phys_splits, block_m=BM)
        PS = ops.choose_splits(B, rt.units, lv, cplan, track_ids, rt.workspace_budget)
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
        err = self.lib.sp_fwd_topk(ctypes.byref(ta), KP, L.block, mt, math.ceil(B / BM), PS, nw,
                                   ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if err != 0:
            raise RuntimeError(f"sp_fwd_topk failed (code {err})")
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
        lv, L, plan = rt.level, rt.layout, rt.plan
        assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
        B = x.shape[0]
        if lv.type == "topk_mean":
            if plan.input_layout == "feature_major" and B % 8 == 0:
                xop = x.t().contiguous()
                return self._topk_forward(rt, xop, xop.stride(0), xop.stride(1), w, b, track_ids, B)
            xop = x.contiguous()
            return self._topk_forward(rt, xop, xop.stride(1), xop.stride(0), w, b, track_ids, B)
        mt, nw = self._cfg(plan)
        BM = 16 * mt
        U = rt.units.U
        _mark("fwd.begin")
        # workspace-checked number of physical splits (reuse the Triton policy)
        cplan = PhysicalPlan(phys_splits=plan.phys_splits, block_m=BM)
        PS = ops.choose_splits(B, rt.units, lv, cplan, track_ids, rt.workspace_budget)
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
        err = self.lib.sp_fwd(ctypes.byref(a), _CUDA_POOLS[lv.type], lv.act_id, int(track_ids), L.block, mt,
                              n_tiles, PS, nw, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if err != 0:
            raise RuntimeError(f"sp_fwd failed (code {err})")
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

    # ---------------------------------------------------------------- backward
    def supports_backward(self, rt) -> bool:
        lv, L = rt.level, rt.layout
        return (L.packed and lv.type in _CUDA_BWD_POOLS and lv.activation in ("identity", "relu")
                and L.input_dim % 8 == 0 and L.input_dim * 16 * 6 <= 99 * 1024)

    @staticmethod
    def dw_staging(plan: PhysicalPlan, L, device, sxb) -> str:
        """Indexed k_dw staging is implemented for packed C = 64 Mode B with a
        feature-major input operand and validated on sm_86; everything else
        uses the original full-slab kernel."""
        if plan.cuda_dw_staging == "full":
            return "full"
        ok = (L.block == 64 and L.packed and L.shared_indices and sxb == 1
              and torch.cuda.get_device_capability(device) == (8, 6))
        return "indexed" if ok else "full"

    def backward(self, rt, x, w, b, g_pre, stats, dw_buf, db_buf):
        """Recompute backward: accumulates into dw_buf/db_buf (packed FP32),
        returns dX [B, D] FP32."""
        lv, L, plan = rt.level, rt.layout, rt.plan
        B, D = x.shape
        U = rt.units.U
        _mark("bwd.begin")
        c1, c2, c3 = ops.dense_coefficients(lv, g_pre, stats.get("pre"), rt.units.counts, stats.get("L2"))
        c1, c2, c3 = c1.contiguous(), c2.contiguous(), c3.contiguous()
        _mark("bwd.coef")
        if plan.input_layout == "feature_major" and B % 8 == 0:
            xop = x.t().contiguous()
            sxf, sxb = xop.stride(0), xop.stride(1)
        else:
            xop = x.contiguous()
            sxf, sxb = xop.stride(1), xop.stride(0)
        _mark("bwd.layout")
        dx = torch.zeros((B, D), device=x.device, dtype=torch.float32)
        _mark("bwd.dx_init")
        n_tiles = math.ceil(B / 16)
        gy = max(1, min(U, math.ceil(4096 / n_tiles)))  # ~4096 CTAs; one dX flush per CTA
        upc = math.ceil(U / gy)
        gy = math.ceil(U / upc)
        staging = self.dw_staging(plan, L, x.device, sxb)
        cfg = plan.cuda_dw_cfg if plan.cuda_dw_cfg >= 0 else DW_CFG[L.block]
        if plan.cuda_dw_cfg < 0 and staging == "indexed":
            cfg = DW_CFG_INDEXED[lv.type]
        if plan.cuda_dw_cfg < 0 and L.block == 8 and lv.type != "mean":
            cfg = 0  # 8 x 4 spills with the exp/power epilogues at C = 8 (measured 1.6x slower)
        if cfg == 3 and staging == "full":
            cfg = 0  # same tiling; the register-capped variant exists only for indexed staging
        nw, wpw = DW_CFGS[cfg]
        ub = L.unit_blocks
        wu_per_cta = nw * wpw
        n_chunks = math.ceil(math.ceil(rt.units.BPS / ub) / wu_per_cta)
        a = BwdArgs(
            _ptr(xop), sxf, sxb, _ptr(rt.idx), _ptr(w), _ptr(b), _ptr(rt.counts_i32),
            _ptr(c1), _ptr(c2), _ptr(c3), _ptr(dx), dx.stride(1), dx.stride(0),
            _ptr(dw_buf), _ptr(db_buf),
            B, U, rt.units.S, L.blocks_per_group, rt.units.BPS, D,
            ops.kernel_scale(lv), float(lv.p), ops._p_int(lv), upc, wu_per_cta,
        )
        split = int(plan.grad_precision == "split")
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        pool = _CUDA_POOLS[lv.type]
        # k_dx is held at 1 CTA/SM by its 96 KiB slab: at 64 registers, 32 warps fill the register
        # file and hide more latency (measured 9-14 % faster at C = 64, neutral at C = 8)
        dx_warps = plan.cuda_dx_warps or (32 if L.block == 64 else 16)
        err = self.lib.sp_bwd(ctypes.byref(a), pool, lv.act_id, L.block, split, 0, dx_warps,
                              n_tiles, gy, stream)
        if err != 0:
            raise RuntimeError(f"sp_bwd(dx) failed (code {err})")
        _mark("bwd.dx")
        kind = 1 if staging == "full" else {2: 2, 3: 3}[DW_INDEXED_STAGES]
        err = self.lib.sp_bwd(ctypes.byref(a), pool, lv.act_id, L.block, split, kind, cfg, n_chunks, U, stream)
        if err != 0:
            raise RuntimeError(f"sp_bwd(dw) failed (code {err})")
        _mark("bwd.dw")
        rt.last_info.update(bwd_dx_grid=(n_tiles, gy), bwd_dw_grid=(n_chunks, U), grad_precision=plan.grad_precision,
                            dw_staging=staging, dw_cfg=cfg, dx_warps=dx_warps)
        return dx
