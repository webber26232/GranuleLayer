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
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from ..config import DX_PLANS, PhysicalPlan
from . import ops

SRC_DIR = Path(__file__).parent / "cuda"
ARCH = os.environ.get("SPARSE_POOL_ARCH", "sm_86")
NVCC_FLAGS = ["-O3", f"-arch={ARCH}", "--use_fast_math", "-lineinfo", "-std=c++17", "-Xcompiler", "-fPIC"]
# extra flags, e.g. the SP_ABLATE_* attribution ablations in sparse_pool_bwd.cuh (benchmarks only:
# ablated kernels compute wrong gradients); part of the build hash, so they get their own library
NVCC_FLAGS += os.environ.get("SPARSE_POOL_NVCC_EXTRA", "").split()
_lock = threading.Lock()
_warned = set()  # fallback reasons already reported (one warning each)


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
    ]


# grouped (input-supergroup) dW tilings: cfg -> (warps, units per warp, 16-token tiles per
# stage, stages, min CTAs/SM); must match SG_DW_CFGS in kernels/cuda/sg_dispatch.cuh
SG_DW_CFGS = {0: (8, 2, 1, 3, 2), 1: (8, 4, 1, 3, 1), 2: (8, 2, 4, 2, 2), 3: (8, 4, 4, 2, 1)}
SG_MAX_SMEM = 99 * 1024
# Measured defaults of the grouped kernels per (C, S) (H = 10M, M = 8192 screens;
# results/INPUT_SUPERGROUP_OPTIMIZATION.md): fwd (tokens, warps) | None, dX (tokens, warps,
# warps per FP32 slab copy) | None, dW cfg | None.  None = the generic kernel is faster for
# that phase, so backend "auto" keeps it (hybrid plans).  Other S use the nearest smaller key.
SG_AUTO = {
    (64, 8): ((128, 4), None, None), (64, 64): ((128, 4), None, None), (64, 128): ((128, 8), None, None),
    (64, 256): ((64, 8), None, None), (64, 512): (None, None, None),
    (8, 8): ((128, 8), (128, 16, 1), 2), (8, 16): ((128, 8), (64, 8, 1), 2), (8, 32): ((128, 8), (32, 8, 1), 2),
    (8, 64): ((128, 8), (64, 8, 8), 2), (8, 128): ((128, 8), (32, 8, 8), 2), (8, 256): ((64, 8), None, 0),
    (8, 512): (None, None, None),
}
SG_CHUNK_UNITS = {"fwd": 2048, "dx": 4096}
# C = 64 with large buckets (mean work units per nonempty (output, supergroup) bucket >= 3; H = 100M:
# ~6, H = 10M: ~1): the grouped dX (all S <= 64) and the grouped dW (S <= 16) are faster at
# H = 100M (-15 ms each, 4 interleaved rounds, +-2 ms) although slower at H = 10M; with BF16
# gradients the grouped dX (generic epilogue) is slower (614 vs 599 ms): split only
SG_AUTO_C64_LARGE_BUCKETS = {"dx": ((64, 8, 8), 64, ("split",)), "dw": (2, 16, ("split", "bf16")),
                             "min_bucket_units": 3.0}


# grouped dX local-accumulation variants (PhysicalPlan.cuda_grouped_dx_scatter): kernel id
# (sparse_pool_sg_dxs.cuh SC_*), slab layout, and FP32 summation-order category relative to
# the baseline ("same": every (feature, token) of a warp's copy receives the same additions in
# the same order; "mma_accumulate": contributions are summed inside the MMA accumulators)
SG_DX_SCATTERS = {
    "baseline": dict(sc=0, layout="float2 pairs, 2 bank groups (sg_dxs_off)", order="same"),
    "baseline_private": dict(sc=5, layout="float2 pairs, 2 bank groups; compile-time warp-private (control)",
                             order="same"),
    "swizzle4": dict(sc=1, layout="float2 pairs, 4 bank groups", order="same"),  # sc 6 when warp-private
    "swizzle4_capped": dict(sc=10, layout="float2 pairs, 4 bank groups; <= 80 registers (3 CTAs / SM at 8 warps)",
                            order="same"),
    "vectorized": dict(sc=2, layout="float4 two-tile records, 4 bank groups", order="same"),
    "vectorized_transpose": dict(sc=7, layout="float4 two-tile records after an 8-shuffle transpose, conflict-free",
                                 order="same"),
    "vectorized_hoisted": dict(sc=8, layout="float4 two-tile records, 4 bank groups; per-block record addresses",
                               order="same"),
    "vectorized_transpose_hoisted": dict(sc=9, layout="float4 records after a shuffle transpose; per-block "
                                         "record addresses", order="same"),
    "vectorized_parity": dict(sc=3, layout="float4 two-tile records, 2 bank groups", order="same"),
    "feature_owner": dict(sc=4, layout="register accumulators [T, S] (MMA-routed), one store per CTA",
                          order="mma_accumulate"),
}
# measured "auto" choice per (C, S, dX tokens, warps, warps per copy, precision, activation) =
# the auto dX tiles of SG_AUTO; anything not listed keeps the baseline.  Mean(ReLU), corrected
# harness (results/C8_DX_SCATTER_OPTIMIZATION.md): S = 16 / 32 at H = 10M and 100M (split, 3 seeds),
# M = 1024, BF16 gradients at 100M; S = 64 at H = 10M and 100M (split; the warp-private T = 32
# vector layout is faster at 100M but slower at 10M, so it stays an explicit choice)
SG_AUTO_DX_SCATTER = {
    (8, 16, 64, 8, 1, "split", "relu"): "vectorized_transpose_hoisted",
    (8, 16, 64, 8, 1, "bf16", "relu"): "vectorized_transpose_hoisted",
    (8, 32, 32, 8, 1, "split", "relu"): "vectorized_transpose_hoisted",
    (8, 32, 32, 8, 1, "bf16", "relu"): "vectorized_transpose_hoisted",
    (8, 64, 64, 8, 8, "split", "relu"): "swizzle4_capped",
}
# scale-aware whole-plan choice (PhysicalPlan.cuda_grouped_dx_plan = "auto" with every grouped dX
# field automatic; results/SCALE_DISPATCH_FUSED_BACKWARD.md section 5).  A rule applies to its (C,
# S, precisions, activation, device capability) and M >= min_tokens; it selects ``plan`` when the
# mean work units per nonempty (output, supergroup) bucket (cached at construction) reaches
# min_units_per_bucket, else ``below``.  Anything else keeps "tiles" (the per-field defaults).
# Measured (H = 1M-100M, G = 256 / 1024, R = 8 / 16 / 32, M = 1024-8192): the private T = 32 plan
# gains with longer buckets (split: +16 % at 0.12 units / bucket ... -3.5 % at 14; crossover ~3);
# thresholds keep the ambiguous region on the shared plan.  BF16: the shared plan beats the
# previous BF16 default (the original scatter) by 5.7-7.3 % at H = 1M-100M; crossover ~1.5.
SG_AUTO_DX_PLAN = [
    dict(C=8, S=64, precisions=("split",), act="relu", cc=(8, 6), min_tokens=1024, min_units_per_bucket=4.0,
         plan="private_t32", below="tiles"),
    dict(C=8, S=64, precisions=("bf16",), act="relu", cc=(8, 6), min_tokens=1024, min_units_per_bucket=2.0,
         plan="private_t32", below="shared_t64"),
]


# experimental fused shared-projection backward tilings (kernels/cuda/sg_bwdf_c8.cu SGF_CFGS):
# cfg -> (warps, units per warp, 16-token tiles per window / ring stage, stages, min CTAs / SM)
# (0-5: v1, tile-outer loops; 6-9: v2, block-outer / tile-inner 32-token windows with vector dX records)
SGF_CFGS = {0: (8, 2, 4, 2, 1), 1: (8, 2, 4, 2, 2), 2: (8, 4, 4, 2, 1), 3: (8, 2, 2, 2, 2), 4: (16, 2, 2, 2, 1),
            5: (8, 4, 2, 2, 1), 6: (8, 2, 2, 2, 2), 7: (16, 1, 2, 2, 1), 8: (8, 2, 2, 2, 1), 9: (8, 2, 2, 3, 2)}


def sgf_smem(D: int, S: int, cfg: int) -> int:
    nw, _, ts, st, _ = SGF_CFGS[cfg]
    return (D * 2 + 15) // 16 * 16 + st * ts * S * 16 * 2 + nw * S * 16 * ts * 4


def sg_auto(C: int, S: int):
    keys = sorted(k[1] for k in SG_AUTO if k[0] == C)
    k = max([s_ for s_ in keys if s_ <= S], default=keys[0])
    return SG_AUTO[(C, k)]



def sg_smem(kind: str, D: int, S: int, tokens: int = 16, warps: int = 8, group_warps: int = 1, dw_cfg: int = 0) -> int:
    """Dynamic shared memory of a grouped kernel launch (sparse_pool_sg.cuh)."""
    m = (D * 2 + 15) // 16 * 16
    if kind == "fwd":
        return m + S * tokens * 2
    if kind == "dx":
        return m + S * tokens * 2 + (warps // group_warps) * S * tokens * 4
    _, _, ts, st, _ = SG_DW_CFGS[dw_cfg]
    return m + st * ts * S * 16 * 2


_CUDA_POOLS = {"max": 0, "mean": 2, "logmeanexp": 3, "power_mean": 4, "softmax_weighted": 5}
_CUDA_BWD_POOLS = ("mean", "logmeanexp", "power_mean", "softmax_weighted")
BWD_FAST_DX_EPI, BWD_FAST_DW_EPI, BWD_MASK_DX_EPI, BWD_MASK_DW_EPI, BWD_FALLBACK = 1, 2, 4, 8, 16  # BwdArgs.flags
_EPI_BITS = {"dx": {"fast": BWD_FAST_DX_EPI, "mask": BWD_MASK_DX_EPI},
             "dw": {"fast": BWD_FAST_DW_EPI, "mask": BWD_MASK_DW_EPI}}
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


def dx_warps_default(C: int, blocks_per_unit: int, unit_blocks: int) -> int:
    """k_dx is held at 1 CTA/SM by its 96 KiB slab; at 64 registers 32 warps fill the register
    file and hide more latency (9-13 % faster dX at C = 64, neutral at C = 8).  Warps take whole
    work units of a pooling unit, so 32 warps only help when a unit has >= 32 work units
    (measured crossover: H = 1M slower, H >= 2M faster, at M = 1024 and 8192)."""
    return 32 if C == 64 and blocks_per_unit // unit_blocks >= 32 else 16


def auto_epilogue(kernel: str, lv, grad_precision: str) -> str:
    """Measured defaults (results/BACKWARD_COMPUTE_OPTIMIZATION.md): dW "fast" for every
    operator that has it; dX "mask" for Mean with split gradients, "generic" for Mean with
    BF16 gradients (the short BF16 generic path is faster there), "fast" otherwise."""
    if kernel == "dx" and lv.type == "mean":
        return "mask" if grad_precision == "split" else "generic"
    return "fast"


def _epilogue(requested: str, lv, grad_precision: str = "split", kernel: str = "dx") -> str:
    if requested == "auto":
        requested = auto_epilogue(kernel, lv, grad_precision)
    act_ok = lv.activation in ("relu", "identity")
    if requested == "mask" and lv.type == "mean" and act_ok:
        return "mask"
    if requested in ("fast", "mask") and ((lv.type in ("mean", "logmeanexp") and act_ok)
                                         or (lv.type == "power_mean" and ops._p_int(lv) != 0)):
        return "fast"
    return "generic"


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
        self.lib.sp_sg_args_size.restype = ctypes.c_int
        assert self.lib.sp_sg_args_size() == ctypes.sizeof(SgArgs), "SgArgs layout mismatch"
        self.sg_fns = {}
        for C in (8, 64):
            f = getattr(self.lib, f"sp_sg_fwd_c{C}")
            f.restype, f.argtypes = ctypes.c_int, [ctypes.POINTER(SgArgs)] + [ctypes.c_int] * 5 + [ctypes.c_void_p]
            self.sg_fns[("fwd", C)] = f
            for act, an in ((0, "a_id"), (1, "a_relu")):
                f = getattr(self.lib, f"sp_sg_dx_c{C}_{an}")
                f.restype, f.argtypes = ctypes.c_int, [ctypes.POINTER(SgArgs)] + [ctypes.c_int] * 6 + [ctypes.c_void_p]
                self.sg_fns[("dx", C, act)] = f
            if C == 8:  # experimental dX scatter variants (sparse_pool_sg_dxs.cuh)
                for act, an in ((0, "a_id"), (1, "a_relu")):
                    f = getattr(self.lib, f"sp_sgx_dx_c8_{an}")
                    f.restype, f.argtypes = ctypes.c_int, [ctypes.POINTER(SgArgs)] + [ctypes.c_int] * 7 + [ctypes.c_void_p]
                    self.sg_fns[("dxs", C, act)] = f
                    f = getattr(self.lib, f"sp_sgo_dx_c8_{an}")
                    f.restype, f.argtypes = ctypes.c_int, [ctypes.POINTER(SgArgs)] + [ctypes.c_int] * 6 + [ctypes.c_void_p]
                    self.sg_fns[("dxo", C, act)] = f
            if C == 8:
                self.lib.sp_sgx_layout.restype = ctypes.c_int
                self.lib.sp_sgx_layout.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
            f = getattr(self.lib, f"sp_sg_dw_c{C}")
            f.restype, f.argtypes = ctypes.c_int, [ctypes.POINTER(SgArgs)] + [ctypes.c_int] * 5 + [ctypes.c_void_p]
            self.sg_fns[("dw", C)] = f
        f = self.lib.sp_sgf_c8  # experimental fused backward (sparse_pool_sg_bwdf.cuh)
        f.restype, f.argtypes = ctypes.c_int, [ctypes.POINTER(SgArgs)] + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        self.sg_fns["fused"] = f

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
        if self.sg_use(rt, "fwd", B):
            return self.sg_forward(rt, x, w, b)
        rt.last_info["forward_kernel"] = "generic"
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


    # ------------------------------------------- input-supergroup (grouped) kernels
    def sg_reason(self, rt, phase: str, B: int) -> str:
        """Why the grouped kernel of a phase cannot run for this layer / batch ("" if it can)."""
        sg = getattr(rt, "sg", None)
        lv, L, plan = rt.level, rt.layout, rt.plan
        if sg is None:
            return "layer has no input-supergroup connectivity"
        if not (L.packed and L.shared_indices):
            return "needs packed Mode B parameters"
        if L.block not in (8, 64):
            return f"implemented for C = 8 and 64 (C = {L.block})"
        if lv.type != "mean" or lv.activation not in ("relu", "identity"):
            return f"implemented for Mean(ReLU / identity) only (pooling {lv.describe()})"
        if rt.units.S != 1:
            return "semantic level-1 shards are not supported"
        if plan.input_layout != "feature_major" or B % 8:
            return "needs the feature-major input operand (input_layout='feature_major' and B % 8 == 0)"
        if L.input_dim % 8 or L.input_dim > 16384:
            return "needs D % 8 == 0 and D <= 16384 (shared-memory feature map)"
        S = sg.S
        if phase == "fwd":
            T, nw, _ = self.sg_fwd_cfg(rt)
            smem = sg_smem("fwd", L.input_dim, S, T, nw)
        elif phase == "dx":
            epi = _epilogue(plan.cuda_dx_epilogue, lv, plan.grad_precision, "dx")
            if epi == "fast":
                return "the grouped dX implements the generic and mask epilogues only (requested fast)"
            if epi == "mask" and plan.grad_precision != "split":
                return "the grouped dX mask epilogue is compiled for split gradients only"
            T, nw, wg, _ = self.sg_dx_cfg(rt, B)
            if nw % wg:
                return f"cuda_grouped_dx_reduction={wg} must divide the {nw} warps"
            smem = sg_smem("dx", L.input_dim, S, T, nw, wg)
        else:
            epi = _epilogue(plan.cuda_dw_epilogue, lv, plan.grad_precision, "dw")
            if epi == "mask":
                return "the grouped dW implements the generic and fast epilogues only (requested mask)"
            smem = sg_smem("dw", L.input_dim, S, dw_cfg=self.sg_dw_cfg(rt))
        if smem > SG_MAX_SMEM:
            return f"{phase} configuration needs {smem / 1024:.1f} KiB of shared memory > 99 KiB"
        return ""

    def sg_use(self, rt, phase: str, B: int) -> bool:
        """Run the grouped kernel for this phase?  Explicit "grouped" requests that cannot run
        raise; "auto" falls back visibly (last_info["grouped_fallback"])."""
        if getattr(rt, "sg", None) is None:
            return False
        req = rt.plan.grouped_request(phase)
        if req == "generic":
            return False
        reason = self.sg_reason(rt, phase, B)
        fb = rt.last_info.setdefault("grouped_fallback", {})
        fb.pop(phase, None)
        if reason:
            if req == "grouped":
                raise ValueError(f"grouped {phase} requested but unavailable: {reason}")
            fb[phase] = reason
            return False
        if req == "auto":
            return self.sg_auto_phase(rt, phase) is not None
        return True

    @staticmethod
    def sg_auto_phase(rt, phase: str):
        """Measured auto configuration of a phase (None: generic is faster)."""
        C, S = rt.layout.block, rt.sg.S
        cfg = sg_auto(C, S)[("fwd", "dx", "dw").index(phase)]
        big = SG_AUTO_C64_LARGE_BUCKETS
        if C == 64 and phase in ("dx", "dw") and rt.sg.mean_bucket_units() >= big["min_bucket_units"]:
            c, s_max, precisions = big[phase]
            if S <= s_max and rt.plan.grad_precision in precisions:
                cfg = c
        return cfg

    def sg_fwd_cfg(self, rt):
        """(tokens, warps, chunk units) of the grouped forward (auto: SG_AUTO, else the largest
        token tile whose slab fits)."""
        p = rt.plan
        T0, nw0 = self.sg_auto_phase(rt, "fwd") or (128, 8)
        T = p.cuda_grouped_fwd_tokens or T0
        nw = p.cuda_grouped_fwd_warps or nw0
        if not p.cuda_grouped_fwd_tokens:
            while T > 16 and sg_smem("fwd", rt.layout.input_dim, rt.sg.S, T, nw) > SG_MAX_SMEM:
                T //= 2
        return T, nw, p.cuda_grouped_fwd_chunk_units or SG_CHUNK_UNITS["fwd"]

    def sg_dx_plan(self, rt, B: int):
        """(plan, reason) of the grouped dX: a DX_PLANS key or "tiles" (per-field settings).
        Host-only: the scale rule reads descriptors cached with the chunk table."""
        p = rt.plan
        if p.cuda_grouped_dx_plan != "auto":
            return p.cuda_grouped_dx_plan, "explicit cuda_grouped_dx_plan"
        explicit = [f for f in ("tokens", "warps", "reduction", "chunk_units") if getattr(p, f"cuda_grouped_dx_{f}")]
        if p.cuda_grouped_dx_scatter != "auto":
            explicit.append("scatter")
        if explicit:
            return "tiles", "explicit grouped dX fields take precedence (" + ", ".join(explicit) + ")"
        C, S, prec, act = rt.layout.block, rt.sg.S, p.grad_precision, rt.level.activation
        for r in SG_AUTO_DX_PLAN:
            if (r["C"], r["S"], act) != (C, S, r["act"]) or prec not in r["precisions"]:
                continue
            cc = torch.cuda.get_device_capability(rt.idx.device)
            if cc != r["cc"]:
                return "tiles", f"scale rule validated on sm_{r['cc'][0]}{r['cc'][1]} only (device sm_{cc[0]}{cc[1]})"
            if B < r["min_tokens"]:
                return "tiles", f"scale rule: M = {B} < {r['min_tokens']} (not measured)"
            ub = rt.sg.mean_bucket_units()
            if ub >= r["min_units_per_bucket"]:
                return r["plan"], f"scale rule: {ub:.2f} units / bucket >= {r['min_units_per_bucket']} (M = {B})"
            return r["below"], f"scale rule: {ub:.2f} units / bucket < {r['min_units_per_bucket']} (M = {B})"

        return "tiles", f"no scale rule for C = {C}, S = {S}, {prec}, {act}"

    def sg_dx_cfg(self, rt, B: int = 0):
        """(tokens, warps, warps per FP32 slab copy, chunk units) of the grouped dX.  A named
        plan fixes the first three; otherwise unset fields come from SG_AUTO; if that does not
        fit, copies are shared, then tiles halved."""
        p, S, D = rt.plan, rt.sg.S, rt.layout.input_dim
        name, _ = self.sg_dx_plan(rt, B)
        if name in DX_PLANS:
            d = DX_PLANS[name]
            return d["tokens"], d["warps"], d["reduction"], p.cuda_grouped_dx_chunk_units or SG_CHUNK_UNITS["dx"]
        T0, nw0, wg0 = self.sg_auto_phase(rt, "dx") or (64, 8, 8)
        T = p.cuda_grouped_dx_tokens or T0
        nw = p.cuda_grouped_dx_warps or nw0
        wg = p.cuda_grouped_dx_reduction
        if not wg:
            wg = min(wg0, nw)
            while wg < nw and sg_smem("dx", D, S, T, nw, wg) > SG_MAX_SMEM:
                wg *= 2
            if not p.cuda_grouped_dx_tokens:
                while T > 16 and sg_smem("dx", D, S, T, nw, wg) > SG_MAX_SMEM:
                    T //= 2
        return T, nw, wg, p.cuda_grouped_dx_chunk_units or SG_CHUNK_UNITS["dx"]

    def sg_dx_scatter(self, rt, T: int, nw: int, wg: int, plan: str = "tiles"):
        """(variant, fallback reason) of the grouped dX local accumulation.  An explicit
        unsupported request (or a named plan's scatter) raises; auto keeps the baseline
        outside the measured cases."""
        req = DX_PLANS[plan]["scatter"] if plan in DX_PLANS else rt.plan.cuda_grouped_dx_scatter
        name = req
        if req == "auto":
            name = SG_AUTO_DX_SCATTER.get((rt.layout.block, rt.sg.S, T, nw, wg, rt.plan.grad_precision,
                                           rt.level.activation), "baseline")
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
                raise ValueError(f"cuda_grouped_dx_scatter={req!r} unavailable: {why}")
            return "baseline", why
        return name, ""

    def sgx_layout(self, sc: int, tokens: int, S: int) -> torch.Tensor:
        """Device offsets sgx_off<sc, tokens / 16>(f, t) [S, tokens] of a dX slab layout (diagnostics)."""
        out = torch.empty((S, tokens), dtype=torch.int32, device="cuda")
        err = self.lib.sp_sgx_layout(sc, tokens // 16, S, _ptr(out),
                                     ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if err != 0:
            raise RuntimeError(f"sp_sgx_layout failed (code {err})")
        return out

    def sg_dw_cfg(self, rt) -> int:
        p = rt.plan
        if p.cuda_grouped_dw_cfg >= 0:
            return p.cuda_grouped_dw_cfg
        cfg = self.sg_auto_phase(rt, "dw")
        cfg = 2 if cfg is None else cfg
        if sg_smem("dw", rt.layout.input_dim, rt.sg.S, dw_cfg=cfg) > SG_MAX_SMEM:
            cfg = 0  # one 16-token tile per stage: the smallest ring (3 S x 16 slabs)
        return cfg

    def _sg_args(self, rt, xop, w, b, B):
        L, sg = rt.layout, rt.sg
        return SgArgs(_ptr(xop), xop.stride(0), _ptr(rt.idx), _ptr(w), _ptr(b), _ptr(rt.counts_i32),
                      _ptr(sg.feat), 0, 0, 0, 0, 0, 0, 0, 0,
                      B, L.groups, L.input_dim, sg.S, L.blocks_per_group, L.blocks_per_group // L.unit_blocks,
                      1, 0, 0)

    def sg_forward(self, rt, x, w, b):
        """Grouped Mean forward: returns {"pre": [B, G]}."""
        lv, L = rt.level, rt.layout
        B = x.shape[0]
        T, nw, cu = self.sg_fwd_cfg(rt)
        _mark("fwd.begin")
        xop = x.t().contiguous()  # [D, B] (same transpose as the generic path)
        _mark("fwd.layout")
        ysum = torch.zeros((L.groups, B), device=x.device, dtype=torch.float32)
        _mark("fwd.zero")
        items, ch = rt.sg.item_chunks(cu)
        a = self._sg_args(rt, xop, w, b, B)
        a.items, a.chunks, a.ysum = _ptr(items), _ptr(ch), _ptr(ysum)
        err = self.sg_fns[("fwd", L.block)](ctypes.byref(a), lv.act_id, T // 16, nw, ch.shape[0], math.ceil(B / T),
                                            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if err != 0:
            raise RuntimeError(f"sp_sg_fwd failed (code {err})")
        _mark("fwd.kernel")
        pre = (ysum / rt.units.counts[:, None]).t().contiguous()
        _mark("fwd.finalize")
        rt.last_info.update(phys_splits=1, workspace_bytes=ysum.numel() * 4, forward_kernel="grouped",
                            grouped_fwd=dict(tokens=T, warps=nw, chunk_units=cu, ctas=ch.shape[0] * math.ceil(B / T),
                                             chunks=ch.shape[0], items=items.shape[0],
                                             work_list_bytes=items.numel() * items.element_size()
                                             + ch.numel() * ch.element_size()))
        return {"pre": pre}

    def _sg_dx(self, rt, a, dxt, c1T, B, split, dx_epi, stream):
        L = rt.layout
        plan, plan_why = self.sg_dx_plan(rt, B)
        T, nw, wg, cu = self.sg_dx_cfg(rt, B)
        items, ch = rt.sg.item_chunks(cu)
        a.items, a.chunks, a.c1T, a.dXT, a.dx_group_warps = _ptr(items), _ptr(ch), _ptr(c1T), _ptr(dxt), wg
        epi = {"generic": 0, "mask": 2}[dx_epi]
        scatter, why = self.sg_dx_scatter(rt, T, nw, wg, plan)
        sc = SG_DX_SCATTERS[scatter]["sc"]
        act = rt.level.act_id
        gx, gy = ch.shape[0], math.ceil(B / T)
        if sc == 0:
            err = self.sg_fns[("dx", L.block, act)](ctypes.byref(a), T // 16, nw, split, epi, gx, gy, stream)
            kernel = f"sg_dx_kernel<{act}, {L.block}, {T // 16}, {nw}, {split}, {epi}>"
        elif sc == 4:
            err = self.sg_fns[("dxo", L.block, act)](ctypes.byref(a), T // 16, nw, split, epi, gx, gy, stream)
            ns = 1 if rt.sg.S <= 8 else (2 if rt.sg.S <= 16 else 4)
            kernel = f"sgx_dx_kernel<{act}, {L.block}, {T // 16}, {nw}, {split}, {epi}, 4, {ns}>"
        else:
            if sc == 1 and wg == 1:
                sc = 6  # SC_SWZ4P: the swizzle4 layout compiled for warp-private copies
            err = self.sg_fns[("dxs", L.block, act)](ctypes.byref(a), sc, T // 16, nw, split, epi, gx, gy, stream)
            kernel = f"sgx_dx_kernel<{act}, {L.block}, {T // 16}, {nw}, {split}, {epi}, {sc}, 1>"
        if err != 0:
            raise RuntimeError(f"sp_sg_dx ({scatter}) failed (code {err})")
        st = rt.sg.item_chunk_stats(cu)
        rt.last_info["grouped_dx"] = dict(plan=plan, plan_reason=plan_why, tokens=T, warps=nw, reduction_group_warps=wg,
                                          chunk_units=cu, units_per_chunk_mean=st["units_per_chunk_mean"],
                                          token_tiles=gy, tail_utilization=B / (gy * T),
                                          ctas=gx * gy, chunks=ch.shape[0],
                                          smem=sg_smem("dx", L.input_dim, rt.sg.S, T, nw, wg),
                                          scatter=scatter, scatter_layout=SG_DX_SCATTERS[scatter]["layout"],
                                          scatter_order=SG_DX_SCATTERS[scatter]["order"], scatter_fallback=why,
                                          kernel=kernel, epilogue=dx_epi,
                                          work_list_bytes=items.numel() * items.element_size()
                                          + ch.numel() * ch.element_size())
        return dxt.t()

    def _sg_dw(self, rt, a, c1T, split, dw_epi, stream):
        L = rt.layout
        cfg = self.sg_dw_cfg(rt)
        nw, wpw = SG_DW_CFGS[cfg][:2]
        units, ch = rt.sg.unit_chunks(nw * wpw)
        a.units, a.chunks, a.c1T = _ptr(units), _ptr(ch), _ptr(c1T)
        fn = self.sg_fns[("dw", L.block)]
        epi = {"generic": 0, "fast": 1}[dw_epi]
        a.flags = 0
        err = fn(ctypes.byref(a), rt.level.act_id, cfg, split, epi, ch.shape[0], stream)
        if err == 0 and epi:  # generic fallback launch: exits at once unless a coefficient is non-finite
            a.flags = BWD_FALLBACK
            err = fn(ctypes.byref(a), rt.level.act_id, cfg, split, 0, ch.shape[0], stream)
        if err != 0:
            raise RuntimeError(f"sp_sg_dw failed (code {err})")
        rt.last_info["grouped_dw"] = dict(cfg=cfg, ctas=ch.shape[0], smem=sg_smem("dw", L.input_dim, rt.sg.S, dw_cfg=cfg),
                                          work_list_bytes=units.numel() * units.element_size()
                                          + ch.numel() * ch.element_size())

    def sg_fused_reason(self, rt, B: int, dx_epi: str, dw_epi: str) -> str:
        """Why the fused shared-projection backward cannot run ("" if it can)."""
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
        smem = sgf_smem(L.input_dim, rt.sg.S, p.cuda_grouped_fused_cfg)
        if smem > SG_MAX_SMEM:
            return f"fused cfg {p.cuda_grouped_fused_cfg} needs {smem / 1024:.1f} KiB of shared memory > 99 KiB"
        return ""

    def _sg_fused(self, rt, a, dxt, c1T, B, stream, share):
        """Fused backward + the guarded generic fallbacks (run only for non-finite coefficients,
        when the fused kernel exits): exactly one path contributes."""
        L = rt.layout
        cfg = rt.plan.cuda_grouped_fused_cfg
        nw, wpw, ts, _, minb = SGF_CFGS[cfg]
        units, ch = rt.sg.unit_chunks(nw * wpw)
        a.units, a.chunks, a.c1T, a.dXT, a.flags = _ptr(units), _ptr(ch), _ptr(c1T), _ptr(dxt), 0
        err = self.sg_fns["fused"](ctypes.byref(a), cfg, int(share), ch.shape[0], stream)
        if err != 0:
            raise RuntimeError(f"sp_sgf_c8 failed (code {err})")
        _mark("bwd.fused")
        # fallback dX: the original grouped kernel, generic epilogue (private copies where they fit)
        T, nwx, wg = 64, 8, (1 if rt.sg.S <= 32 else 8)
        items, chx = rt.sg.item_chunks(SG_CHUNK_UNITS["dx"])
        a.items, a.chunks, a.dx_group_warps, a.flags = _ptr(items), _ptr(chx), wg, BWD_FALLBACK
        err = self.sg_fns[("dx", L.block, rt.level.act_id)](ctypes.byref(a), T // 16, nwx, 1, 0, chx.shape[0],
                                                             math.ceil(B / T), stream)
        if err != 0:
            raise RuntimeError(f"sp_sg_dx fallback failed (code {err})")
        # fallback dW / db: the grouped dW generic launch (BWD_FALLBACK), cfg with the same ownership
        units_w, chw = rt.sg.unit_chunks(nw * wpw) if (nw, wpw) in ((8, 2), (8, 4)) else rt.sg.unit_chunks(16)
        dwc = {(8, 2): 2, (8, 4): 3}.get((nw, wpw), 2)
        a.units, a.chunks, a.flags = _ptr(units_w), _ptr(chw), BWD_FALLBACK
        err = self.sg_fns[("dw", L.block)](ctypes.byref(a), rt.level.act_id, dwc, 1, 0, chw.shape[0], stream)
        if err != 0:
            raise RuntimeError(f"sp_sg_dw fallback failed (code {err})")
        a.flags = 0
        S = rt.sg.S
        rt.last_info["grouped_fused"] = dict(
            cfg=cfg, share=bool(share), warps=nw, units_per_warp=wpw, window_tokens=16 * ts, min_ctas_per_sm=minb,
            ctas=ch.shape[0], windows=math.ceil(B / (16 * ts)), smem=sgf_smem(L.input_dim, S, cfg),
            kernel=(f"sgf_kernel<{nw}, {wpw}, {ts}, {SGF_CFGS[cfg][3]}, {minb}, {str(bool(share)).lower()}>" if cfg < 6
                    else f"sgf2_kernel<{nw}, {wpw}, {SGF_CFGS[cfg][3]}, {minb}, {str(bool(share)).lower()}>"),
            dx_flush_atomics_upper=ch.shape[0] * S * B,
            fallback="sg_dx_kernel generic + sg_dw generic, flags BWD_FALLBACK (run only if coefficients are non-finite)",
            work_list_bytes=units.numel() * units.element_size() + ch.numel() * ch.element_size())
        rt.last_info.pop("grouped_dx", None)
        rt.last_info.pop("grouped_dw", None)
        return dxt.t()

    # ---------------------------------------------------------------- backward
    def supports_backward(self, rt) -> bool:
        lv, L = rt.level, rt.layout
        return (L.packed and lv.type in _CUDA_BWD_POOLS and lv.activation in ("identity", "relu")
                and L.input_dim % 8 == 0 and L.input_dim * 16 * 6 <= 99 * 1024)

    @staticmethod
    def dw_staging_reason(L, device, sxb) -> str:
        """Why indexed k_dw staging cannot be used ("" if it can).  It is implemented for
        packed C = 64 Mode B with a feature-major input operand and validated on sm_86."""
        if not (L.packed and L.shared_indices):
            return "needs packed Mode B parameters"
        if L.block != 64:
            return f"implemented for C = 64 only (C = {L.block})"
        if sxb != 1:
            return "needs the feature-major input operand (input_layout='feature_major' and B % 8 == 0)"
        cc = torch.cuda.get_device_capability(device)
        if cc != (8, 6):
            return f"validated on sm_86 only (device is sm_{cc[0]}{cc[1]})"
        return ""

    @classmethod
    def dw_staging(cls, plan: PhysicalPlan, L, device, sxb, info=None) -> str:
        if info is not None:
            info.pop("dw_staging_fallback", None)
        if plan.cuda_dw_staging == "full":
            return "full"
        reason = cls.dw_staging_reason(L, device, sxb)
        if not reason:
            return "indexed"
        if plan.cuda_dw_staging == "indexed":  # explicit request: never fall back silently
            if info is not None:
                info["dw_staging_fallback"] = reason
            if reason not in _warned:
                _warned.add(reason)
                warnings.warn(f"cuda_dw_staging='indexed' unavailable ({reason}); using the full-slab k_dw",
                              stacklevel=3)
        return "full"

    def backward(self, rt, x, w, b, g_pre, stats, dw_buf, db_buf):
        """Recompute backward: accumulates into dw_buf/db_buf (packed FP32),
        returns dX [B, D] FP32."""
        lv, L, plan = rt.level, rt.layout, rt.plan
        B, D = x.shape
        U = rt.units.U
        _mark("bwd.begin")
        c1, c2, c3 = ops.dense_coefficients(lv, g_pre, stats.get("pre"), rt.units.counts, stats.get("L2"))
        c1, c2, c3 = c1.contiguous(), c2.contiguous(), c3.contiguous()
        sg_dx, sg_dw = self.sg_use(rt, "dx", B), self.sg_use(rt, "dw", B)
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
        dxt = torch.zeros((D, B), device=x.device, dtype=torch.float32) if sg_dx else None  # grouped: [D, B]
        _mark("bwd.dx_init")
        n_tiles = math.ceil(B / 16)
        gy = max(1, min(U, math.ceil(4096 / n_tiles)))  # ~4096 CTAs; one dX flush per CTA
        upc = math.ceil(U / gy)
        gy = math.ceil(U / upc)
        staging = self.dw_staging(plan, L, x.device, sxb, rt.last_info)
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
            _ptr(c1), _ptr(c2), _ptr(c3), _ptr(dx), D if dx is None else dx.stride(1), 1 if dx is None else dx.stride(0),
            _ptr(dw_buf), _ptr(db_buf),
            B, U, rt.units.S, L.blocks_per_group, rt.units.BPS, D,
            ops.kernel_scale(lv), float(lv.p), ops._p_int(lv), upc, wu_per_cta, 0,
        )
        # pooling-specialised dZ epilogues (FastDz in sparse_pool_bwd.cuh): "fast" for Mean and
        # LME (ReLU / identity) and integer-p power mean, "mask" for Mean only; the generic dW
        # variants exist for the indexed kernel only (the grouped dW has generic and fast).
        dx_epi = _epilogue(plan.cuda_dx_epilogue, lv, plan.grad_precision, "dx")
        dw_epi = (_epilogue(plan.cuda_dw_epilogue, lv, plan.grad_precision, "dw")
                  if (staging == "indexed" or sg_dw) else "generic")
        nonfinite = None
        if dx_epi != "generic" or dw_epi != "generic":  # device flag (no host sync): inf/NaN coefficients make the kernels take the generic path
            # scalar inf-norm reductions (NaN propagates and fails the compare): no [B, U] temporaries
            ok = torch.linalg.vector_norm(c1, float("inf")) < 2.0**64
            if lv.type != "mean":  # Mean never reads c2
                ok = ok & (torch.linalg.vector_norm(c2, float("inf")) < float("inf"))
            nonfinite = (~ok).to(torch.uint8)
            a.nonfinite = _ptr(nonfinite)
        split = int(plan.grad_precision == "split")
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        pool = _CUDA_POOLS[lv.type]
        dx_warps = plan.cuda_dx_warps or dx_warps_default(L.block, rt.units.BPS, L.unit_blocks)
        def launch(kind, cfg_, gx, gy_, flags, what):
            a.flags = flags
            err = self.lib.sp_bwd(ctypes.byref(a), pool, lv.act_id, L.block, split, kind, cfg_, gx, gy_, stream)
            if err != 0:
                raise RuntimeError(f"sp_bwd({what}) failed (code {err})")

        if sg_dx or sg_dw:
            sa = self._sg_args(rt, xop, w, b, B)
            sa.dW, sa.dB, sa.nonfinite = _ptr(dw_buf), _ptr(db_buf), _ptr(nonfinite)
        org = plan.cuda_grouped_backward
        fused = False
        if org in ("fused", "fused_recompute") and getattr(rt, "sg", None) is not None:
            why = self.sg_fused_reason(rt, B, dx_epi, dw_epi)
            if why:
                raise ValueError(f"cuda_grouped_backward={org!r} unavailable: {why}")
            if not (sg_dx and sg_dw):
                raise ValueError(f"cuda_grouped_backward={org!r} needs grouped dX and dW (requested generic)")
            fused = True
        rt.last_info["backward_organization"] = ("fused" if org == "fused" else "fused_recompute") if fused else "separate"
        if not fused:
            rt.last_info.pop("grouped_fused", None)
        if fused:
            dx = self._sg_fused(rt, sa, dxt, c1T, B, stream, share=(org == "fused"))
            _mark("bwd.dx")
            _mark("bwd.dw")
        elif sg_dx:
            dx = self._sg_dx(rt, sa, dxt, c1T, B, split, dx_epi, stream)
        else:
            dx_flags = _EPI_BITS["dx"].get(dx_epi, 0)
            launch(0, dx_warps, n_tiles, gy, dx_flags, "dx")  # k_dx falls back to the generic body in-kernel
        if not fused:
            _mark("bwd.dx")
        if fused:
            pass
        elif sg_dw:
            self._sg_dw(rt, sa, c1T, split, dw_epi, stream)
        else:
            kind = 1 if staging == "full" else {2: 2, 3: 3}[DW_INDEXED_STAGES]
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
