"""ctypes bridge of the EXPERIMENTAL generalized-fan-in kernels (bundle "kgen").

The bundle is built / loaded through the production registry (backends/cuda/runtime.load_bundle: per
target JIT, build identity, ABI version, bundle / kind and production struct-layout checks), and its
own argument struct is checked against ``sp_kg_struct_layout`` before any entry point is bound.
"""
from __future__ import annotations

import ctypes
import threading
from typing import Dict

import torch

from ..backends.capabilities import query
from ..backends.cuda import build as B_
from ..backends.cuda import runtime as R_


class KgArgs(ctypes.Structure):
    _fields_ = [
        ("X", ctypes.c_void_p), ("sxf", ctypes.c_longlong), ("idx", ctypes.c_void_p), ("W", ctypes.c_void_p),
        ("bias", ctypes.c_void_p), ("counts", ctypes.c_void_p), ("feat", ctypes.c_void_p),
        ("items", ctypes.c_void_p), ("chunks", ctypes.c_void_p), ("tasks", ctypes.c_void_p),
        ("ysum", ctypes.c_void_p), ("c1T", ctypes.c_void_p), ("dXT", ctypes.c_void_p), ("dW", ctypes.c_void_p),
        ("dB", ctypes.c_void_p),
        ("B", ctypes.c_int), ("G", ctypes.c_int), ("D", ctypes.c_int), ("S", ctypes.c_int), ("NB", ctypes.c_int),
        ("C", ctypes.c_int), ("dx_group_warps", ctypes.c_int), ("flags", ctypes.c_int),
    ]


KIND = {"fwd": 0, "dx": 1, "dw": 2}
CFG_FIELDS = ("fwd_mt", "fwd_warps", "dx_mt", "dx_warps", "dw_tasks_per_warp", "dw_warps", "dw_tiles_per_stage",
              "dw_stages")


class KgenBackend:
    def __init__(self, lib: R_.NativeLibrary, target: str, target_reason: str):
        self.native, self.lib, self.target, self.target_reason = lib, lib.lib, target, target_reason
        f = lib.fn("sp_kg_struct_layout", [ctypes.POINTER(ctypes.c_longlong), ctypes.c_int])
        buf = (ctypes.c_longlong * 64)()
        n = f(buf, 64)
        want = [ctypes.sizeof(KgArgs)] + [getattr(KgArgs, nm).offset for nm, _ in KgArgs._fields_]
        if n != len(want) or list(buf[:n]) != want:
            raise RuntimeError(f"{lib.path}: KgArgs layout mismatch (native {list(buf[:max(n, 0)])}, ctypes {want})")
        cf = lib.fn("sp_kg_config", [ctypes.c_int, ctypes.POINTER(ctypes.c_int)])
        self.cfgs: Dict[int, tuple] = {}
        self.fns = {}
        for K in (8, 16, 32, 64):
            o = (ctypes.c_int * 8)()
            if cf(K, o) != 8:
                raise RuntimeError(f"sp_kg_config({K}) failed")
            self.cfgs[K] = tuple(o)
            self.fns[K] = lib.fn(f"sp_kg_k{K}", [ctypes.POINTER(KgArgs)] + [ctypes.c_int] * 5 + [ctypes.c_void_p])
        # K16 dX tiling variants (kgen_k16_dxv.cu; experimental/kgen_dx_variants.py)
        self.dxv_fn = lib.fn("sp_kg_k16_dxv", [ctypes.POINTER(KgArgs)] + [ctypes.c_int] * 7 + [ctypes.c_void_p])
        lf = lib.fn("sp_kg_k16_dxv_list", [ctypes.POINTER(ctypes.c_int), ctypes.c_int])
        buf = (ctypes.c_int * 96)()
        nv = lf(buf, 96)
        if nv < 0:
            raise RuntimeError("sp_kg_k16_dxv_list failed")
        self.dxv_compiled = [tuple(buf[3 * i:3 * i + 3]) for i in range(nv)]
        self._dxv_attrs = lib.fn("sp_kg_k16_dxv_attrs", [ctypes.c_int] * 5 + [ctypes.c_longlong,
                                                                              ctypes.POINTER(ctypes.c_longlong), ctypes.c_int])

    def build_info(self) -> Dict:
        i = self.native.ident
        return dict(bundle="kgen", status="experimental", target=self.target, target_reason=self.target_reason,
                    kind=i["kind"], library=str(self.native.path), library_hash=i["library_hash"], flags=i["flags"],
                    nvcc=i["nvcc"], nvcc_version=i["nvcc_version"], host_compiler=i["host_compiler"],
                    unit_hashes=i["unit_hashes"])

    def cfg_dict(self, K: int) -> Dict:
        return dict(zip(CFG_FIELDS, self.cfgs[K]))

    def launch_dxv(self, a: KgArgs, act: int, split: int, mt: int, nw: int, impl: int, n_chunks: int, n_tiles: int,
                   device):
        err = self.dxv_fn(ctypes.byref(a), act, split, mt, nw, impl, n_chunks, n_tiles, R_._stream(device))
        R_._check(err, f"sp_kg_k16_dxv(mt {mt}, nw {nw}, impl {impl})")

    def dxv_attributes(self, mt: int, nw: int, impl: int, act: int, split: int, smem: int) -> Dict:
        """Compiled attributes of a K16 dX variant (cudaFuncGetAttributes) and the theoretical active CTAs per SM at
        ``smem`` dynamic shared bytes (cudaOccupancyMaxActiveBlocksPerMultiprocessor) on the CURRENT device."""
        o = (ctypes.c_longlong * 8)()
        n = self._dxv_attrs(mt, nw, impl, act, split, smem, o, 8)
        if n != 8:
            return dict(error=n)
        return dict(registers_per_thread=o[0], local_bytes_per_thread=o[1], static_smem=o[2], max_threads_per_block=o[3],
                    ptx_version=o[4], binary_version=o[5], active_ctas_per_sm=o[6], max_dynamic_smem_configured=o[7])

    def launch(self, K: int, kind: str, a: KgArgs, act: int, split: int, n_chunks: int, n_tiles: int, device):
        err = self.fns[K](ctypes.byref(a), KIND[kind], act, split, n_chunks, n_tiles, R_._stream(device))
        R_._check(err, f"sp_kg_k{K}({kind})")


_lock = threading.Lock()
_backends: Dict[tuple, KgenBackend] = {}


def kgen_backend(device) -> KgenBackend:
    caps = query(device)
    target, why = B_.resolve_target(caps)
    lib = R_.load_bundle("kgen", target)
    with _lock:
        key = (target, lib.ident["kind"], lib.ident["library_hash"])
        if key not in _backends:
            _backends[key] = KgenBackend(lib, target, why)
        return _backends[key]
