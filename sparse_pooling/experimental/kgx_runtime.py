"""ctypes bridge of the EXPERIMENTAL K = 64 kgx kernels (bundle "kgx", kernels/cuda/experimental/kgx).

Built / loaded through the production registry (backends/cuda/runtime.load_bundle: per-target JIT, build
identity, ABI version, bundle / kind checks); the KxArgs struct is checked against ``sp_kx_struct_layout``
before any entry point is bound.
"""
from __future__ import annotations

import ctypes
import threading
from typing import Dict

from ..backends.capabilities import query
from ..backends.cuda import build as B_
from ..backends.cuda import runtime as R_


class KxArgs(ctypes.Structure):
    _fields_ = [
        ("X", ctypes.c_void_p), ("sxf", ctypes.c_longlong), ("idx", ctypes.c_void_p), ("W", ctypes.c_void_p),
        ("bias", ctypes.c_void_p), ("counts", ctypes.c_void_p), ("feat", ctypes.c_void_p),
        ("items", ctypes.c_void_p), ("chunks", ctypes.c_void_p), ("tasks", ctypes.c_void_p),
        ("tmeta", ctypes.c_void_p), ("rowptr", ctypes.c_void_p), ("kix", ctypes.c_void_p), ("wins", ctypes.c_void_p),
        ("ysum", ctypes.c_void_p), ("c1T", ctypes.c_void_p), ("dXT", ctypes.c_void_p), ("dW", ctypes.c_void_p),
        ("dB", ctypes.c_void_p),
        ("B", ctypes.c_int), ("G", ctypes.c_int), ("D", ctypes.c_int), ("S", ctypes.c_int), ("NB", ctypes.c_int),
        ("C", ctypes.c_int), ("dx_group_warps", ctypes.c_int), ("flags", ctypes.c_int),
    ]


class KwArgs(ctypes.Structure):
    """kgx_wide.cuh KwArgs: the KxArgs of the tile kernels + backbone pointers and the (tile width, backbone)
    selection (checked by the native entry points)."""
    _fields_ = [("a", KxArgs), ("bbk", ctypes.c_void_p), ("bbw", ctypes.c_void_p), ("dBB", ctypes.c_void_p),
                ("tb", ctypes.c_int), ("bb", ctypes.c_int)]


# kernel variant ids (kgx_dispatch.cuh)
VARIANTS = {"direct": 0, "idf_rmw": 1, "idf": 2, "tile": 3, "csr": 4}
KIND = {"fwd": 0, "dx": 1, "dw": 2}
CFG_FIELDS = ("fwd_mt_direct", "fwd_mt_index_free", "fwd_warps", "dx_mt", "dx_warps", "dw_warps", "dw_tiles_per_stage",
              "dw_stages", "csr_tokens_per_lane", "csr_warps", "csr_dw_tokens_per_stage", "csr_dw_stages",
              "csr_dx_warps")


class KgxBackend:
    def __init__(self, lib: R_.NativeLibrary, target: str, target_reason: str):
        self.native, self.lib, self.target, self.target_reason = lib, lib.lib, target, target_reason
        f = lib.fn("sp_kx_struct_layout", [ctypes.POINTER(ctypes.c_longlong), ctypes.c_int])
        buf = (ctypes.c_longlong * 64)()
        n = f(buf, 64)
        want = [ctypes.sizeof(KxArgs)] + [getattr(KxArgs, nm).offset for nm, _ in KxArgs._fields_]
        if n != len(want) or list(buf[:n]) != want:
            raise RuntimeError(f"{lib.path}: KxArgs layout mismatch (native {list(buf[:max(n, 0)])}, ctypes {want})")
        o = (ctypes.c_int * 16)()
        if lib.fn("sp_kx_config", [ctypes.POINTER(ctypes.c_int)])(o) != len(CFG_FIELDS):
            raise RuntimeError("sp_kx_config failed")
        self.cfg = dict(zip(CFG_FIELDS, list(o)[:len(CFG_FIELDS)]))
        self._fwd = lib.fn("sp_kx_fwd", [ctypes.POINTER(KxArgs)] + [ctypes.c_int] * 3 + [ctypes.c_void_p])
        self._dx = lib.fn("sp_kx_dx", [ctypes.POINTER(KxArgs)] + [ctypes.c_int] * 4 + [ctypes.c_void_p])
        self._dw = lib.fn("sp_kx_dw", [ctypes.POINTER(KxArgs)] + [ctypes.c_int] * 3 + [ctypes.c_void_p])
        self._kc = lib.fn("sp_kc", [ctypes.POINTER(KxArgs)] + [ctypes.c_int] * 3 + [ctypes.c_void_p])
        self._smem = lib.fn("sp_kx_smem", [ctypes.POINTER(KxArgs), ctypes.c_int, ctypes.c_int,
                                            ctypes.POINTER(ctypes.c_longlong)])
        # width-preserving kernels (kgx_wide.cuh): their own argument struct, checked the same way
        f = lib.fn("sp_kw_struct_layout", [ctypes.POINTER(ctypes.c_longlong), ctypes.c_int])
        n = f(buf, 64)
        want = [ctypes.sizeof(KwArgs)] + [getattr(KwArgs, nm).offset for nm, _ in KwArgs._fields_]
        if n != len(want) or list(buf[:n]) != want:
            raise RuntimeError(f"{lib.path}: KwArgs layout mismatch (native {list(buf[:max(n, 0)])}, ctypes {want})")
        self._wfwd = lib.fn("sp_kw_fwd", [ctypes.POINTER(KwArgs)] + [ctypes.c_int] * 2 + [ctypes.c_void_p])
        self._wdx = lib.fn("sp_kw_dx", [ctypes.POINTER(KwArgs)] + [ctypes.c_int] * 3 + [ctypes.c_void_p])
        self._wdw = lib.fn("sp_kw_dw", [ctypes.POINTER(KwArgs)] + [ctypes.c_int] * 2 + [ctypes.c_void_p])

    def build_info(self) -> Dict:
        i = self.native.ident
        return dict(bundle="kgx", status="experimental", target=self.target, target_reason=self.target_reason,
                    kind=i["kind"], library=str(self.native.path), library_hash=i["library_hash"], flags=i["flags"],
                    nvcc=i["nvcc"], nvcc_version=i["nvcc_version"], host_compiler=i["host_compiler"],
                    unit_hashes=i["unit_hashes"])

    def smem(self, a: KxArgs, kind: str, variant: str) -> int:
        out = ctypes.c_longlong()
        self._smem(ctypes.byref(a), KIND[kind], VARIANTS[variant], ctypes.byref(out))
        return int(out.value)

    def launch(self, kind: str, variant: str, a: KxArgs, split: int, n_chunks: int, n_tiles: int, device):
        s = R_._stream(device)
        v = VARIANTS[variant]
        if variant == "csr":
            err = self._kc(ctypes.byref(a), KIND[kind], n_chunks, n_tiles, s)
        elif kind == "fwd":
            err = self._fwd(ctypes.byref(a), v, n_chunks, n_tiles, s)
        elif kind == "dx":
            err = self._dx(ctypes.byref(a), v, split, n_chunks, n_tiles, s)
        else:
            err = self._dw(ctypes.byref(a), v, split, n_chunks, s)
        R_._check(err, f"kgx {variant} {kind}")


    def launch_wide(self, kind: str, w: KwArgs, split: int, n_chunks: int, n_tiles: int, device):
        s = R_._stream(device)
        if kind == "fwd":
            err = self._wfwd(ctypes.byref(w), n_chunks, n_tiles, s)
        elif kind == "dx":
            err = self._wdx(ctypes.byref(w), split, n_chunks, n_tiles, s)
        else:
            err = self._wdw(ctypes.byref(w), split, n_chunks, s)
        R_._check(err, f"kgx wide (tb {w.tb}, backbone {w.bb}) {kind}")


_lock = threading.Lock()
_backends: Dict[tuple, KgxBackend] = {}


def kgx_backend(device) -> KgxBackend:
    caps = query(device)
    target, why = B_.resolve_target(caps)
    lib = R_.load_bundle("kgx", target)
    with _lock:
        key = (target, lib.ident["kind"], lib.ident["library_hash"])
        if key not in _backends:
            _backends[key] = KgxBackend(lib, target, why)
        return _backends[key]
