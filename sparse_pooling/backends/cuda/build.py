"""Build and cache the native CUDA libraries (one shared library per bundle and target).

Distribution model: per-target JIT.  On the first CUDA operation for a device, each needed bundle is
compiled with the installed nvcc for that device's target (``-arch=sm_XY``: SASS for sm_XY plus
compute_XY PTX) and cached; nothing is compiled at import time.  The embedded PTX is not a
portability promise: the runtime only loads a library whose target is binary-compatible with the
device (same major, device minor >= target minor), and the kernels need sm_80+ features anyway.

Bundles (explicit manifests; no directory globbing):
  production    every unit the dispatch can select automatically, plus ABI identity;
  experimental  correct variants reachable only through explicit PhysicalPlan options (built on
                first explicit request);
  diagnostics   tests / tools kernels (layout dumps, projection diagnostics); never used by a layer;
  kgen          EXPERIMENTAL generalized-fan-in kernels of sparse_pooling.experimental (explicit use only);
  kgx           EXPERIMENTAL K = 64 large-C / index-free / tile-sparse / CSR kernels of
                sparse_pooling.experimental, plus the width-preserving 16 x 8 tile / backbone + tile
                kernels (explicit use only).
Build kinds: "production" (no extra flags), "development" (SPARSE_POOL_NVCC_EXTRA without
ablations), "ablation" (SP_ABLATE_* in SPARSE_POOL_NVCC_EXTRA: deliberately WRONG results for
attribution).  The kind is compiled into the library (sp_build_kind), named in its path and recorded
in last_info / benchmark JSON; validation harnesses refuse non-production builds.

Identity: the cache key of an object hashes the build schema, bundle / kind ids, target, effective
flags, nvcc ``--version`` text, the host compiler identity, and the relative path and content of the
unit and of every header in its transitive #include closure; the library key hashes its objects'
keys.  Different targets, flags or toolchains never share an artifact; unchanged inputs reuse it.

Cache: ``SPARSE_POOL_CACHE_DIR``, else ``$XDG_CACHE_HOME/granulelayer/cuda``, else
``~/.cache/granulelayer/cuda`` (the source tree may be read-only).  Concurrency: compiles are bounded
by ``SPARSE_POOL_BUILD_JOBS`` (default min(CPUs, 8)); every artifact is written to a unique temporary
file and published with an atomic rename, and a per-library file lock serialises duplicate builds
across processes, so a reader never sees a partial object or library.  ``clear_cache()`` /
``python -m sparse_pooling.backends.cuda.build --clean`` removes the cache.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..capabilities import NATIVE_FAMILY_FEATURES, DeviceCapabilities, ToolchainError, ToolchainInfo, toolchain

SRC_DIR = Path(__file__).resolve().parents[2] / "kernels" / "cuda"
BUILD_SCHEMA = "granule.cuda_build/3"
MIN_ARCH = (8, 0)  # kernel family requirement (BF16 mma.sync, cp.async)

BUNDLES: Dict[str, Tuple[str, ...]] = {
    "production": (
        "abi.cu", "sparse_pool.cu",
        "fwd_max.cu", "fwd_mean.cu", "fwd_lme.cu", "fwd_pow.cu", "fwd_sw.cu", "fwd_topk.cu",
        "bwd_mean.cu", "bwd_lme.cu", "bwd_pow.cu", "bwd_sw.cu",
        "sg_fwd_c8.cu", "sg_fwd_c64.cu", "sg_dx_c8_a_id.cu", "sg_dx_c8_a_relu.cu", "sg_dx_c64_a_id.cu",
        "sg_dx_c64_a_relu.cu", "sg_dxs_c8_a_id.cu", "sg_dxs_c8_a_relu.cu", "sg_dw_c8.cu", "sg_dw_c64.cu", "sg_det.cu",
    ),
    "experimental": (
        "abi.cu", "experimental/sg_dxs_exp_c8_a_id.cu", "experimental/sg_dxs_exp_c8_a_relu.cu",
        "experimental/sg_dxo_c8_a_id.cu", "experimental/sg_dxo_c8_a_relu.cu", "experimental/sg_bwdf_c8.cu",
        "experimental/sg_dw_c8_exp.cu",
    ),
    "diagnostics": ("abi.cu", "diagnostics/sgx_layout.cu", "diagnostics/diag_proj_c8.cu"),
    # EXPERIMENTAL generalized fan-in (K = 8 / 16 / 32 / 64) input-supergroup kernels of
    # sparse_pooling.experimental (random K/C/S/R sweep); never selected by SparsePoolingLayer
    "kgen": ("abi.cu", "experimental/kgen/kgen_abi.cu", "experimental/kgen/kgen_k8.cu", "experimental/kgen/kgen_k16.cu",
             "experimental/kgen/kgen_k32.cu", "experimental/kgen/kgen_k64.cu"),
    # EXPERIMENTAL K = 64 kernels of the large-C / index-free / L1-sparsity study (direct large-C, S = K
    # index-free, tile-sparse, CSR fixed mask); never selected by SparsePoolingLayer
    "kgx": ("abi.cu", "experimental/kgx/kgx_abi.cu", "experimental/kgx/kgx_fwd.cu", "experimental/kgx/kgx_dx.cu",
            "experimental/kgx/kgx_dw.cu", "experimental/kgx/kgx_csr.cu", "experimental/kgx/kgx_wfwd.cu",
            "experimental/kgx/kgx_wdx.cu", "experimental/kgx/kgx_wdw.cu"),
}
BUNDLE_ID = {"production": 0, "experimental": 1, "diagnostics": 2, "kgen": 3, "kgx": 4}
KIND_ID = {"production": 0, "development": 1, "ablation": 2}
BASE_FLAGS = ("-O3", "--use_fast_math", "-lineinfo", "-std=c++17", "-Xcompiler", "-fPIC")
ARCH_ENV, EXTRA_ENV, CACHE_ENV, JOBS_ENV = ("SPARSE_POOL_ARCH", "SPARSE_POOL_NVCC_EXTRA", "SPARSE_POOL_CACHE_DIR",
                                            "SPARSE_POOL_BUILD_JOBS")
# Plain SASS targets, matched as a complete token.  CUDA writes compute capability X.Y as "XY": the
# minor is the LAST digit and the major everything before it (sm_86 = 8.6, sm_100 = 10.0, sm_121 = 12.1).
_ARCH_RE = re.compile(r"sm_([1-9]\d*)(\d)")
# Architecture-specific ("a") / family ("f") targets (sm_90a, sm_100a, sm_100f, ...): real compiler
# targets with their own compatibility rules.  The kernels never use them, so they are refused
# explicitly instead of being stripped to the plain target.
_SUFFIXED_ARCH_RE = re.compile(r"sm_([1-9]\d*)(\d)[af]")


class BuildError(RuntimeError):
    pass


# --------------------------------------------------------------------------- targets
def parse_arch(arch: str) -> Tuple[int, int]:
    """(major, minor) of a plain ``sm_<major><minor>`` target: sm_86 -> (8, 6), sm_100 -> (10, 0),
    sm_120 -> (12, 0).  The whole string must match; anything else raises BuildError."""
    m = _ARCH_RE.fullmatch(arch) if isinstance(arch, str) else None
    if not m:
        if isinstance(arch, str) and _SUFFIXED_ARCH_RE.fullmatch(arch):
            raise BuildError(f"{arch!r} is an architecture-specific ('a') / family ('f') target; these kernels "
                             f"are built only for plain sm_<major><minor> targets (e.g. {arch[:-1]})")
        raise BuildError(f"invalid CUDA target {arch!r} (expected sm_<major><minor>, e.g. sm_86, sm_90, sm_120)")
    return int(m.group(1)), int(m.group(2))


def arch_name(cc: Tuple[int, int]) -> str:
    """Plain target of a compute capability: (8, 6) -> sm_86, (12, 0) -> sm_120."""
    return f"sm_{cc[0]}{cc[1]}"


def binary_compatible(target: Tuple[int, int], cc: Tuple[int, int]) -> bool:
    """A cubin for sm_XY runs on compute capability X.Z with Z >= Y (CUDA binary compatibility)."""
    return target[0] == cc[0] and cc[1] >= target[1]


def resolve_target(caps: DeviceCapabilities, tc: Optional[ToolchainInfo] = None,
                   override: Optional[str] = None) -> Tuple[str, str]:
    """(target, reason) for a device.  ``override`` / SPARSE_POOL_ARCH force a target, which must
    still be binary-compatible with the device and supported by nvcc.  Default: the device's own
    sm_XY; if the toolkit cannot emit it, the newest supported same-major target <= the device
    (binary compatible); otherwise an actionable error — an unknown device is never mapped to an
    arbitrary target."""
    if caps.backend != "cuda":
        raise BuildError(f"no CUDA target for a {caps.backend} device")
    cc = tuple(caps.cc)
    if cc < MIN_ARCH:
        raise BuildError(f"{caps.name} is {arch_name(cc)}: the native kernels need sm_80+ (BF16 mma.sync, cp.async); "
                         "use backend='triton'")
    miss = caps.missing(NATIVE_FAMILY_FEATURES)
    if miss:  # a capability record without the kernel family's instructions never gets a target
        raise BuildError(f"{caps.name} ({arch_name(cc)}) lacks {', '.join(miss)} required by the native kernels; "
                         "use backend='triton'")
    tc = tc or toolchain()
    override = override or os.environ.get(ARCH_ENV) or None
    if override:
        t = parse_arch(override)
        if not binary_compatible(t, cc):
            raise BuildError(f"{ARCH_ENV}={override} cannot run on {caps.name} ({arch_name(cc)}): a cubin runs only on "
                             "the same major architecture with an equal or newer minor version")
        if t < MIN_ARCH:
            raise BuildError(f"{override}: the native kernels need sm_80+")
        if not tc.supports(override):
            raise BuildError(f"nvcc {tc.version} ({tc.nvcc}) cannot emit {override} (supported: {' '.join(tc.real_archs)})")
        return override, f"explicit {ARCH_ENV}={override}"
    own = arch_name(cc)
    if tc.supports(own):
        return own, f"device {caps.name} ({own})"
    plain = [parse_arch(a) for a in tc.real_archs if _ARCH_RE.fullmatch(a)]  # 'a' / 'f' targets never qualify
    compat = sorted((t for t in plain if binary_compatible(t, cc) and t >= MIN_ARCH), reverse=True)
    if compat:
        t = arch_name(compat[0])
        return t, f"nvcc {tc.version} cannot emit {own}; using binary-compatible {t} (same major)"
    raise BuildError(f"nvcc {tc.version} ({tc.nvcc}) cannot target {caps.name} ({own}) and has no binary-compatible "
                     f"target (supported: {' '.join(tc.real_archs) or 'unknown'}); install a CUDA toolkit that "
                     f"supports {own} (and set NVCC=...), or use backend='triton'")


# ---------------------------------------------------------------------- build identity
def extra_flags() -> Tuple[str, ...]:
    return tuple(os.environ.get(EXTRA_ENV, "").split())


def build_kind(extra: Tuple[str, ...]) -> str:
    if any("SP_ABLATE" in f for f in extra):
        return "ablation"
    return "development" if extra else "production"


def cache_root() -> Path:
    if os.environ.get(CACHE_ENV):
        return Path(os.environ[CACHE_ENV]).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "granulelayer" / "cuda"


def include_closure(src: Path, seen=None) -> set:
    """Transitive local #include "..." closure (paths resolved relative to the including file)."""
    seen = set() if seen is None else seen
    for line in src.read_text().splitlines():
        line = line.strip()
        if line.startswith('#include "'):
            dep = (src.parent / line.split('"')[1]).resolve()
            if not dep.exists():
                raise BuildError(f"{src.name}: included file {line.split(chr(34))[1]} not found")
            if dep not in seen:
                seen.add(dep)
                include_closure(dep, seen)
    return seen


@dataclass(frozen=True)
class BuildSpec:
    bundle: str
    target: str
    kind: str
    flags: Tuple[str, ...]
    toolchain: ToolchainInfo = field(repr=False, compare=False)

    @property
    def nvcc_flags(self) -> List[str]:
        return [*BASE_FLAGS, f"-arch={self.target}", f"-DSP_BUILD_KIND_ID={KIND_ID[self.kind]}",
                f"-DSP_BUNDLE_ID={BUNDLE_ID[self.bundle]}", *self.flags]

    def unit_hash(self, unit: Path) -> str:
        h = hashlib.sha256()
        for part in (BUILD_SCHEMA, self.bundle, self.kind, self.target, " ".join(self.nvcc_flags),
                     self.toolchain.version_text, self.toolchain.host_compiler):
            h.update(part.encode())
            h.update(b"\0")
        for p in [unit.resolve()] + sorted(include_closure(unit)):
            h.update(str(p.relative_to(SRC_DIR)).encode())
            h.update(b"\0")
            h.update(p.read_bytes())
        return h.hexdigest()[:20]

    def units(self) -> List[Path]:
        return [SRC_DIR / u for u in BUNDLES[self.bundle]]

    def identity(self) -> Dict:
        hashes = {str(u.relative_to(SRC_DIR)): self.unit_hash(u) for u in self.units()}
        lib = hashlib.sha256("".join(hashes[k] for k in sorted(hashes)).encode()).hexdigest()[:20]
        return dict(schema=BUILD_SCHEMA, bundle=self.bundle, kind=self.kind, target=self.target, flags=self.nvcc_flags,
                    nvcc=self.toolchain.nvcc, nvcc_version=self.toolchain.version,
                    host_compiler=self.toolchain.host_compiler, library_hash=lib, unit_hashes=hashes)


def make_spec(bundle: str, target: str, tc: Optional[ToolchainInfo] = None) -> BuildSpec:
    if bundle not in BUNDLES:
        raise BuildError(f"unknown bundle {bundle!r}")
    if parse_arch(target) < MIN_ARCH:  # complete-token check: a malformed target never gets a cache entry
        raise BuildError(f"{target}: the native kernels need sm_80+")
    extra = extra_flags()
    for f in extra:
        if f.startswith(("-arch", "-gencode", "--gpu-architecture", "-code")):
            raise BuildError(f"{EXTRA_ENV} may not set the target ({f}); use {ARCH_ENV}")
    return BuildSpec(bundle, target, build_kind(extra), extra, tc or toolchain())


# ----------------------------------------------------------------------------- building
@contextlib.contextmanager
def _file_lock(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _atomic_output(final: Path) -> Path:
    fd, tmp = tempfile.mkstemp(prefix=f".{final.name}.", suffix=".tmp", dir=final.parent)
    os.close(fd)
    return Path(tmp)


def _jobs() -> int:
    try:
        return max(1, int(os.environ.get(JOBS_ENV, "")))
    except ValueError:
        return max(1, min(os.cpu_count() or 4, 8))


_proc_lock = threading.Lock()


def library_path(spec: BuildSpec, ident: Optional[Dict] = None) -> Path:
    ident = ident or spec.identity()
    return (cache_root() / spec.target / "lib" / f"{spec.bundle}-{spec.kind}-{ident['library_hash']}"
            / f"libgranule_{spec.bundle}.so")


def build(spec: BuildSpec, verbose: bool = False) -> Tuple[Path, Dict]:
    """Compile (incrementally, per translation unit) and link a bundle; returns (library, identity).
    Reuses a published library with the same identity."""
    ident = spec.identity()
    lib = library_path(spec, ident)
    if lib.exists():
        return lib, ident
    root = cache_root() / spec.target
    with _proc_lock, _file_lock(root / "locks" / f"{lib.parent.name}.lock"):
        if lib.exists():  # built by another process while we waited
            return lib, ident
        obj_dir = root / "objs"
        obj_dir.mkdir(parents=True, exist_ok=True)
        lib.parent.mkdir(parents=True, exist_ok=True)
        nvcc = spec.toolchain.nvcc

        def compile_one(rel: str) -> Path:
            src = SRC_DIR / rel
            obj = obj_dir / f"{rel.replace('/', '__')[:-3]}-{ident['unit_hashes'][rel]}.o"
            if obj.exists():
                return obj
            tmp = _atomic_output(obj)
            cmd = [nvcc, *spec.nvcc_flags, "-c", str(src), "-o", str(tmp)]
            if verbose:
                print(" ".join(cmd), flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                tmp.unlink(missing_ok=True)
                raise BuildError(f"nvcc failed for {rel} ({spec.target}):\n{r.stderr}")
            os.replace(tmp, obj)
            return obj

        with ThreadPoolExecutor(max_workers=_jobs()) as ex:
            objs = list(ex.map(compile_one, BUNDLES[spec.bundle]))
        tmp = _atomic_output(lib)
        r = subprocess.run([nvcc, "-shared", f"-arch={spec.target}", *map(str, objs), "-o", str(tmp)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise BuildError(f"link failed ({spec.bundle}, {spec.target}):\n{r.stderr}")
        os.replace(tmp, lib)
    return lib, ident


def clear_cache(target: Optional[str] = None) -> Path:
    """Remove the build cache (all targets, or one).  Loaded libraries stay valid in this process."""
    p = cache_root() / target if target else cache_root()
    shutil.rmtree(p, ignore_errors=True)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description="GranuleLayer native CUDA build cache")
    ap.add_argument("--clean", action="store_true", help="remove the cache (all targets or --target)")
    ap.add_argument("--target", help="sm_<major><minor>, e.g. sm_86 (default: the current device's)")
    ap.add_argument("--bundle", default="production", choices=sorted(BUNDLES))
    ap.add_argument("--build", action="store_true", help="build the bundle now")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    if a.clean:
        print(f"removed {clear_cache(a.target)}")
        return 0
    try:
        if a.target:
            target = a.target
            parse_arch(target)
        else:
            from ..capabilities import query

            target, _ = resolve_target(query("cuda"))
        spec = make_spec(a.bundle, target)
        ident = spec.identity()
        print(f"cache {cache_root()}\nlibrary {library_path(spec, ident)} ({'present' if library_path(spec, ident).exists() else 'absent'})")
        if a.build:
            lib, _ = build(spec, a.verbose)
            print(f"built {lib}")
    except (BuildError, ToolchainError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
