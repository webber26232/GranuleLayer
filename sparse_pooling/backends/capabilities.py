"""Immutable, inspectable facts about an execution device and its toolchain.

Nothing here names a GPU model or carries a measured preference: tuning lives in
``backends.profiles``.  Capabilities answer "may this run here?", profiles "what is fastest here?".

Four statuses are kept apart throughout the backend (docs/gpu_portability.md):
  1. the source compiles for a target (toolchain: ``ToolchainInfo.supports``),
  2. the instructions and the launch are legal on the device (``features`` + the resource limits
     below + the launch guard in kernels/cuda/launch.cuh, which re-checks against the driver),
  3. numerical correctness was tested on the device (profile ``validation``),
  4. a performance policy was measured on the device (profile ``validation``).
"""
from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple

# Instruction families used by the native kernels, with the minimum compute capability at which
# the installed toolkit accepts them (PTX ISA "Target ISA Notes"; checked against ptxas 12.1 by
# tests/test_isa_requirements.py, which compiles each probe for the targets around the boundary).
FEATURE_MIN_CC: Dict[str, Tuple[int, int]] = {
    "mma.sync.m16n8k8.bf16": (8, 0),   # BF16 warp MMA, FP32 accumulate (forward / recompute)
    "mma.sync.m16n8k16.bf16": (8, 0),  # BF16 warp MMA, FP32 accumulate (gradient products)
    "cp.async": (8, 0),                # cp.async.cg + commit_group / wait_group / wait_all
    "ldmatrix": (7, 5),                # ldmatrix.sync.aligned.m8n8.{x2,x4}[.trans].shared.b16
    "movmatrix": (7, 5),               # movmatrix.sync.aligned.m8n8.trans.b16 (PTX 7.8)
    "ex2.approx.ftz.f32": (1, 0),      # all targets
    "set.gt.ftz": (1, 0),              # all targets (FTZ compare semantics: kernels/cuda/numerics.cuh)
    "atomicAdd.f32.global": (2, 0),
    "atomicCAS.b64.shared": (1, 2),
    "shfl.sync": (3, 0),
}

# The native CUDA kernel family (kernels/cuda) needs every one of these.
NATIVE_FAMILY_FEATURES = frozenset(FEATURE_MIN_CC)

# CUDA compute-capability table: maximum threads per block is 1024 on every capability >= 2.0
# (the driver value is re-checked per kernel by the launch guard).
_MAX_THREADS_PER_BLOCK = 1024


def features_for(cc: Tuple[int, int]) -> FrozenSet[str]:
    return frozenset(f for f, m in FEATURE_MIN_CC.items() if tuple(cc) >= m)


@dataclass(frozen=True)
class DeviceCapabilities:
    """Facts of one device.  ``source`` says where the values came from ("runtime" = the CUDA
    runtime through PyTorch; "synthetic" = a test fixture, never hardware evidence)."""

    backend: str                      # "cuda" | "cpu"
    vendor: str                       # "nvidia" | "none"
    index: int
    name: str
    uuid: str
    cc: Tuple[int, int]
    sm_count: int
    warp_size: int
    max_threads_per_block: int
    smem_per_block: int               # default dynamic + static allowance per block (bytes)
    smem_per_block_optin: int         # opt-in maximum per block (cudaFuncAttributeMaxDynamicSharedMemorySize)
    smem_per_sm: int
    regs_per_sm: int
    total_memory: int
    l2_bytes: int
    features: FrozenSet[str] = field(default_factory=frozenset)
    source: str = "runtime"

    @property
    def arch(self) -> str:
        return f"sm_{self.cc[0]}{self.cc[1]}"

    def has(self, *feats: str) -> bool:
        return all(f in self.features for f in feats)

    def missing(self, feats) -> Tuple[str, ...]:
        return tuple(sorted(f for f in feats if f not in self.features))

    def summary(self) -> Dict:
        d = asdict(self)
        d["features"] = sorted(self.features)
        d["arch"] = self.arch
        return d


def synthetic(name: str = "synthetic", cc=(8, 6), *, sm_count=80, smem_per_block_optin=99 * 1024,
              smem_per_block=48 * 1024, smem_per_sm=None, regs_per_sm=65536, warp_size=32,
              max_threads_per_block=_MAX_THREADS_PER_BLOCK, total_memory=24 << 30, l2_bytes=6 << 20,
              index=0, uuid="00000000-0000-0000-0000-000000000000", features=None) -> DeviceCapabilities:
    """A capability record for tests / resolver experiments.  Marked source="synthetic": a
    resolver decision made on it is control-flow evidence only, never hardware evidence."""
    return DeviceCapabilities(
        backend="cuda", vendor="nvidia", index=index, name=name, uuid=uuid, cc=tuple(cc), sm_count=sm_count,
        warp_size=warp_size, max_threads_per_block=max_threads_per_block, smem_per_block=smem_per_block,
        smem_per_block_optin=smem_per_block_optin,
        smem_per_sm=smem_per_sm if smem_per_sm is not None else smem_per_block_optin + 1024,
        regs_per_sm=regs_per_sm, total_memory=total_memory, l2_bytes=l2_bytes,
        features=frozenset(features) if features is not None else features_for(cc), source="synthetic")


CPU = DeviceCapabilities(backend="cpu", vendor="none", index=-1, name="cpu", uuid="", cc=(0, 0), sm_count=0,
                         warp_size=0, max_threads_per_block=0, smem_per_block=0, smem_per_block_optin=0,
                         smem_per_sm=0, regs_per_sm=0, total_memory=0, l2_bytes=0, features=frozenset(),
                         source="static")


@functools.lru_cache(maxsize=None)
def _query_cuda(index: int) -> DeviceCapabilities:
    import torch

    p = torch.cuda.get_device_properties(index)
    cc = (p.major, p.minor)
    return DeviceCapabilities(
        backend="cuda", vendor="nvidia", index=index, name=p.name, uuid=str(getattr(p, "uuid", "")), cc=cc,
        sm_count=p.multi_processor_count, warp_size=getattr(p, "warp_size", 32),
        max_threads_per_block=_MAX_THREADS_PER_BLOCK, smem_per_block=p.shared_memory_per_block,
        smem_per_block_optin=p.shared_memory_per_block_optin, smem_per_sm=p.shared_memory_per_multiprocessor,
        regs_per_sm=p.regs_per_multiprocessor, total_memory=p.total_memory,
        l2_bytes=getattr(p, "L2_cache_size", 0), features=features_for(cc))


def query(device) -> DeviceCapabilities:
    """Capabilities of the device a tensor / module lives on (not the current device).  Cached
    per device index: queried once at runtime construction, never per step."""
    import torch

    dev = torch.device(device)
    if dev.type == "cpu":
        return CPU
    if dev.type != "cuda":
        raise ValueError(f"no capability query for device type {dev.type!r}")
    if getattr(torch.version, "hip", None):
        raise RuntimeError("ROCm/HIP devices are not supported by the native CUDA backend (PTX kernels); "
                           "use backend='triton' where Triton supports the device")
    return _query_cuda(dev.index if dev.index is not None else torch.cuda.current_device())


# ------------------------------------------------------------------------------------- toolchain
@dataclass(frozen=True)
class ToolchainInfo:
    nvcc: str
    version: str          # "12.1"
    version_text: str     # full `nvcc --version` output (part of the build identity)
    real_archs: Tuple[str, ...]     # sm_* targets the compiler can emit (exact tokens, incl. sm_90a)
    virtual_archs: Tuple[str, ...]  # compute_* (recorded; no PTX target is ever selected from them)
    host_compiler: str    # `<host compiler> --version` first line (part of the build identity)

    def supports(self, arch: str) -> bool:
        return arch in self.real_archs


class ToolchainError(RuntimeError):
    pass


# Complete compiler target tokens.  Plain sm_<major><minor> (the major may have several digits:
# sm_100, sm_120) and the architecture-specific / family forms (sm_90a, sm_100f) are kept exactly as
# printed; the build decides which of them it may use (backends/cuda/build.py).
_GPU_CODE_RE = re.compile(r"sm_[1-9]\d*\d[af]?")
_GPU_ARCH_RE = re.compile(r"compute_[1-9]\d*\d[af]?")


def parse_gpu_list(text: str, kind: str = "sm") -> Tuple[str, ...]:
    """Targets in ``nvcc --list-gpu-code`` (kind "sm") / ``--list-gpu-arch`` (kind "compute") output,
    in order; whitespace-separated tokens that are not complete target names are ignored."""
    pat = _GPU_CODE_RE if kind == "sm" else _GPU_ARCH_RE
    return tuple(t for t in text.split() if pat.fullmatch(t))


def find_nvcc() -> Optional[str]:
    for cand in (os.environ.get("NVCC"), shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"):
        if cand and Path(cand).exists():
            return str(Path(cand).resolve())
    return None


@functools.lru_cache(maxsize=None)
def _toolchain(nvcc: str) -> ToolchainInfo:
    def run(*a):
        r = subprocess.run(a, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise ToolchainError(f"{' '.join(a)} failed: {r.stderr.strip()}")
        return r.stdout

    text = run(nvcc, "--version")
    m = re.search(r"release (\d+\.\d+)", text)
    version = m.group(1) if m else "unknown"
    try:
        real = parse_gpu_list(run(nvcc, "--list-gpu-code"), "sm")
    except ToolchainError:
        real = ()
    try:
        virt = parse_gpu_list(run(nvcc, "--list-gpu-arch"), "compute")
    except ToolchainError:
        virt = ()
    host = os.environ.get("NVCC_CCBIN") or os.environ.get("CUDAHOSTCXX") or shutil.which("g++") or ""
    hv = ""
    if host:
        try:
            hv = subprocess.run([host, "--version"], capture_output=True, text=True, timeout=60).stdout.splitlines()[0]
        except Exception:  # noqa: BLE001
            hv = host
    return ToolchainInfo(nvcc=nvcc, version=version, version_text=text.strip(), real_archs=real,
                         virtual_archs=virt, host_compiler=hv)


def toolchain() -> ToolchainInfo:
    """The nvcc used for native builds (``NVCC`` overrides PATH).  Raises ToolchainError with an
    actionable message when there is none: only when a CUDA operation is requested."""
    nvcc = find_nvcc()
    if nvcc is None:
        raise ToolchainError("nvcc not found: set NVCC=/path/to/nvcc (CUDA toolkit >= 11.8 for sm_80 - sm_90), "
                             "or use backend='triton'")
    return _toolchain(nvcc)
