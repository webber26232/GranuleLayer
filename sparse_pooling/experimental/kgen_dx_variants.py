"""EXPERIMENTAL K16 dX variant selector of the kgen bundle (results/INTERACTION_COVERAGE_K16_DX.md section 5).

A variant changes ONLY the dX launch of a K = 16 kgen layer (indexed ``KgenLayer`` / ``LoweredOneTile`` and the K16
pack of a grouped layer): compile-time token tile MT (16 MT tokens per CTA) and warps NW, runtime warps per FP32
shared-memory slab copy WG (1 = warp-private read-modify-write, > 1 = shared copies with the 64-bit CAS).  The
function (support, weights, z recomputation, BF16 hi + lo split dZ, FP32 accumulation, activation decisions, pooling
denominator, one final dX cast) is unchanged; only the reduction order of the FP32 sums differs.

Requests (``layer.dx_variant``):

* ``"legacy"`` (DEFAULT, every device): the original sp_kg_k16 launch (MT 4, NW 8, WG = backends.dispatch.
  kgen_dx_group_warps: 2 at D 1024 / S 64 on a 99 KiB device) -- byte-identical code and arguments;
* ``"mt<MT>_nw<NW>_wg<WG>"``: an explicit variant (sp_kg_k16_dxv); rejected with a reason if not compiled or illegal
  on the current device / shape (``reason``: NW % WG, shared memory vs the device opt-in limit, kgen_reason);
* ``"auto"``: the empirical tuning table ``TUNED`` for the measured device / shape, else legacy.

Portability: legality is capability based (opt-in shared memory read from the device); the tuning table is EMPIRICAL
and keyed by the exact RTX 3090 identity (backends/profiles/rtx3090.is_rtx3090) and the shape (D, S, K, C); any other
device or shape gets the generic legacy launch.  Resolutions are cached per (device index, name, cc, D, S, request).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

SCREEN = (  # (MT, NW, WG): the declared screen (section 5.2); (4, 8, 2) is the legacy tiling
    (4, 8, 2), (4, 8, 4), (4, 8, 8), (2, 8, 1), (2, 8, 2), (4, 4, 1), (4, 4, 2), (2, 4, 1))
# empirical 3090 table (filled only from a qualifying confirmation, section 5.3): {(D, S, K, C): name}
TUNED: Dict[Tuple[int, int, int, int], str] = {}
TUNED_EVIDENCE = "none: no variant has been selected (legacy is the default everywhere)"
_RE = re.compile(r"mt(\d+)_nw(\d+)_wg(\d+)(?:_c16)?$")
_cache: Dict[tuple, "DxVariant"] = {}


@dataclass(frozen=True)
class DxVariant:
    name: str
    mt: int
    nw: int
    wg: int
    impl: int = 0  # 0 generic template
    legacy: bool = False
    source: str = "explicit"

    def describe(self) -> dict:
        return dict(name=self.name, dx_mt=self.mt, dx_warps=self.nw, dx_group_warps=self.wg, impl=self.impl,
                    legacy_entry=self.legacy, source=self.source, tokens_per_cta=16 * self.mt)


def smem_bytes(D: int, S: int, mt: int, nw: int, wg: int) -> int:
    """Dynamic shared memory of a K16 dX launch: align16(2 D) + 2 S (16 MT) + 4 (NW / WG) S (16 MT)."""
    bm = 16 * mt
    return ((2 * D + 15) // 16) * 16 + 2 * S * bm + 4 * (nw // wg) * S * bm


def parse(name: str) -> Tuple[int, int, int]:
    m = _RE.fullmatch(name)
    if not m:
        raise ValueError(f"dX variant {name!r} (expected 'legacy', 'auto' or 'mt<MT>_nw<NW>_wg<WG>')")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def reason(caps, backend, D: int, S: int, K: int, mt: int, nw: int, wg: int, impl: int = 0) -> str:
    """Why the variant cannot run on this device / shape ("" if it can)."""
    from ..backends.dispatch import smem_limit

    if K != 16:
        return f"dX variants exist for K = 16 only (K = {K})"
    if (mt, nw, impl) not in set(backend.dxv_compiled):
        return f"(MT {mt}, NW {nw}, impl {impl}) is not compiled (compiled: {backend.dxv_compiled})"
    if wg < 1 or nw % wg:
        return f"WG = {wg} must divide NW = {nw}"
    need, lim = smem_bytes(D, S, mt, nw, wg), smem_limit(caps)
    if need > lim:
        return f"needs {need / 1024:.1f} KiB of shared memory > device limit {lim / 1024:.1f} KiB"
    return ""


def resolve(request: Optional[str], caps, backend, D: int, S: int, K: int, C: int, device_index: int = 0,
            legacy_wg: int = 2) -> DxVariant:
    """The DxVariant of a request (module docstring); raises ValueError for an illegal explicit request."""
    req = request or "legacy"
    key = (device_index, caps.name, tuple(caps.cc), D, S, K, C, req, legacy_wg)
    if key in _cache:
        return _cache[key]
    if req == "legacy" or K != 16:
        v = DxVariant("legacy", 4, 8, legacy_wg, 0, True, "default")
    elif req == "auto":
        from ..backends.profiles.rtx3090 import is_rtx3090

        name = TUNED.get((D, S, K, C)) if is_rtx3090(caps)[0] else None
        if name is None:
            v = DxVariant("legacy", 4, 8, legacy_wg, 0, True, "auto: no tuned entry for this device / shape")
        else:
            mt, nw, wg = parse(name)
            why = reason(caps, backend, D, S, K, mt, nw, wg)
            v = (DxVariant("legacy", 4, 8, legacy_wg, 0, True, f"auto: tuned {name} illegal here ({why})") if why else
                 DxVariant(name, mt, nw, wg, 0, False, f"auto: RTX 3090 table ({TUNED_EVIDENCE})"))
    else:
        mt, nw, wg = parse(req)
        why = reason(caps, backend, D, S, K, mt, nw, wg)
        if why:
            raise ValueError(f"dX variant {req}: {why}")
        v = DxVariant(req, mt, nw, wg, 0, False, "explicit")
    _cache[key] = v
    return v
