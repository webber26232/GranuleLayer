"""K16 dX tiling variants of the kgen bundle: names, the screened set and thin wrappers over the dispatch authority.

A variant changes ONLY the dX launch of a K = 16 kgen layer: compile-time token tile MT (16 MT tokens per CTA) and
warps NW (32 NW threads), runtime warps per FP32 shared-memory slab copy WG (1 = warp-private read-modify-write,
> 1 = NW / WG shared copies updated with the 64-bit CAS).  The function (support, weights, z recomputation, BF16
hi + lo split dZ, FP32 accumulation, activation decisions, pooling denominator, one final dX cast) is unchanged; only
the reduction order of the FP32 sums differs (atomic paths are not bitwise deterministic).  Forward and dW launches
are never touched by the dX choice.

Requests (``layer.dx_variant``):

* ``"legacy"``: the original sp_kg_k16 launch (MT 4, NW 8, WG = backends.dispatch.kgen_dx_group_warps: 2 at D 1024 /
  S 64 on a 99 KiB device) -- byte-identical code and arguments.  The default of the generic experimental
  ``KgenLayer``; the A/B and troubleshooting override of the named Cover-TopK/16 layer;
* ``"auto"``: the selected tuning profile's measured choice for the call (backends/profiles: the RTX 3090 profile's
  ``KGEN_DX_AUTO`` scope; every other profile, incl. a forced ``generic_cuda``, keeps legacy).  The default of
  ``sparse_pooling.cover_topk16.CoverTopK16Layer``;
* ``"mt<MT>_nw<NW>_wg<WG>"``: an explicit variant; rejected with a reason if not compiled or illegal on the current
  device / shape.

Resolution is backends.dispatch.kgen_dx_resolve (precedence, legality, cache keys incl. the batch and the call
context).  The research-branch ``TUNED`` dict (always empty there, so "auto" meant legacy) is replaced by the profile
question; its history: research commit f4c538a, results/INTERACTION_COVERAGE_K16_DX.md section 6.
"""
from __future__ import annotations

from typing import Tuple

from ..backends import dispatch as _D
from ..backends.cuda import variants as _V

SCREEN = (  # (MT, NW, WG): the declared screen of the research round; (4, 8, 2) is the legacy tiling at D 1024 / S 64
    (4, 8, 2), (4, 8, 4), (4, 8, 8), (2, 8, 1), (2, 8, 2), (4, 4, 1), (4, 4, 2), (2, 4, 1))
DxVariant = _D.KgenDxDecision  # (name of the research branch's record)


def smem_bytes(D: int, S: int, mt: int, nw: int, wg: int) -> int:
    """Dynamic shared memory of a K16 dX launch: align16(2 D) + 2 S (16 MT) + 4 (NW / WG) S (16 MT)."""
    return _V.kgen_k16_dx_smem(D, S, mt, nw, wg)


def parse(name: str) -> Tuple[int, int, int]:
    return _D.kgen_dx_parse(name)


def reason(caps, backend, D: int, S: int, K: int, mt: int, nw: int, wg: int, impl: int = 0) -> str:
    """Why the variant cannot run on this device / shape ("" if it can)."""
    return _D.kgen_dx_variant_reason(caps, backend.dxv_compiled, D, S, K, mt, nw, wg, impl)
