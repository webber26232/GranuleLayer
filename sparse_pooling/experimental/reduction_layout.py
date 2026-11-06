"""Logical reduction geometry and physical packing of the learned sliding-window reducers (EXPERIMENTAL;
docs/experiments/learned_sliding_window_reduction.md).

Notation (never overloaded with the input-supergroup S or the permutation round R):

    H  valid sparse hidden neurons     L  window length        s  window stride
    J  outputs of the reduction        F  physical coefficient slots per neuron

Logical semantics: hidden neuron identity is the global neuron stream of ``layout.SparseLayout`` (output g owns n_g
valid slots, slot n of g is neuron offset_g + n); graph padding is never part of the stream.  With ``boundary="valid"``

    windows W = floor((H - L) / s) + 1,  window w covers [w s, w s + L),  no padding, no wraparound

and each window has ``channels`` independent outputs, so J = W * channels and output j belongs to window j // channels.
Logical coefficients are P[j, r], r in [0, L).  Overlap (L > s) with channels > 1 is not defined here (ValueError).

Physical packing: P_packed [G * NB * C, F] BF16, neuron-major in the storage order of the sparse layout, so a C = 16
record's coefficients are contiguous like its weights.  A block routes to the F consecutive outputs w0[block] + f;
this is the stride-block reordering of the window sum.  A slot is ``member`` iff its output really contains the neuron.
With the aligned shapes every in-range slot is a member; for irregular shapes (non-multiple L / s, records crossing a
stride boundary) ``needs_grad_mask`` is True: non-member slots hold zero and their gradient is masked once per backward.

``aligned_constraints`` is this experiment's performance-shape requirement, not a redefinition of legal API widths.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch

H_QUANTUM = 262_144
F_MAX = 8  # the kernels' n8 column block


@dataclass(frozen=True)
class ReductionSpec:
    hidden_dim: int
    window: int  # L
    stride: int  # s
    channels: int = 1
    boundary: str = "valid"

    def __post_init__(self):
        if self.boundary != "valid":
            raise ValueError(f"boundary {self.boundary!r}: only 'valid' is implemented (no padding, no wraparound); "
                             "other policies must be separately named variants")
        if self.stride < 1 or self.window < 1 or self.channels < 1:
            raise ValueError("window, stride and channels must be >= 1")
        if self.window > self.hidden_dim:
            raise ValueError(f"window L = {self.window} > H = {self.hidden_dim}")
        if self.window < self.stride:
            raise ValueError(f"window L = {self.window} < stride s = {self.stride} leaves uncovered gaps")
        if self.channels > 1 and self.window != self.stride:
            raise ValueError("channels > 1 is defined for non-overlapping windows (L == s) only")

    @property
    def windows(self) -> int:
        return (self.hidden_dim - self.window) // self.stride + 1

    @property
    def J(self) -> int:
        return self.windows * self.channels

    @property
    def weights(self) -> int:
        """Valid first-stage coefficients."""
        return self.J * self.window

    @property
    def uncovered_tail(self) -> int:
        return self.hidden_dim - ((self.windows - 1) * self.stride + self.window)

    def first_last(self, i: torch.Tensor):
        """(first, last) output of global neuron ids i (int64); first > last means uncovered."""
        s, L, ch = self.stride, self.window, self.channels
        wmax = torch.clamp(i // s, max=self.windows - 1)
        wmin = torch.clamp((i - L + s) // s, min=0)  # ceil((i - L + 1) / s)
        covered = (wmin * s <= i) & (i < wmax * s + L) & (wmin <= wmax)
        first = wmin * ch
        last = wmax * ch + ch - 1
        return torch.where(covered, first, torch.full_like(first, 1)), torch.where(covered, last, torch.zeros_like(last))

    def multiplicity(self) -> torch.Tensor:
        """[H] windows (not outputs) covering each neuron (boundary taper; small H: O(H))."""
        i = torch.arange(self.hidden_dim, dtype=torch.int64)
        first, last = self.first_last(i)
        return torch.where(first <= last, (last - first + 1) // self.channels, torch.zeros_like(first))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "ReductionSpec":
        return ReductionSpec(**json.loads(s))


def aligned_constraints(spec: ReductionSpec, multi_channel: bool = False) -> dict:
    """The intended constraints of a full-scale configuration; ``assert_aligned`` raises if one fails."""
    H, L, s = spec.hidden_dim, spec.window, spec.stride
    c = {"H % 262144 == 0": H % H_QUANTUM == 0, "H % s == 0": H % s == 0, "L % 256 == 0": L % 256 == 0,
         "s % 256 == 0": s % 256 == 0, "(H - L) % s == 0": (H - L) % s == 0}
    if multi_channel:
        c["H % L == 0"] = H % L == 0
    return c


def assert_aligned(spec: ReductionSpec, multi_channel: bool = False):
    bad = [k for k, v in aligned_constraints(spec, multi_channel).items() if not v]
    if bad:
        raise ValueError(f"{spec}: violated experiment shape constraints {bad}")


def core_specs(H: int, L: int = 10_240, s: int = 2_048) -> dict:
    """The reducer geometries of the model matrix at width H (recomputed, never copied from the primary table)."""
    out = dict(A3=ReductionSpec(H, s, s), A4=ReductionSpec(H, L, s))
    if H % L == 0:
        out["MC"] = ReductionSpec(H, L, L, channels=L // s)
    return out


def parameter_counts(H: int, O: int = 1024, D: int = 1024, G: int = 1024, K: int = 16, L: int = 10_240,
                     s: int = 2_048) -> dict:
    """Weight accounting of the model matrix (biases separately; sparse projection excluded from 'added')."""
    J5 = (H - L) // s + 1
    nb = H // s
    rows = {
        "A0": dict(intermediate=G, added_weights=0, added_biases=0),
        "A1": dict(intermediate=G, added_weights=G * O, added_biases=O),
        "A2": dict(intermediate=nb, added_weights=nb * O, added_biases=O),
        "A3": dict(intermediate=nb, added_weights=nb * s + nb * O, added_biases=nb + O),
        "A4": dict(intermediate=J5, added_weights=J5 * L + J5 * O, added_biases=J5 + O),
        "A5": dict(intermediate=J5, added_weights=J5 * O, added_biases=O),
    }
    if H % L == 0:
        w = H // L
        ch = L // s
        rows["MC"] = dict(intermediate=w * ch, added_weights=w * ch * L + w * ch * O, added_biases=w * ch + O)
    return dict(H=H, sparse_weights=K * H, sparse_biases=H, models=rows)


def three_stage_counts(H: int, L1: int = 10_240, s1: int = 2_048, L2: int = 256, s2: int = 4, O: int = 1024) -> dict:
    J1 = (H - L1) // s1 + 1
    J2 = (J1 - L2) // s2 + 1
    return dict(J1=J1, J2=J2, uncovered_second_stage=J1 - ((J2 - 1) * s2 + L2), P1=J1 * L1, P2=J2 * L2, V=J2 * O,
                weights=J1 * L1 + J2 * L2 + J2 * O)


class Routing:
    """Logical-to-physical map of a reduction over a K / C storage layout [G, NB, C] (module docstring)."""

    def __init__(self, spec: ReductionSpec, counts: torch.Tensor, NB: int, C: int):
        counts = counts.to("cpu", torch.int64)
        if int(counts.sum()) != spec.hidden_dim:
            raise ValueError(f"layout holds {int(counts.sum())} valid neurons, reduction H = {spec.hidden_dim}")
        self.spec, self.NB, self.C, self.G = spec, NB, C, counts.numel()
        self.counts = counts
        self.slots = self.G * NB * C
        off = torch.cumsum(counts, 0) - counts
        s = torch.arange(NB * C, dtype=torch.int64)
        valid = s[None, :] < counts[:, None]  # [G, NB C]
        gid = off[:, None] + s[None, :]  # global neuron id of (g, slot)
        first, last = spec.first_last(gid.reshape(-1))
        valid = valid.reshape(-1)
        covered = valid & (first <= last)
        # per block: the outputs of its covered neurons
        big = torch.iinfo(torch.int64).max // 4
        lastb = torch.where(covered, last, torch.full_like(last, -big)).view(-1, C).max(1).values
        firstb = torch.where(covered, first, torch.full_like(first, big)).view(-1, C).min(1).values
        live = lastb > -big
        span = (lastb - firstb + 1)[live]
        self.F = int(span.max()) if span.numel() else 1
        if self.F > F_MAX:
            raise ValueError(f"a {C}-neuron record routes to {self.F} consecutive outputs (> {F_MAX}): reject this "
                             f"shape (L = {spec.window}, s = {spec.stride}, channels = {spec.channels})")
        # w0 = last - (F - 1): slots of boundary blocks map outside [0, J) and read padded (zero) Delta rows
        w0 = lastb - (self.F - 1)
        w0[~live] = 0
        self.w0_blocks = w0.to(torch.int32)  # [G NB]
        self.live_blocks = live
        self.valid = valid
        self.covered = covered
        self.first, self.last = first, last
        rep_w0 = torch.repeat_interleave(w0, C)
        f_first = first - rep_w0
        f_last = last - rep_w0
        self.f_first, self.f_last = f_first.to(torch.int16), f_last.to(torch.int16)
        # non-member slots inside [0, J): only irregular shapes have them
        nmember = (f_last - f_first + 1).clamp(min=0) * covered
        # in-range slots per neuron (vectorised intersection of [w0, w0 + F) with [0, J))
        lo = torch.clamp(rep_w0, min=0)
        hi = torch.clamp(rep_w0 + self.F, max=spec.J)
        n_in_range = (hi - lo).clamp(min=0)
        self.needs_grad_mask = bool(((n_in_range != nmember) & valid).any()) or bool((~covered & valid).any())
        self.uncovered_neurons = int((valid & ~covered).sum())
        self.packed_valid = int(nmember.sum())

    # -------------------------------------------------- masks and maps
    def member_mask(self, device="cpu") -> torch.Tensor:
        """[slots, F] bool: valid coefficients."""
        f = torch.arange(self.F, dtype=torch.int16)
        m = (f[None, :] >= self.f_first[:, None]) & (f[None, :] <= self.f_last[:, None]) & self.covered[:, None]
        return m.to(device)

    def global_ids(self) -> torch.Tensor:
        """[slots] global neuron id (padded slots: -1)."""
        off = torch.cumsum(self.counts, 0) - self.counts
        s = torch.arange(self.NB * self.C, dtype=torch.int64)
        gid = (off[:, None] + s[None, :]).reshape(-1)
        return torch.where(self.valid, gid, torch.full_like(gid, -1))

    def slot_table(self):
        """(slot position p, f, output j, r): the bijection of member coefficients and logical P[j, r]."""
        m = self.member_mask()
        p, f = torch.nonzero(m, as_tuple=True)
        gid = self.global_ids()[p]
        j = torch.repeat_interleave(self.w0_blocks.to(torch.int64), self.C)[p] + f
        start = (j // self.spec.channels) * self.spec.stride
        return p, f, j, gid - start

    def pack(self, logical: torch.Tensor) -> torch.Tensor:
        """Logical [J, L] -> packed [slots, F] (small H; tests)."""
        p, f, j, r = self.slot_table()
        dev = logical.device
        out = torch.zeros((self.slots, self.F), dtype=logical.dtype, device=dev)
        out[p.to(dev), f.to(dev)] = logical[j.to(dev), r.to(dev)]
        return out

    def unpack(self, packed: torch.Tensor) -> torch.Tensor:
        """Packed [slots, F] -> logical [J, L] (small H; tests)."""
        p, f, j, r = self.slot_table()
        dev = packed.device
        out = torch.zeros((self.spec.J, self.spec.window), dtype=packed.dtype, device=dev)
        out[j.to(dev), r.to(dev)] = packed[p.to(dev), f.to(dev)]
        return out

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.spec.to_json().encode())
        h.update(f"{self.G},{self.NB},{self.C},{self.F}".encode())
        h.update(self.w0_blocks.numpy().tobytes())
        return h.hexdigest()

    def report(self) -> dict:
        return dict(spec=json.loads(self.spec.to_json()), windows=self.spec.windows, J=self.spec.J, F=self.F,
                    physical_slots=self.slots * self.F, valid_coefficients=self.packed_valid,
                    padding_slots=self.slots * self.F - self.packed_valid,
                    padding_fraction=1.0 - self.packed_valid / max(self.slots * self.F, 1),
                    uncovered_neurons=self.uncovered_neurons, uncovered_tail=self.spec.uncovered_tail,
                    needs_grad_mask=self.needs_grad_mask, routing_fingerprint=self.fingerprint())


# --------------------------------------------------------------------------- static estimator
def static_estimate(H: int, M: int, spec: ReductionSpec, O: int = 1024, K: int = 16, D: int = 1024, F=None) -> dict:
    """Accounting estimates (NOT measurements): compact sizes vs the M x H tensors the production path must not
    allocate, and logical arithmetic under the explicit convention 'one multiply-add = 2 FLOPs', excluding padding,
    activation and recomputation."""
    J, L = spec.J, spec.window
    if F is None:
        F = math.ceil(L / spec.stride) if spec.window != spec.stride else spec.channels
    gib = 2 ** 30
    return dict(H=H, M=M, J=J, L=L, s=spec.stride, F=F, P_weights=spec.weights, V_weights=J * O,
                estimate_note="static accounting, not a measurement",
                forbidden=dict(h_bf16_gib=M * H * 2 / gib, unfolded_bf16_gib=M * J * L * 2 / gib),
                compact=dict(u_fp32_mib=M * J * 4 / 2 ** 20, u_bf16_mib=M * J * 2 / 2 ** 20),
                parameters_mib=dict(P_bf16=H * F * 2 / 2 ** 20, P_main_grad=H * F * 4 / 2 ** 20,
                                    P_bf16_moments=H * F * 4 / 2 ** 20, V_fp32=J * O * 4 / 2 ** 20),
                flops=dict(reduction_P=2 * M * spec.weights, reduction_V=2 * M * J * O,
                           total_reduction_forward=2 * M * (spec.weights + J * O), sparse_projection=2 * M * H * K,
                           convention="2 FLOPs per multiply-add; one forward; no padding / recomputation"))
