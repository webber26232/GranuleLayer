"""Synthetic tasks of the learned-reduction quality study.  All are synthetic diagnostics, NOT downstream-accuracy claims.

Common chain: x0 [M, D0] -> proj = Linear(D0, 1024) -> BF16 -> reducer model -> y [M, 1024] -> head = Linear(1024, 64) -> MSE.
The proj / head budget is identical across reducers and reported separately.

compose   the historical k64_learn teacher (seed 31): t = tanh(A2 relu(A1 x0 + c1)), D0 = 256, teacher dense and signed,
          independent of every student graph / window order.
interact  second- and third-order input products: t_o = sqrt(1 / T) sum_a w_oa prod_{i in S_oa} x0_i (T = 16 terms per
          output, half of order 2, half of order 3, seed 41, D0 = 256): a compositional, non-additive target.
local     support-local group aggregation (counterbalance to the globally mixed tasks): t_o = scale sum_u tanh(2 w_u . x[S_u]),
          16 units per output, |S_u| = 8 inputs from a contiguous 64-input neighbourhood, seed 51, D0 = 1024, proj
          initialized to the identity so the input basis of the student's supports stays aligned with the teacher's.

Streams are disjoint: teacher seeds above; graph / init seeds in the base layer; proj / head 29 + seed; reducer P 1000 + seed;
readout 3000 + seed; train batch i = TRAIN_BASE + (i mod pool) + 1_000_000 data_seed; validation 9000..9003; locked test
LSW_TEST (fresh for this study); calibration of gamma from training batch 0 only.
"""
from __future__ import annotations

import math

import torch

M_TRAIN = 512
OUT = 64
TRAIN_BASE = 10_000
POOL = 256
VAL_SEEDS = tuple(range(9_000, 9_004))
LSW_TEST = tuple(range(49_000, 49_008))
TRAIN_EVAL_BATCHES = 4


def gaussian(seed: int, M: int, D0: int, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(M, D0, generator=g).to(device)


class Compose:
    name, D0, identity_proj = "compose", 256, False

    def __init__(self, device):
        g = torch.Generator(device="cpu").manual_seed(31)
        self.A1 = (torch.randn(512, self.D0, generator=g) / math.sqrt(self.D0)).to(device)
        self.c1 = (0.1 * torch.randn(512, generator=g)).to(device)
        self.A2 = (torch.randn(OUT, 512, generator=g) * math.sqrt(2.0 / 512)).to(device)

    def __call__(self, x0):
        return torch.tanh(torch.relu(x0 @ self.A1.t() + self.c1) @ self.A2.t())


class Interact:
    name, D0, identity_proj = "interact", 256, False
    T = 16

    def __init__(self, device):
        g = torch.Generator(device="cpu").manual_seed(41)
        T, D0 = self.T, self.D0
        self.i2 = torch.stack([torch.randperm(D0, generator=g)[:2] for _ in range(OUT * T // 2)]).to(device)
        self.i3 = torch.stack([torch.randperm(D0, generator=g)[:3] for _ in range(OUT * T // 2)]).to(device)
        self.w2 = (torch.randn(OUT, T // 2, generator=g) / math.sqrt(T)).to(device)
        self.w3 = (torch.randn(OUT, T // 2, generator=g) / math.sqrt(T)).to(device)

    def __call__(self, x0):
        p2 = x0[:, self.i2].prod(2).reshape(x0.shape[0], OUT, -1)
        p3 = x0[:, self.i3].prod(2).reshape(x0.shape[0], OUT, -1)
        return (p2 * self.w2).sum(2) + (p3 * self.w3).sum(2)


class Local:
    name, D0, identity_proj = "local", 1024, True
    U, S = 16, 8

    def __init__(self, device):
        g = torch.Generator(device="cpu").manual_seed(51)
        n = OUT * self.U
        start = torch.randint(0, self.D0 - 64, (n,), generator=g)
        self.idx = torch.stack([s + torch.randperm(64, generator=g)[:self.S] for s in start]).to(device)
        self.w = (torch.randn(n, self.S, generator=g) / math.sqrt(self.S)).to(device)
        with torch.no_grad():
            t = self._raw(gaussian(777_001, 4096, self.D0, device))
        self.scale = float(1.0 / t.std())

    def _raw(self, x0):
        v = torch.tanh(2.0 * (x0[:, self.idx] * self.w).sum(2))
        return v.reshape(x0.shape[0], OUT, self.U).sum(2)

    def __call__(self, x0):
        return self._raw(x0) * self.scale


TASKS = dict(compose=Compose, interact=Interact, local=Local)


def shifted(x0: torch.Tensor, kind: str):
    """Fixed distribution shifts of the held-out inputs (the teacher is applied to the SHIFTED input)."""
    if kind == "scale":  # input scale 1.5
        return x0 * 1.5
    if kind == "corr":  # half of the variance in 8 shared latent directions (fixed seed): correlated inputs
        g = torch.Generator(device="cpu").manual_seed(61)
        D0 = x0.shape[1]
        F = (torch.randn(D0, 8, generator=g) / math.sqrt(8)).to(x0.device)
        c = torch.randn(x0.shape[0], 8, generator=torch.Generator(device="cpu").manual_seed(62)).to(x0.device)
        return math.sqrt(0.5) * x0 + math.sqrt(0.5) * c @ F.t()
    raise ValueError(kind)


SHIFTS = ("scale", "corr")
