"""Canonical data of the A3mean follow-up (docs/experiments/a3mean_long_horizon_batch_scaling.md).

Every input vector belongs to one canonical block of 512 independent N(0, I) rows, named by (study, split, data seed, block id,
D0).  The generator seed of a block is the first 63 bits of the SHA-256 of that name: stable across processes and machines, and
independent of Python's randomized hash().  A direct batch of 8,192 rows concatenates 16 consecutive blocks, and accumulated
execution consumes those same blocks one at a time, so compared batch modes share exact prefixes by sample position.  There is
no wrap-around: a block id grows with the sample position for the whole run.

Validation and test blocks are fixed across student seeds (data seed 0); training and calibration blocks belong to the student
seed.  The rows are independent vectors with no temporal relationship: they are not video frames or trajectories.
"""
from __future__ import annotations

import hashlib

import torch

STUDY = "a3mean_followup/1"
BLOCK = 512
SPLITS = ("train", "calibration", "validation", "test")
N_VALIDATION, N_TEST = 8, 16
HELDOUT_DATA_SEED = 0


def block_seed(split: str, data_seed: int, block: int, D0: int) -> int:
    if split not in SPLITS:
        raise ValueError(f"split {split!r}")
    if block < 0:
        raise ValueError("block id < 0")
    name = f"{STUDY}|{split}|{data_seed}|{block}|{D0}".encode()
    return int.from_bytes(hashlib.sha256(name).digest()[:8], "big") & (2 ** 63 - 1)


def block(split: str, data_seed: int, block_id: int, D0: int) -> torch.Tensor:
    """One canonical block [512, D0], FP32, on the CPU."""
    g = torch.Generator(device="cpu").manual_seed(block_seed(split, data_seed, block_id, D0))
    return torch.randn(BLOCK, D0, generator=g)


def blocks(split: str, data_seed: int, first: int, n: int, D0: int) -> torch.Tensor:
    return torch.cat([block(split, data_seed, first + i, D0) for i in range(n)])


def checksum(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def validation_ids():
    return range(N_VALIDATION)


def test_ids():
    return range(N_TEST)


def split_manifest(D0s=(256, 1024), seeds=(0, 1, 2)) -> dict:
    """Identifiers and checksums of the held-out splits, the calibration blocks and representative training blocks."""
    out = dict(study=STUDY, block_rows=BLOCK, heldout_data_seed=HELDOUT_DATA_SEED, D0={})
    for D0 in D0s:
        d = dict(validation={i: checksum(block("validation", HELDOUT_DATA_SEED, i, D0)) for i in validation_ids()},
                 test={i: checksum(block("test", HELDOUT_DATA_SEED, i, D0)) for i in test_ids()}, seeds={})
        for s in seeds:
            d["seeds"][s] = dict(calibration=checksum(block("calibration", s, 0, D0)),
                                 train_block_0=checksum(block("train", s, 0, D0)),
                                 train_block_16383=checksum(block("train", s, 16_383, D0)))
        out["D0"][D0] = d
    return out
