"""Importing the package and host-only work (graph generation, bucket statistics, plan resolution
on synthetic devices) must neither initialize CUDA nor need nvcc (CPU-only machines, CI)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = r"""
import os, sys, torch
import sparse_pooling
from sparse_pooling import PhysicalPlan, SparsePoolingLayer
from sparse_pooling.backends.capabilities import synthetic, query
from sparse_pooling.backends.dispatch import Resolver
from sparse_pooling.backends.profiles import select
from sparse_pooling.kernels import cuda_ops                      # facade: no build at import
from sparse_pooling.supergroups import ConnectivityPlan, bucket_statistics
lay = SparsePoolingLayer(256, 50_000, 16, shared_neurons=8, pooling="mean", device="cpu",
                         connectivity=ConnectivityPlan(16, 4))   # host graph generation
st = lay.graph_stats
caps = synthetic("NVIDIA GeForce RTX 3090", (8, 6), sm_count=82)
assert select(caps).profile.name == "rtx3090"
assert query("cpu").backend == "cpu"
info = lay.explain(1024)
assert "CPU" in info["backend"], info
assert not torch.cuda.is_initialized(), "CUDA was initialized"
print("ok", st["generation_seconds"] >= 0)
"""


def test_cpu_only_import_without_nvcc(tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", NVCC=str(tmp_path / "no-nvcc"), PATH="/usr/bin:/bin",
               SPARSE_POOL_CACHE_DIR=str(tmp_path / "cache"), PYTHONPATH=str(ROOT))
    env.pop("SPARSE_POOL_PROFILE", None)  # the script checks automatic matching
    r = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr
    assert not (tmp_path / "cache").exists()  # nothing was built


def test_missing_nvcc_error_is_actionable(tmp_path):
    env = dict(os.environ, NVCC=str(tmp_path / "no-nvcc"), PATH="/usr/bin:/bin", PYTHONPATH=str(ROOT))
    code = ("from sparse_pooling.backends import capabilities as c\n"
            "c.find_nvcc = lambda: None  # also hide /usr/local/cuda/bin/nvcc\n"
            "try:\n    c.toolchain()\nexcept c.ToolchainError as e:\n    print('ERR', e)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert "ERR nvcc not found" in r.stdout and "NVCC=" in r.stdout and "triton" in r.stdout
