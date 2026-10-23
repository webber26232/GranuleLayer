"""The instruction requirements declared in backends/capabilities.FEATURE_MIN_CC hold for the
installed toolkit: ptxas accepts each primitive at its declared minimum target and rejects it one
architecture below (compile only; no GPU needed).  Also: the production sources refuse to compile
for a target below sm_80 (isa_sm80.cuh #error)."""
import subprocess
from pathlib import Path

import pytest

from sparse_pooling.backends.capabilities import FEATURE_MIN_CC, find_nvcc, toolchain
from sparse_pooling.backends.cuda.build import SRC_DIR

PROBE = Path(__file__).resolve().parent / "data" / "isa_probe.cu"
NVCC = find_nvcc()
pytestmark = pytest.mark.skipif(NVCC is None, reason="nvcc not found")
CASES = {  # probe macro -> declared feature
    "P_MMA8": "mma.sync.m16n8k8.bf16", "P_MMA16": "mma.sync.m16n8k16.bf16", "P_LDM": "ldmatrix",
    "P_MOVM": "movmatrix", "P_CPASYNC": "cp.async",
}
BELOW = {(8, 0): "sm_75", (7, 5): "sm_70"}


def compiles(macro, arch, tmp_path):
    r = subprocess.run([NVCC, f"-arch={arch}", f"-D{macro}", "-c", str(PROBE), "-o", str(tmp_path / "p.o")],
                       capture_output=True, text=True)
    return r.returncode == 0, r.stderr


@pytest.mark.parametrize("macro", list(CASES))
def test_declared_minimum_target(macro, tmp_path):
    cc = FEATURE_MIN_CC[CASES[macro]]
    tc = toolchain()
    at = f"sm_{cc[0]}{cc[1]}"
    ok, err = compiles(macro, at, tmp_path)
    assert ok, f"{CASES[macro]} rejected at its declared minimum {at}: {err}"
    below = BELOW[cc]
    if tc.supports(below):
        ok, err = compiles(macro, below, tmp_path)
        assert not ok and "requires" in err, f"{CASES[macro]} accepted below its declared minimum ({below})"


def test_production_sources_reject_pre_sm80(tmp_path):
    tc = toolchain()
    if not tc.supports("sm_75"):
        pytest.skip("toolkit cannot emit sm_75")
    r = subprocess.run([NVCC, "-arch=sm_75", "-std=c++17", "-c", str(SRC_DIR / "sg_det.cu"), "-o", str(tmp_path / "a.o")],
                       capture_output=True, text=True)
    assert r.returncode == 0  # the reduction kernel uses no sm_80 instruction
    r = subprocess.run([NVCC, "-arch=sm_75", "-std=c++17", "-c", str(SRC_DIR / "abi.cu"), "-o", str(tmp_path / "b.o")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "need sm_80 or newer" in r.stderr
