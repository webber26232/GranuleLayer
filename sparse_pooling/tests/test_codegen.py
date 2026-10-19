"""Code-generation checks of representative production kernels (cuobjdump on the built library).

The fast / mask ReLU epilogues must keep the generic path's flush-to-zero compare semantics
(kernels/cuda/numerics.cuh: ptxas 12.1 was seen to emit a non-FTZ FSETP for a mask that is only
tested for non-zero); the tuned instantiations must keep their resource targets (swizzle4_capped:
<= 80 registers for 3 CTAs / SM) without spilling; tensor-core and async-copy instructions are used."""
import re
import shutil
import subprocess

import pytest
import torch

from conftest import requires_cuda_backend

pytestmark = [requires_cuda_backend, pytest.mark.skipif(shutil.which("cuobjdump") is None, reason="cuobjdump not found")]

KERNELS = {  # mangled name -> what it is
    "_Z12sg_dx_kernelILi1ELi8ELi4ELi8ELb1ELi2EEv6SgArgs": "grouped dX C=8 ReLU mask epilogue (baseline scatter)",
    "_Z13sgx_dx_kernelILi1ELi8ELi4ELi8ELb1ELi2ELi10ELi1EEv6SgArgs": "grouped dX swizzle4_capped, mask epilogue",
    "_Z13sgx_dx_kernelILi1ELi8ELi2ELi8ELb1ELi2ELi9ELi1EEv6SgArgs": "grouped dX private_t32 plan, mask epilogue",
    "_Z12sg_dw_kernelILi1ELi8ELi8ELi2ELi4ELi2ELb1ELi2ELi1ELi0EEv6SgArgs": "grouped dW C=8 ReLU fast epilogue",
}


@pytest.fixture(scope="module")
def lib():
    from sparse_pooling.backends.cuda.runtime import backend_for

    return str(backend_for(torch.device("cuda", 0)).native.path)


def sass(lib, fn):
    out = subprocess.run(["cuobjdump", "-sass", "-fun", fn, lib], capture_output=True, text=True).stdout
    assert "Function" in out, f"{fn} not in {lib}"
    return re.findall(r"\b([A-Z][A-Z0-9]*(?:\.[A-Z0-9_]+)*)\b(?= )", out)


def resources(lib, fn):
    out = subprocess.run(["cuobjdump", "-res-usage", lib], capture_output=True, text=True).stdout
    m = re.search(re.escape(fn) + r":\s*\n\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", out)
    assert m, fn
    return dict(zip(("reg", "stack", "shared", "local"), map(int, m.groups())))


@pytest.mark.parametrize("fn", list(KERNELS), ids=list(KERNELS.values()))
def test_relu_decisions_flush_subnormals(lib, fn):
    ops = sass(lib, fn)
    gt = [o for o in ops if o.startswith(("FSETP.GT", "FSET.BF.GT", "FSET.GT"))]
    assert gt, "no ReLU compare found"
    assert all(".FTZ" in o for o in gt), sorted(set(o for o in gt if ".FTZ" not in o))
    if "dx" in fn:  # the mask epilogue ANDs the FTZ mask into BF16 operands
        assert "FSET.BF.GT.FTZ.AND" in ops


@pytest.mark.parametrize("fn", list(KERNELS), ids=list(KERNELS.values()))
def test_tensor_cores_and_async_copies(lib, fn):
    ops = set(sass(lib, fn))
    assert {"HMMA.1688.F32.BF16"} <= ops and any(o.startswith("LDGSTS") for o in ops)
    assert any(o.startswith("LDSM") for o in ops)


def test_tuned_instantiations_keep_resource_targets(lib):
    r = resources(lib, "_Z13sgx_dx_kernelILi1ELi8ELi4ELi8ELb1ELi2ELi10ELi1EEv6SgArgs")
    assert r["reg"] <= 80 and r["local"] == 0 and r["stack"] == 0  # 3 CTAs / SM at 8 warps
    for fn in KERNELS:
        r = resources(lib, fn)
        assert r["local"] == 0, (fn, r)  # no spills in the production hot kernels
