import sys; sys.path.insert(0, "/tmp/claude-1000/-home-evga-Projects-pamachine-GranuleLayer/4fd262c0-821a-4403-ab57-b25da7c8e4c5/scratchpad/fresh/src")  # the fresh clone's package
"""Selected profile, device resources, build target and backend / cache identity on this GPU, plus
mocked selections for devices that are not present (control flow only)."""
import json, os, sys
import torch
from sparse_pooling import PhysicalPlan, SparsePoolingLayer
from sparse_pooling.supergroups import ConnectivityPlan
from sparse_pooling.backends import capabilities as C
from sparse_pooling.backends.cuda import build as B
from sparse_pooling.backends.profiles import select

out = {}
caps = C.query(torch.device("cuda", 0))
out["device"] = {k: v for k, v in caps.summary().items() if k != "uuid"}
tc = C.toolchain()
out["toolchain"] = dict(nvcc=tc.nvcc, version=tc.version, real_archs=tc.real_archs, host_compiler=tc.host_compiler)
out["resolve_target"] = B.resolve_target(caps, tc)
out["cache_root"] = str(B.cache_root())
for prof in ("auto", "generic_cuda"):
    lay = SparsePoolingLayer(1024, 1_000_000, 1024, shared_neurons=8, pooling="mean", device="cuda", seed=0,
                             plan=PhysicalPlan(tuning_profile=prof, cuda_input_supergroup_backend="auto"),
                             connectivity=ConnectivityPlan(64, 16, seed=0))
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    lay(x).float().square().mean().backward()
    ex = lay.last_info["execution"]
    out[f"execution_{prof}"] = dict(profile=ex["profile"], build=ex["build"],
                                    device={k: v for k, v in ex["device"].items() if k != "uuid"},
                                    kernels={k: lay.last_info.get(k) for k in ("forward_kernel", "dx_kernel", "dw_kernel")})
# mocked devices (synthetic records: never hardware evidence)
mock = {}
for name, cc, sms, smem in [("NVIDIA GeForce RTX 3080", (8, 6), 68, 99 * 1024), ("NVIDIA GeForce RTX 3090", (8, 6), 82, 99 * 1024),
                            ("unknown sm_89 board", (8, 9), 128, 99 * 1024), ("unknown sm_120 board", (12, 0), 84, 99 * 1024)]:
    s = C.synthetic(name, cc, sm_count=sms, smem_per_block_optin=smem)
    mock[name] = dict(auto=select(s).summary(), forced_rtx3090=select(s, "rtx3090").summary()["status"])
out["mocked_selection"] = mock
json.dump(out, sys.stdout, indent=1, default=str)
