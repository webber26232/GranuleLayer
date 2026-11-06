"""Shared pieces of the learned sliding-window reduction benchmarks: the packed sparse base, its cache, the common
provenance record.  Machine paths come from the command line / environment, never from this file."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import common  # noqa: F401  (sys.path)
import torch

D = G = 1024
INIT_SEED = 17  # historical K64 per-neuron init seed (k64_cases.INIT_SEED)
SHAPES = {"screen": 10 * 262_144, "primary": 40 * 262_144, "secondary": 48 * 262_144}


def base_path(root, H: int, graph_seed: int, init_seed: int = INIT_SEED, normalization: str = "energy") -> Path:
    return Path(root) / f"packed_H{H}_g{graph_seed}_i{init_seed}_{normalization}.pt"


def get_base(root, H: int, graph_seed: int, device="cuda", init_seed: int = INIT_SEED, normalization: str = "energy",
             dx_variant: str = "mt2_nw4_wg2"):
    """The Cover-TopK/16 sparse base: K64 / C256 random envelope (S64 / R256, per-neuron init), Cover-TopK selection,
    energy restoration; built once and reloaded from its packed file so every paired model starts identically."""
    from sparse_pooling import CoverTopK16Layer
    from sparse_pooling.experimental.kgen_layout import KLayout, init_parameters_by_neuron
    from sparse_pooling.supergroups import ConnectivityPlan

    p = base_path(root, H, graph_seed, init_seed, normalization)
    if not p.exists():
        L = KLayout(D, H, G, 64, 256)
        w, b = init_parameters_by_neuron(L, device, seed=init_seed)
        lay = CoverTopK16Layer.from_envelope(w, b, input_dim=D, hidden_dim=H, output_dim=G,
                                             connectivity=ConnectivityPlan(64, 256, seed=graph_seed),
                                             normalization=normalization, dx_variant=dx_variant)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        lay.save_packed(tmp)
        tmp.replace(p)
        del lay, w, b
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return CoverTopK16Layer.load_packed(p, device=device, dx_variant=dx_variant)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def env_record() -> dict:
    from provenance import hardware_state, provenance

    rec = dict(provenance=provenance(Path(common.__file__).resolve().parent.parent), hardware=hardware_state(),
               torch=torch.__version__, cuda_runtime=torch.version.cuda, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if torch.cuda.is_available():
        rec["gpu_uuid"] = getattr(torch.cuda.get_device_properties(0), "uuid", None)
        rec["gpu_name"] = torch.cuda.get_device_name(0)
    rec["provenance"].pop("source_diff", None)
    return rec
