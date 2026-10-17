"""Benchmark utilities: CUDA-event timing, memory accounting, JSON results."""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def gpu_info() -> Dict:
    p = torch.cuda.get_device_properties(0)
    return {"name": p.name, "sm": f"{p.major}.{p.minor}", "sms": p.multi_processor_count,
            "memory_GiB": round(p.total_memory / 2**30, 2), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda}


def time_cuda(fn: Callable, warmup: int = 1, iters: int = 5, flush_l2: bool = False) -> Dict[str, float]:
    """Median / min / IQR of fn() in ms, measured with CUDA events.
    flush_l2=True writes a 64 MiB buffer before each iteration (cold cache)."""
    flush = torch.empty(64 * 2**20, dtype=torch.uint8, device="cuda") if flush_l2 else None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts: List[float] = []
    for _ in range(iters):
        if flush is not None:
            flush.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    q1 = ts[len(ts) // 4]
    q3 = ts[(3 * len(ts)) // 4]
    return {"median_ms": statistics.median(ts), "min_ms": ts[0], "iqr_ms": q3 - q1, "n": len(ts)}


class KernelCounter:
    """Counts CUDA kernel launches inside a region with the PyTorch profiler."""

    def __enter__(self):
        from torch.profiler import ProfilerActivity, profile

        self.prof = profile(activities=[ProfilerActivity.CUDA])
        self.prof.__enter__()
        return self

    def __exit__(self, *a):
        torch.cuda.synchronize()
        self.prof.__exit__(*a)
        evs = [e for e in self.prof.events() if e.device_type.name == "CUDA"]
        self.launches = len(evs)
        self.names = sorted({e.name for e in evs})


def save(name: str, payload: Dict):
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = dict(payload, gpu=gpu_info(), timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    path = RESULTS / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str))
    return path


def useful_tflops(B: int, H: int, ms: float, passes: float = 1.0) -> float:
    """2 * B * H * 8 FLOPs of useful sparse-projection work per pass."""
    return passes * 2 * B * H * 8 / (ms * 1e-3) / 1e12


def fmt_gib(n: float) -> str:
    return f"{n / 2**30:.2f} GiB"
