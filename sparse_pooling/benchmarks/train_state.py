"""Complete training-state snapshot / in-place restore for controlled benchmarks.

FusedAdamW.state_dict() is not sufficient: the stochastic-rounding counter ``_seed`` is an
attribute, not optimizer state.  A ``TrainState`` captures, for a set of parameters and one
FusedAdamW:

* parameter data, FP32 ``main_grad`` and ``.grad`` (None or a tensor),
* per-parameter optimizer state (step, BF16/FP32 moments, FP32 master weights),
* param-group hyperparameters and the optimizer's ``_seed``,
* the torch CPU / CUDA RNG states.

``restore()`` copies back into the existing tensors (no allocation, so variants see the same
allocator state); a parameter whose ``.grad`` was None at capture gets ``.grad = None``.
Snapshot tensors live on the host (pinned) by default: they are benchmark overhead, never part
of the device memory a variant is charged with.  ``fingerprint()`` is an exact, order-free
checksum (integer sums of the raw bits), used to verify restorations outside timed regions.
"""
from __future__ import annotations

import copy

import torch


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


FP_CHUNK = 1 << 22  # elements per reduction: bounds the int64 temporaries (32 MiB) on the device


def tensor_fingerprint(t: torch.Tensor):
    """(sum of raw bits, sum of raw bits of every 3rd element) as exact Python ints.  Reduced in
    chunks: a whole-tensor sum(dtype=int64) materialises an int64 copy (6.4 GB for 800M BF16
    weights) that the caching allocator then keeps reserved."""
    b = _bits(t.detach().reshape(-1))
    s0 = torch.zeros((), dtype=torch.int64, device=b.device)
    s1 = torch.zeros((), dtype=torch.int64, device=b.device)
    for i in range(0, b.numel(), FP_CHUNK):  # FP_CHUNK % 3 == 1: track the stride-3 phase
        c = b[i:i + FP_CHUNK]
        s0 += c.sum(dtype=torch.int64)
        s1 += c[(1 - i) % 3::3].sum(dtype=torch.int64)
    return (int(s0), int(s1))


class TrainState:
    def __init__(self, params, opt, device: str = "cpu", extra: dict | None = None):
        self.params = list(params)
        self.opt = opt
        self.device = device
        self.extra = dict(extra or {})  # name -> tensor (e.g. data-order state), restored in place
        self._snap = None

    # tensors that make up the state, keyed by a stable name
    def _live(self):
        out = {}
        for i, p in enumerate(self.params):
            out[f"p{i}.data"] = p.data
            if getattr(p, "main_grad", None) is not None:
                out[f"p{i}.main_grad"] = p.main_grad
            if p.grad is not None:
                out[f"p{i}.grad"] = p.grad
            for k, v in self.opt.state.get(p, {}).items():
                if torch.is_tensor(v):
                    out[f"p{i}.opt.{k}"] = v
        for k, v in self.extra.items():
            out[f"extra.{k}"] = v
        return out

    def _scalars(self):
        steps = []
        for p in self.params:
            st = self.opt.state.get(p, {})
            steps.append({k: v for k, v in st.items() if not torch.is_tensor(v)})
        groups = [{k: copy.deepcopy(v) for k, v in g.items() if k != "params"} for g in self.opt.param_groups]
        return {"param_state": steps, "groups": groups, "seed": getattr(self.opt, "_seed", None),
                "grad_none": [p.grad is None for p in self.params]}

    def nbytes(self) -> int:
        return 0 if self._snap is None else sum(t.numel() * t.element_size() for t in self._snap["tensors"].values())

    @torch.no_grad()
    def capture(self):
        torch.cuda.synchronize()
        snap = self._snap["tensors"] if self._snap is not None else {}
        live = self._live()
        if self._snap is not None and set(snap) != set(live):
            snap = {}  # the set of state tensors changed (e.g. lazy optimizer state): re-allocate
        for k, v in live.items():
            if k not in snap:
                snap[k] = torch.empty(v.shape, dtype=v.dtype, device=self.device,
                                      pin_memory=(self.device == "cpu"))
            snap[k].copy_(v, non_blocking=True)
        torch.cuda.synchronize()
        self._snap = {"tensors": snap, "scalars": self._scalars(), "rng_cpu": torch.get_rng_state(),
                      "rng_cuda": torch.cuda.get_rng_state_all()}
        return self

    @torch.no_grad()
    def restore(self):
        assert self._snap is not None, "capture() first"
        live = self._live()
        sc = self._snap["scalars"]
        for i, p in enumerate(self.params):  # .grad that did not exist at capture time
            if sc["grad_none"][i]:
                p.grad = None
        live = self._live()
        missing = set(self._snap["tensors"]) - set(live)
        extra = set(live) - set(self._snap["tensors"])
        if missing or extra:
            raise RuntimeError(f"training state layout changed since capture: missing {sorted(missing)}, "
                               f"new {sorted(extra)}")
        for k, v in live.items():
            v.copy_(self._snap["tensors"][k], non_blocking=True)
        for p, st in zip(self.params, sc["param_state"]):
            self.opt.state[p].update(copy.deepcopy(st))
        for g, saved in zip(self.opt.param_groups, sc["groups"]):
            g.update(copy.deepcopy(saved))
        if sc["seed"] is not None:
            self.opt._seed = sc["seed"]
        torch.set_rng_state(self._snap["rng_cpu"])
        torch.cuda.set_rng_state_all(self._snap["rng_cuda"])
        torch.cuda.synchronize()

    @torch.no_grad()
    def fingerprint(self) -> dict:
        """Exact checksum of the live state (tensors + scalars)."""
        fp = {k: tensor_fingerprint(v) for k, v in sorted(self._live().items())}
        sc = self._scalars()
        fp["scalars"] = repr((sc["param_state"], sc["seed"], sc["groups"]))
        return fp
