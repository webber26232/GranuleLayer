"""EXPERIMENTAL generalized-fan-in input-supergroup layer (fan-in K in {8, 16, 32, 64}, logical C <= 256).

    y[b, g] = mean_{valid neurons i of g} act(z_i(b)),   z_i = bias_i + sum_{k < K} w_ik x[b, idx_r(i)[k]]

where the C neurons of logical block r share the record idx_r (Mode B).  Mean(ReLU) (default) or
Mean(identity) only; BF16 parameters / inputs, FP32 accumulation, ``plan.grad_precision`` "split"
(dZ as a BF16 hi + lo pair in the gradient products) or "bf16"; FP32-atomic (fast) reductions.
Any other request raises (backends.dispatch.kgen_reason).  The production layer is not involved.

Gradients go to FP32 ``main_grad`` when enabled (``enable_main_grad``), else are returned through
autograd in the parameter dtype; ``optim.FusedAdamW`` consumes them unchanged.
"""
from __future__ import annotations

import json
import math
from typing import Optional

import torch
from torch import nn

from ..config import LevelSpec, PhysicalPlan, resolve_reduction
from . import kgen_graph as KG
from .kgen_layout import KLayout, init_parameters


def _mark(name):
    from ..kernels import cuda_ops

    cuda_ops._mark(name)


class _Rt:
    """Per-device execution state (the ``_rt`` the benchmarks inspect): backend, work lists, records."""

    def __init__(self, layer: "KgenLayer"):
        from .kgen_runtime import kgen_backend

        dev = layer.indices.device
        self.backend = kgen_backend(dev)
        from ..backends.capabilities import query

        self.caps = query(dev)
        L = layer.layout
        self.cfg = self.backend.cfgs[L.fan_in]
        self.sg = KG.KgExec(L.blocks_per_group, L.block, L.fan_in, layer.counts, layer.sg_feat, layer.kg_buckets,
                            self.cfg)
        from ..backends.dispatch import kgen_dx_group_warps

        self.dx_group_warps = kgen_dx_group_warps(self.caps, L.input_dim, layer.sg_feat.shape[1], self.cfg)


class KgenFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, layer: "KgenLayer"):
        rt = layer._runtime()
        L = layer.layout
        B = x.shape[0]
        layer._check_call(B)
        xc = x.to(torch.bfloat16).contiguous()
        wc, bc = w.contiguous(), b.contiguous()
        _mark("fwd.begin")
        xt = xc.t().contiguous()
        _mark("fwd.layout")
        ysum = torch.zeros((L.groups, B), device=x.device, dtype=torch.float32)
        _mark("fwd.zero")
        a = layer._args(rt, xt, wc, bc, B)
        a.items, a.chunks, a.ysum = rt.sg.items.data_ptr(), rt.sg.chunks.data_ptr(), ysum.data_ptr()
        n_tiles = math.ceil(B / (16 * rt.cfg[0]))
        rt.backend.launch(L.fan_in, "fwd", a, layer.level.act_id, 0, rt.sg.chunks.shape[0], n_tiles, x.device)
        _mark("fwd.kernel")
        pre = (ysum / layer.counts_f[:, None]).t().contiguous()
        _mark("fwd.finalize")
        ctx.layer = layer
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(xc, w, b)
        layer.last_info.update(forward_backend="kgen", forward_kernel="kgen",
                               kgen_fwd=dict(ctas=rt.sg.chunks.shape[0] * n_tiles, token_tiles=n_tiles,
                                             chunks=rt.sg.chunks.shape[0], items=rt.sg.items.shape[0],
                                             tokens=16 * rt.cfg[0], warps=rt.cfg[1]))
        return pre

    @staticmethod
    def backward(ctx, dy):
        layer: KgenLayer = ctx.layer
        xc, w, b = ctx.saved_tensors
        rt = layer._runtime()
        L = layer.layout
        B, D = xc.shape
        _mark("bwd.begin")
        c1T = (dy.float() / layer.counts_f[None, :]).t().contiguous()  # Mean: dL/dpre / n_g, [G, B]
        _mark("bwd.coef")
        xt = xc.t().contiguous()
        _mark("bwd.layout")
        dxt = torch.zeros((D, B), device=xc.device, dtype=torch.float32)
        _mark("bwd.dx_init")
        mg_w, mg_b = getattr(w, "main_grad", None), getattr(b, "main_grad", None)
        use_main = mg_w is not None and mg_b is not None
        dw = mg_w if use_main else torch.zeros(w.shape, device=w.device, dtype=torch.float32)
        db = mg_b if use_main else torch.zeros(b.shape, device=b.device, dtype=torch.float32)
        a = layer._args(rt, xt, w.contiguous(), b.contiguous(), B)
        a.c1T, a.dXT, a.dW, a.dB = c1T.data_ptr(), dxt.data_ptr(), dw.data_ptr(), db.data_ptr()
        a.items, a.chunks, a.dx_group_warps = rt.sg.items.data_ptr(), rt.sg.chunks.data_ptr(), rt.dx_group_warps
        split = int(layer.plan.grad_precision == "split")
        n_tiles = math.ceil(B / (16 * rt.cfg[2]))
        rt.backend.launch(L.fan_in, "dx", a, layer.level.act_id, split, rt.sg.chunks.shape[0], n_tiles, xc.device)
        _mark("bwd.dx")
        a.tasks, a.chunks = rt.sg.tasks.data_ptr(), rt.sg.dw_chunks.data_ptr()
        rt.backend.launch(L.fan_in, "dw", a, layer.level.act_id, split, rt.sg.dw_chunks.shape[0], 0, xc.device)
        _mark("bwd.dw")
        layer.last_info.update(backward_backend="kgen", dx_kernel="kgen", dw_kernel="kgen",
                               grad_precision=layer.plan.grad_precision,
                               kgen_dx=dict(ctas=rt.sg.chunks.shape[0] * n_tiles, tokens=16 * rt.cfg[2],
                                            warps=rt.cfg[3], group_warps=rt.dx_group_warps,
                                            reduction="fp32_atomics" if rt.dx_group_warps == 1 else
                                            "shared_cas+fp32_atomics"),
                               kgen_dw=dict(ctas=rt.sg.dw_chunks.shape[0], tasks=rt.sg.tasks.shape[0],
                                            tasks_per_warp=rt.cfg[4], warps=rt.cfg[5]))
        dx = dxt.t().to(ctx.x_dtype, memory_format=torch.contiguous_format)  # (as the production layer)
        gw = None if use_main else dw.to(w.dtype)
        gb = None if use_main else db.to(b.dtype)
        return dx, gw, gb, None


class KgenLayer(nn.Module):
    """EXPERIMENTAL: see the module docstring.  ``graph=(idx [G, NB, K] int16, feat [Q, S] int16,
    block buckets [n, 4] int32, GraphSpec)`` uses a given logical graph (e.g. a native K = 8 graph,
    for the matched backend control); otherwise the generalized generator builds it from
    ``connectivity`` (quantum = 1 logical record per assignment unit)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, fan_in: int, shared_neurons: int,
                 connectivity, pooling: str = "mean", dtype=torch.bfloat16, device=None, seed: int = 0,
                 bias_std: float = 0.0, plan: Optional[PhysicalPlan] = None, graph=None, quantum: int = 1,
                 weights=None):
        super().__init__()
        if dtype != torch.bfloat16:
            raise ValueError("kgen: BF16 parameters only")
        self.level = LevelSpec(type=pooling) if isinstance(pooling, str) else LevelSpec(**pooling)
        self.layout = L = KLayout(input_dim, hidden_dim, output_dim, fan_in, shared_neurons)
        self.plan = plan or PhysicalPlan()
        self.connectivity = connectivity
        device = torch.device(device or "cuda")
        counts = L.counts_tensor(device)
        if graph is None:
            idx, feat, ubk, st = KG.generate(input_dim, counts, L.block, fan_in, L.blocks_per_group, quantum,
                                             connectivity, device)
            bk = KG.unit_to_block_buckets(ubk, counts, L.block, quantum)
            spec = KG.GraphSpec(fan_in, L.block, connectivity.input_supergroup_dim, connectivity.partition_rounds,
                                connectivity.seed, quantum, connectivity.generation)
            self.graph_stats = dict(st, generator=KG.GENERATOR, generator_version=KG.GENERATOR_VERSION,
                                    quantum=quantum)
        else:
            idx, feat, bk, spec = graph
            self.graph_stats = dict(source=spec.source, quantum=spec.quantum)
        if weights is None:
            w, b = init_parameters(L, device, dtype, seed=seed, bias_std=bias_std)
        else:
            w, b = weights
        self.register_buffer("indices", idx.to(device).contiguous())
        self.register_buffer("sg_feat", feat.to(device).contiguous())
        self.register_buffer("kg_buckets", bk.to(device).contiguous())
        self.register_buffer("kg_spec", torch.frombuffer(bytearray(spec.to_json().encode()), dtype=torch.uint8).clone())
        self.spec = spec
        self.weight = nn.Parameter(w.to(device).contiguous())
        self.bias = nn.Parameter(b.to(device).contiguous())
        self.counts = counts.contiguous()
        self.counts_f = counts.to(torch.float32)
        self._rt = None
        self._rt_key = None
        self.last_info = {}

    # compatibility with the benchmark tools
    @property
    def sg_buckets(self):
        return self.kg_buckets

    def _runtime(self) -> _Rt:
        key = (self.indices.device, self.indices.data_ptr(), self.kg_buckets.data_ptr())
        if self._rt is None or self._rt_key != key:
            self._rt = _Rt(self)
            self._rt_key = key
            self.last_info["kgen_work_lists"] = dict(self._rt.sg.stats)
        return self._rt

    def _check_call(self, B: int):
        from ..backends.dispatch import kgen_reason

        mode, why, strict = resolve_reduction(self.plan)
        rt = self._rt
        r = kgen_reason(rt.caps, self.layout.fan_in, self.layout.block, self.sg_feat.shape[1], self.layout.input_dim,
                        B, self.level, self.plan, mode == "deterministic", rt.cfg)
        if r:
            raise ValueError(f"experimental kgen layer cannot run this call: {r}")
        self.last_info["reduction"] = {"mode": "fast", "reason": why}
        ex = self.last_info.get("execution")
        if ex is None or ex.get("batch") != B or ex.get("grad_precision") != self.plan.grad_precision:
            self.last_info["execution"] = dict(self.explain(B), grad_precision=self.plan.grad_precision)

    def _args(self, rt, xt, w, b, B):
        from .kgen_runtime import KgArgs

        L = self.layout
        return KgArgs(xt.data_ptr(), xt.stride(0), self.indices.data_ptr(), w.data_ptr(), b.data_ptr(),
                      self.counts.data_ptr(), self.sg_feat.data_ptr(), 0, 0, 0, 0, 0, 0, 0, 0,
                      B, L.groups, L.input_dim, self.sg_feat.shape[1], L.blocks_per_group, L.block, 1, 0)

    def explain(self, B: int = 8192) -> dict:
        rt = self._runtime()
        return dict(backend="kgen (experimental)", status="experimental",
                    device=dict(name=rt.caps.name, arch=rt.caps.arch, sm_count=rt.caps.sm_count,
                                smem_per_block_optin=rt.caps.smem_per_block_optin),
                    build=rt.backend.build_info(), tiling=rt.backend.cfg_dict(self.layout.fan_in),
                    dx_group_warps=rt.dx_group_warps, profile="none (one compiled tiling per K; no tuning profile)",
                    numerics=self.numerics(), batch=B, work_lists=dict(rt.sg.stats))

    def numerics(self) -> str:
        p = "dZ as BF16 hi + lo pair" if self.plan.grad_precision == "split" else "dZ as one BF16"
        return (f"BF16 x / W / bias operands, FP32 accumulation over all K = {self.layout.fan_in} products before one "
                f"activation; backward recomputes z; {p} in dX / dW products (FP32 accumulate); FP32 atomics for "
                "cross-CTA sums (summation order differs from the production kernels)")

    def enable_main_grad(self):
        for p in (self.weight, self.bias):
            if getattr(p, "main_grad", None) is None:
                p.main_grad = torch.zeros(p.shape, device=p.device, dtype=torch.float32)

    def zero_main_grad(self):
        for p in (self.weight, self.bias):
            if getattr(p, "main_grad", None) is not None:
                p.main_grad.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        y = KgenFunction.apply(x.reshape(-1, x.shape[-1]), self.weight, self.bias, self)
        return y.reshape(*lead, y.shape[-1])

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        sp = state_dict.get(prefix + "kg_spec")
        if sp is not None:
            got = KG.GraphSpec.from_json(bytes(sp.cpu().tolist()).decode())
            if got != self.spec:
                raise ValueError(f"checkpoint graph spec {got} != layer spec {self.spec}")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._rt = None

    # ---- kcsr sweep: logical record table, validation, description
    def kcsr_table(self):
        L = self.layout
        return KG.record_table(L.input_dim, L.hidden_dim, self.counts, L.block, L.fan_in, self.indices, self.sg_feat,
                               self.kg_buckets)

    def kcsr_validate(self):
        L = self.layout
        return KG.validate(L.input_dim, self.counts, L.block, L.fan_in, self.indices, self.sg_feat, self.kg_buckets,
                           self.spec.R)

    def kcsr_describe(self):
        L = self.layout
        rt = self._runtime()
        return dict(backend="kgen", status="experimental", logical_K=L.fan_in, logical_C=L.block, physical_C=L.block,
                    physical_subblocks=1, index_duplication=1, NB=L.blocks_per_group, spec=json.loads(self.spec.to_json()),
                    padding_fraction=L.padding_fraction(), param_bytes=L.param_bytes(),
                    tiling=rt.backend.cfg_dict(L.fan_in), dx_group_warps=rt.dx_group_warps, numerics=self.numerics(),
                    build=rt.backend.build_info())

    def extra_repr(self):
        L = self.layout
        return (f"EXPERIMENTAL kgen: in={L.input_dim}, hidden={L.hidden_dim:,}, out={L.groups}, fan_in={L.fan_in}, "
                f"C={L.block}, S={self.spec.S}, R={self.spec.R}, {self.level.describe()}")


@torch.no_grad()
def from_native(lay, plan: Optional[PhysicalPlan] = None) -> KgenLayer:
    """The matched-backend control: a KgenLayer (K = 8) with EXACTLY the graph, weights and bias of a
    production input-supergroup layer (canonical records / parameters; the production buckets of
    fragment-packed units converted to logical-block ranges, padded blocks of a partially valid unit
    excluded).  Both layers compute the same function; the kernels and summation orders differ."""
    L = lay.layout
    if not (L.shared_indices and L.fan_in == 8) or lay.connectivity is None:
        raise ValueError("from_native needs a Mode B input-supergroup layer")
    idx, w, b = L.to_canonical(lay.indices, lay.weight.detach(), lay.bias.detach())
    KL = KLayout(L.input_dim, L.hidden_dim, L.groups, 8, L.block)
    nb = KL.blocks_per_group
    counts = KL.counts_tensor(idx.device)
    blk = KG.unit_to_block_buckets(lay.sg_buckets, counts, L.block, L.unit_blocks)
    c = lay.connectivity
    spec = KG.GraphSpec(8, L.block, c.input_supergroup_dim, c.partition_rounds, c.seed, L.unit_blocks, c.generation,
                        generator="production supergroups.generate_indices", generator_version=c.version,
                        source="converted-from-native")
    out = KgenLayer(L.input_dim, L.hidden_dim, L.groups, 8, L.block, c, pooling=lay.cfg.levels[0].type,
                    device=idx.device, plan=plan or PhysicalPlan(grad_precision=lay.plan.grad_precision),
                    graph=(idx[:, :nb].contiguous(), lay.sg_feat, blk, spec),
                    weights=(w[:, :nb].contiguous(), b[:, :nb].contiguous()))
    out.graph_stats = dict(lay.graph_stats or {}, converted_from_native=True, native_unit_blocks=L.unit_blocks)
    return out
