"""The named Cover-TopK/16 layer (sparse_pooling.cover_topk16): support semantics, budgets, identity, the function
against the FP64 reference of the masked K64 envelope (forward / dX / dW / db), token / neuron tails, zero values and
near-ReLU-boundary inputs, batch x time inputs, the absence of an M x H intermediate, FusedAdamW with FP32 main_grad
(updates, moments, zeroing, accumulation, restore), the one-time energy normalization, packed save / load without
reselection, the research checkpoint format, device moves / reload invalidating runtime caches, CPU-only import, and
the documented quickstart.  Full-size (H 12.5M) identity: benchmarks/cover_topk16_identity.py."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from conftest import requires_cuda_backend

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sparse_pooling" / "benchmarks"))

D, H, G, C = 256, 8 * 431 + 4, 8, 256  # ragged H: tail records of 175 / 176 neurons (partial 16-neuron groups)


def envelope_values(H=H, D=D, G=G, C=C, seed=0, bias_scale=0.3, device="cuda"):
    from sparse_pooling.experimental.kgen_layout import KLayout, zero_padding_

    L = KLayout(D, H, G, 64, C)
    g = torch.Generator(device=device).manual_seed(seed)
    w = (torch.randn(L.weight_shape, device=device, generator=g) / 8).bfloat16()
    b = (bias_scale * torch.randn(L.bias_shape, device=device, generator=g)).bfloat16()
    zero_padding_(L, w, b)
    return w, b


def conn(seed=1, R=2):
    from sparse_pooling.supergroups import ConnectivityPlan

    return ConnectivityPlan(64, R, seed=seed)


def build(normalization="none", dx_variant="auto", device="cuda", H=H, D=D, bias_scale=0.3, w=None, b=None):
    from sparse_pooling import CoverTopK16Layer

    if w is None:
        w, b = envelope_values(H=H, D=D, bias_scale=bias_scale, device=device)
    lay = CoverTopK16Layer.from_envelope(w, b, input_dim=D, hidden_dim=H, output_dim=G, connectivity=conn(),
                                         normalization=normalization, dx_variant=dx_variant, device=device)
    return lay, w, b


def masked_reference(w, b, normalization="none", H=H, D=D):
    """(envelope with indices, mask, the masked / normalized K64 values) computed independently of the layer."""
    from sparse_pooling.experimental import local_supports as LS
    from sparse_pooling.experimental import row_normalize as RN
    from sparse_pooling.experimental.kgen_lowered import Envelope

    env = Envelope.generate(D, H, G, C, conn(), w, b, device=w.device, keep_indices=True)
    sup = LS.build("Cover-TopK", 16, env.layout, env.kg_buckets, w=w, graph_seed=1, device=w.device,
                   Q=env.sg_feat.shape[0])
    em = sup.element_mask(env.layout)
    if normalization == "energy":
        wm, bm, _ = RN.apply(w, b, em, env.layout.valid_mask(w.device), "energy")
    else:
        wm, bm = w * em, b
    return env, em, wm, bm, sup


def maxrel(a, r):
    a, r = a.double(), r.double()
    return float((a - r).abs().max() / (r.abs().max() + 1e-300))


def run(lay, x, dy, variant=None):
    if variant is not None:
        lay.dx_variant = variant
    lay.enable_main_grad()
    lay.zero_main_grad()
    xx = x.clone().requires_grad_(True)
    y = lay(xx)
    y.backward(dy)
    return y.detach(), xx.grad.detach(), lay.weight.main_grad.clone(), lay.bias.main_grad.clone()


def canonical_grad_rows(lay, dw, db):
    """[valid neurons, 64] / [valid neurons] canonical rows of compact gradients (scattered into the envelope)."""
    from sparse_pooling.experimental import canonical as Cn

    g, n = Cn.neurons(lay.counts, 0, lay.counts.numel())
    return lay.canon_rows({"weight": dw, "bias": db}, g, n)


def envelope_rows(L, t, tb):
    from sparse_pooling.experimental import canonical as Cn

    counts = L.counts_tensor(t.device)
    g, n = Cn.neurons(counts, 0, counts.numel())
    s = g * (L.blocks_per_group * L.block) + n
    return t.reshape(-1, 64)[s], tb.reshape(-1)[s]


# ---------------------------------------------------------------------------------------------- support semantics
@requires_cuda_backend
def test_mask_structure_budget_coverage_and_identity():
    from sparse_pooling.experimental import canonical as Cn

    lay, w, b = build()
    env, em, wm, bm, sup = masked_reference(w, b)
    m = lay.metadata
    assert m["envelope"]["K"] == 64 and m["envelope"]["C"] == 256 and m["envelope"]["S"] == 64
    assert m["executed"] == dict(K=16, C=16, records_16=lay.graph_stats["records_16"])
    assert m["valid_slots"] == 16 * H and m["biases"] == H and m["policy"] == "Cover-TopK"
    assert m["support_fingerprint"] == sup.fingerprint()
    assert lay.layout.fan_in == 16 and lay.layout.block == 16 and lay.envelope_KC == (64, 256)
    # exact 16 sorted distinct inputs per valid 16-neuron record; sorted global ids paired with the values
    cnt = lay.counts.long()
    nb16 = lay.layout.blocks_per_group
    live = (cnt[:, None] > 16 * torch.arange(nb16, device=cnt.device)[None, :]).reshape(-1)
    envm = lay.env.long()[live]
    assert bool((envm[:, 1:] > envm[:, :-1]).all())
    idx = lay.indices.long().reshape(-1, 16)[live]
    assert bool((idx[:, 1:] > idx[:, :-1]).all())
    # every valid neuron: 16 enabled slots; union coverage of all 64 candidates per original record
    rows = Cn.mask_digest(lay)
    assert rows["enabled_slots"] == 16 * H
    per_neuron = em.reshape(-1, 64).sum(1)[env.layout.valid_mask(em.device).reshape(-1)]
    assert bool((per_neuron == 16).all())
    rec_cov = em.view(G, env.layout.blocks_per_group, C, 64).amax(2)  # [G, NB, 64] union over the record's neurons
    valid_rec = env.layout.valid_mask(em.device).any(2)
    assert bool(rec_cov[valid_rec].all())
    # coverage is a union constraint, not equal-frequency quotas: multiplicities differ inside records
    mult = em.view(G, env.layout.blocks_per_group, C // 16, 16, 64)[:, :, :, 0, :].sum(2)
    assert int(mult[valid_rec].max()) > int(mult[valid_rec].min())
    # identical canonical function / mask as the dense-masked envelope; padded slots zero
    env.weight, env.bias, env.mask = wm, bm, em
    assert Cn.digest(lay)["sha"] == Cn.digest(env)["sha"]
    assert Cn.mask_digest(lay)["sha"] == Cn.mask_digest(env)["sha"]
    valid = lay.layout.valid_mask(lay.weight.device)
    assert int((lay.weight.detach()[~valid] != 0).sum()) == 0 and int((lay.bias.detach()[~valid] != 0).sum()) == 0
    assert lay.counts.dtype == torch.int32 and torch.equal(lay.counts.cpu(), env.layout.counts_tensor("cpu"))


@requires_cuda_backend
def test_infeasible_geometry_raises_clearly():
    from sparse_pooling import CoverTopK16Layer

    Hbad = G * (256 + 40)  # every output's last record has 40 neurons = 3 row groups: 3 x 16 < 64
    w, b = envelope_values(H=Hbad)
    with pytest.raises(ValueError, match="infeasible"):
        CoverTopK16Layer.from_envelope(w, b, input_dim=D, hidden_dim=Hbad, output_dim=G, connectivity=conn())
    with pytest.raises(ValueError, match="S = 64"):
        from sparse_pooling.supergroups import ConnectivityPlan

        w, b = envelope_values()
        CoverTopK16Layer.from_envelope(w, b, input_dim=D, hidden_dim=H, output_dim=G,
                                       connectivity=ConnectivityPlan(32, 2))
    with pytest.raises(ValueError, match="normalization"):
        CoverTopK16Layer.from_envelope(w, b, input_dim=D, hidden_dim=H, output_dim=G, connectivity=conn(),
                                       normalization="fanin")
    with pytest.raises(ValueError, match="KLayout"):
        CoverTopK16Layer.from_envelope(w[:, :1], b[:, :1], input_dim=D, hidden_dim=H, output_dim=G,
                                       connectivity=conn())


# ---------------------------------------------------------------------------------------------- function
@requires_cuda_backend
@pytest.mark.parametrize("normalization", ["none", "energy"])
def test_packed_function_equals_fp64_reference(normalization):
    from sparse_pooling.experimental import reference as REF

    lay, w, b = build(normalization)
    env, em, wm, bm, _ = masked_reference(w, b, normalization)
    L = env.layout
    for B in (8, 24, 40, 64, 512):
        g = torch.Generator(device="cuda").manual_seed(B)
        x = torch.randn(B, D, device="cuda", generator=g).bfloat16().float()
        dy = torch.randn(B, G, device="cuda", generator=g)
        yr, dxr, dwr, dbr = REF.grads(x.double(), env.indices.long(), wm.double(), bm.double(), L.counts_tensor("cuda"),
                                      dy.double())
        rw, rb = envelope_rows(L, dwr * em, dbr)
        for v in ("legacy", "auto", "mt4_nw8_wg4"):
            y, dx, dw, db = run(lay, x, dy, v)
            assert maxrel(y, yr) < 2e-6, (v, B)
            assert maxrel(dx, dxr) < 2e-5, (v, B)
            cw, cb = canonical_grad_rows(lay, dw, db)
            assert maxrel(cw, rw) < 2e-5 and maxrel(cb, rb) < 2e-5, (v, B)


@requires_cuda_backend
def test_zero_values_and_near_relu_boundary():
    from sparse_pooling.experimental import reference as REF

    lay, w, b = build()
    x = torch.zeros(40, D, device="cuda")
    dy = torch.randn(40, G, device="cuda")
    for v in ("legacy", "mt4_nw8_wg4"):
        _, dx, _, _ = run(lay, x, dy, v)
        assert float(dx.abs().max()) >= 0.0 and torch.isfinite(dx).all()
    zw = torch.zeros_like(w)
    lay0, _, _ = build(w=zw, b=b)
    _, dx, dw, _ = run(lay0, torch.randn(40, D, device="cuda").bfloat16().float(), dy, "mt4_nw8_wg4")
    assert float(dx.abs().max()) == 0.0
    # near-boundary pre-activations (tiny biases): differences only through flipped near-zero decisions
    lay, w, b = build(bias_scale=0.01)
    env, em, wm, bm, _ = masked_reference(w, b)
    g = torch.Generator(device="cuda").manual_seed(9)
    x = torch.randn(64, D, device="cuda", generator=g).bfloat16().float()
    dy = torch.randn(64, G, device="cuda", generator=g)
    yr, dxr, _, _ = REF.grads(x.double(), env.indices.long(), wm.double(), bm.double(),
                              env.layout.counts_tensor("cuda"), dy.double())
    for v in ("legacy", "mt4_nw8_wg4"):
        y, dx, _, _ = run(lay, x, dy, v)
        assert maxrel(y, yr) < 1e-5 and maxrel(dx, dxr) < 5e-5, v


@requires_cuda_backend
def test_batch_time_inputs_and_shapes():
    lay, _, _ = build()
    g = torch.Generator(device="cuda").manual_seed(3)
    x3 = torch.randn(2, 20, D, device="cuda", generator=g).bfloat16()
    dy3 = torch.randn(2, 20, G, device="cuda", generator=g)
    xx = x3.clone().requires_grad_(True)
    y3 = lay(xx)
    assert y3.shape == (2, 20, G)
    y3.backward(dy3)
    assert xx.grad.shape == x3.shape and xx.grad.dtype == torch.bfloat16
    y2, dx2, _, _ = run(lay, x3.reshape(40, D), dy3.reshape(40, G))
    assert torch.allclose(y3.detach().reshape(40, G).float(), y2.float(), rtol=1e-5, atol=1e-6)
    assert torch.allclose(xx.grad.reshape(40, D).float(), dx2.float(), rtol=2e-2, atol=1e-6)
    with pytest.raises(ValueError, match="B % 8"):
        lay(torch.randn(3, 5, D, device="cuda").bfloat16())


@requires_cuda_backend
def test_no_m_by_h_intermediate():
    Hbig = 8 * 12_500  # 100k neurons: an M x H BF16 tensor at M 1024 would be 195 MiB
    lay, _, _ = build(H=Hbig)
    lay.enable_main_grad()
    B = 1024
    x = torch.randn(B, D, device="cuda").bfloat16().requires_grad_(True)
    dy = torch.randn(B, G, device="cuda")
    lay(x).backward(dy)  # warm-up: runtime / work lists
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    lay(x).backward(dy)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base
    assert extra < B * Hbig * 2 // 8, extra  # far below one BF16 M x H tensor


# ---------------------------------------------------------------------------------------------- optimizer
@requires_cuda_backend
def test_fused_adamw_main_grad_moments_accumulation_and_restore():
    from sparse_pooling.experimental import canonical as Cn
    from sparse_pooling.optim import FusedAdamW
    from train_state import TrainState

    lay, _, _ = build(normalization="energy")
    lay.enable_main_grad()
    opt = FusedAdamW([lay.weight, lay.bias], lr=1e-3, moments="bf16", stochastic_rounding=True)
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(64, D, device="cuda", generator=g).bfloat16()
    dy = torch.randn(64, G, device="cuda", generator=g)
    # main_grad accumulates over two backward passes (FP32), equals 2 x one pass
    lay(x.clone().requires_grad_(True)).backward(dy)
    one = lay.weight.main_grad.clone()
    lay(x.clone().requires_grad_(True)).backward(dy)
    assert torch.allclose(lay.weight.main_grad, 2 * one, rtol=1e-6, atol=1e-7)
    assert lay.weight.main_grad.dtype == torch.float32 and lay.weight.grad is None
    opt.step()
    assert float(lay.weight.main_grad.abs().max()) == 0.0 and float(lay.bias.main_grad.abs().max()) == 0.0  # fused zero
    st = opt.state[lay.weight]
    assert st["m"].dtype == torch.bfloat16 and st["m"].shape == lay.weight.shape and st["step"] == 1
    valid = lay.layout.valid_mask(lay.weight.device)
    assert int((lay.weight.detach()[~valid] != 0).sum()) == 0  # padding never trained
    ts = TrainState([lay.weight, lay.bias], opt).capture()
    fp0, ident = ts.fingerprint(), Cn.identity(lay, opt)
    for _ in range(2):
        lay(x.clone().requires_grad_(True)).backward(dy)
        opt.step()
    assert ts.fingerprint() != fp0
    ts.restore()
    assert ts.fingerprint() == fp0 and Cn.identity(lay, opt) == ident


@requires_cuda_backend
def test_legacy_and_wg4_give_identical_parameter_trajectories_sr_off():
    """Only the dX launch differs; the layer's input is not trained here, so dW / db and the FusedAdamW updates are
    bitwise identical between the legacy launch and the WG4 tiling."""
    from sparse_pooling.optim import FusedAdamW

    out = {}
    for v in ("legacy", "mt4_nw8_wg4"):
        lay, _, _ = build(dx_variant=v)
        lay.enable_main_grad()
        opt = FusedAdamW([lay.weight, lay.bias], lr=1e-3, stochastic_rounding=False)
        g = torch.Generator(device="cuda").manual_seed(0)
        x = torch.randn(64, D, device="cuda", generator=g).bfloat16()
        dy = torch.randn(64, G, device="cuda", generator=g)
        for _ in range(3):
            lay(x.clone().requires_grad_(True)).backward(dy)
            opt.step()
        out[v] = (lay.weight.detach().clone(), opt.state[lay.weight]["m"].clone(), opt.state[lay.weight]["v"].clone(),
                  lay.last_info["kgen_dx"]["variant"]["name"])
    assert out["legacy"][3] == "legacy" and out["mt4_nw8_wg4"][3] == "mt4_nw8_wg4"
    for a, b in zip(out["legacy"][:3], out["mt4_nw8_wg4"][:3]):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------------------------- normalization
@requires_cuda_backend
def test_energy_normalization_applied_once_and_recorded():
    from sparse_pooling import CoverTopK16Layer
    from sparse_pooling.experimental import canonical as Cn
    from sparse_pooling.experimental.kgen_lowered import Envelope

    lay, w, b = build(normalization="energy")
    env, em, wm, bm, _ = masked_reference(w, b, "energy")
    n = lay.normalization
    assert n["treatment"] == "energy" and n["applied"] and n["s_max"] == 8.0 and n["eps_rel"] == 1e-6
    assert n["scale_bias"] is False and lay.metadata["normalization_scale_bias"] is False
    env.weight, env.bias, env.mask = wm, bm, em
    assert Cn.digest(lay)["sha"] == Cn.digest(env)["sha"]
    assert torch.equal(bm, b)  # bias untouched
    none, _, _ = build("none", w=w, b=b)
    assert none.normalization == dict(treatment="none", applied=False)
    assert not torch.equal(none.weight, lay.weight)
    # an envelope already marked normalized is refused
    env2 = Envelope.generate(D, H, G, C, conn(), w, b, device="cuda")
    env2.graph_stats["normalized"] = True
    with pytest.raises(ValueError, match="already energy-normalized"):
        CoverTopK16Layer.from_envelope(w, b, input_dim=D, hidden_dim=H, output_dim=G, connectivity=conn(),
                                       normalization="energy", envelope=env2)


# ---------------------------------------------------------------------------------------------- persistence
@requires_cuda_backend
def test_save_load_round_trip_without_reselection(tmp_path, monkeypatch):
    from sparse_pooling import CoverTopK16Layer
    from sparse_pooling.experimental import canonical as Cn
    from sparse_pooling.experimental import local_supports as LS

    lay, _, _ = build(normalization="energy")
    x = torch.randn(40, D, device="cuda").bfloat16().float()
    dy = torch.randn(40, G, device="cuda")
    y0, dx0, dw0, db0 = run(lay, x, dy, "legacy")
    ident = Cn.identity(lay, state=False)
    rec = lay.save_packed(tmp_path / "c.pt")
    del lay

    def boom(*a, **k):
        raise AssertionError("support selection must not run on load")

    monkeypatch.setattr(LS, "build", boom)
    lay2 = CoverTopK16Layer.load_packed(tmp_path / "c.pt", device="cuda")
    assert Cn.identity(lay2, state=False) == ident and lay2.immutable_fingerprints() == rec["fingerprints"]
    assert lay2.normalization["treatment"] == "energy" and lay2.dx_variant == "auto"
    assert lay2.metadata["envelope"]["C"] == 256 and lay2.envelope_KC == (64, 256)
    y, dx, dw, db = run(lay2, x, dy, "legacy")
    assert maxrel(y, y0) < 1e-6 and maxrel(dx, dx0) < 1e-6 and torch.equal(dw, dw0) and torch.equal(db, db0)
    # a tampered file is refused
    blob = torch.load(tmp_path / "c.pt", weights_only=True)
    blob["state"]["indices"] = blob["state"]["indices"].flip(-1).contiguous()
    torch.save(blob, tmp_path / "bad.pt")
    with pytest.raises((ValueError, RuntimeError)):
        CoverTopK16Layer.load_packed(tmp_path / "bad.pt", device="cuda")


@requires_cuda_backend
def test_research_checkpoint_format(tmp_path):
    from sparse_pooling import CoverTopK16Layer
    from sparse_pooling.cover_topk16 import CKPT_SCHEMA, tensor_sha256
    from sparse_pooling.experimental import canonical as Cn
    from sparse_pooling.experimental.kgen_lowered import Envelope

    w, b = envelope_values()
    env = Envelope.generate(D, H, G, C, conn(), w, b, device="cuda")
    rec = dict(schema=CKPT_SCHEMA, spec=json.loads(env.spec.to_json()), C=C, H=H, meta=dict(tag="t"),
               weight_sha256=tensor_sha256(w.view(torch.int16)), bias_sha256=tensor_sha256(b.view(torch.int16)))
    nbits = (w.numel() + 7) // 8
    torch.save(dict(rec=rec, weight=w.cpu(), bias=b.cpu(), mask_bits=torch.full((nbits,), 255, dtype=torch.uint8)),
               tmp_path / "ck.pt")
    a = CoverTopK16Layer.from_checkpoint(tmp_path / "ck.pt", input_dim=D, output_dim=G, normalization="energy")
    ref, _, _ = build("energy", w=w, b=b)
    assert Cn.digest(a)["sha"] == Cn.digest(ref)["sha"] and a.metadata["source_checkpoint"]["verified"]
    rec2 = dict(rec, weight_sha256="0" * 64)
    torch.save(dict(rec=rec2, weight=w.cpu(), bias=b.cpu(), mask_bits=None), tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="sha256"):
        CoverTopK16Layer.from_checkpoint(tmp_path / "bad.pt", input_dim=D, output_dim=G)
    torch.save(dict(rec=rec, weight=w.cpu(), bias=b.cpu(), mask_bits=torch.zeros(nbits, dtype=torch.uint8)),
               tmp_path / "masked.pt")
    with pytest.raises(ValueError, match="disables valid slots"):
        CoverTopK16Layer.from_checkpoint(tmp_path / "masked.pt", input_dim=D, output_dim=G)


@requires_cuda_backend
def test_device_move_and_reload_invalidate_runtime():
    lay, _, _ = build()
    x = torch.randn(40, D, device="cuda").bfloat16().float()
    dy = torch.randn(40, G, device="cuda")
    y0, dx0, _, _ = run(lay, x, dy, "legacy")
    k0 = lay._rt_key
    lay.to("cpu")
    assert lay.counts.device.type == "cpu" and lay.counts_f.device.type == "cpu"
    lay.to("cuda")
    y1, dx1, _, _ = run(lay, x, dy, "legacy")
    assert lay._rt_key != k0 and maxrel(y1, y0) < 1e-6 and maxrel(dx1, dx0) < 1e-6
    sd = {k: v.clone() for k, v in lay.state_dict().items()}
    lay.load_state_dict(sd)
    assert lay._rt is None
    y2, _, _, _ = run(lay, x, dy, "legacy")
    assert maxrel(y2, y0) < 1e-6


# ---------------------------------------------------------------------------------------------- import / docs
CPU_SCRIPT = r"""
import torch
from sparse_pooling import CoverTopK16Layer
from sparse_pooling.experimental.kgen_layout import KLayout, zero_padding_
from sparse_pooling.supergroups import ConnectivityPlan
L = KLayout(256, 3452, 8, 64, 256)
g = torch.Generator().manual_seed(0)
w = (torch.randn(L.weight_shape, generator=g) / 8).bfloat16(); b = torch.zeros(L.bias_shape, dtype=torch.bfloat16)
zero_padding_(L, w, b)
lay = CoverTopK16Layer.from_envelope(w, b, input_dim=256, hidden_dim=3452, output_dim=8,
                                     connectivity=ConnectivityPlan(64, 2, seed=1), device="cpu")
assert lay.metadata["valid_slots"] == 16 * 3452
assert not torch.cuda.is_initialized(), "CUDA was initialized"
print("ok")
"""


def test_cpu_only_import_and_host_construction(tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", NVCC=str(tmp_path / "no-nvcc"), PATH="/usr/bin:/bin",
               SPARSE_POOL_CACHE_DIR=str(tmp_path / "cache"), PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, "-c", CPU_SCRIPT], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr
    assert not (tmp_path / "cache").exists()


@requires_cuda_backend
def test_documented_quickstart_runs(tmp_path):
    doc = (ROOT / "docs" / "cover_topk16.md").read_text()
    m = re.search(r"<!-- quickstart -->\s*```python\n(.*?)```", doc, re.S)
    assert m, "quickstart block missing"
    code = m.group(1).replace("cover_topk16_packed.pt", str(tmp_path / "q.pt"))
    ns = {}
    exec(compile(code, "quickstart", "exec"), ns)
    assert ns["layer"].metadata["valid_slots"] == 16 * ns["H"]
    assert "kgen_dx" in ns["layer"].last_info
