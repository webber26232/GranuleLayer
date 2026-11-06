"""K16 dX selection of the kgen path through the profile / dispatch system (backends/profiles KgenDxContext +
TuningProfile.kgen_dx_variant, backends/dispatch.kgen_dx_resolve) and the tiling variants' correctness.

Host-only (synthetic capability records = control-flow evidence): the RTX 3090 scope selects mt4_nw8_wg4 exactly
inside its validated workload family, every out-of-scope field falls back to legacy with a reason, a forced
generic_cuda profile (plan or environment) and non-3090 devices never receive the measured choice, a forced rtx3090
profile on another device is labelled user-forced-unvalidated, explicit requests are capability checked, strict
determinism is rejected, and the cache keys separate devices / profiles / batches / builds.

GPU: every screened tiling equals the FP64 reference and the legacy launch (forward and dW / db untouched); the
executed record agrees with the decision and the native launch arguments; auto resolves to WG4 through the real path
when the call is in scope and generic_cuda suppresses it; the kernel-attribute interleave regression.
"""
import dataclasses

import pytest
import torch

from conftest import requires_cuda_backend
from test_cover_topk16 import D, G, build, masked_reference, maxrel, run

SCREEN_NAMES = ["mt4_nw8_wg2", "mt4_nw8_wg4", "mt4_nw8_wg8", "mt2_nw8_wg1", "mt2_nw8_wg2", "mt4_nw4_wg1", "mt4_nw4_wg2",
                "mt2_nw4_wg1"]
COMPILED = [(4, 8, 0), (2, 8, 0), (4, 4, 0), (2, 4, 0)]
LEGACY = (4, 8, 2)


def ctx(**kw):
    from sparse_pooling.backends.profiles import KgenDxContext

    base = dict(D=1024, S=64, K=16, C=16, G=1024, H=12_500_000, R=256, envelope_K=64, envelope_C=256,
                source="cover-topk16", act="relu", precision="split", reduction="fast", batch=8192)
    base.update(kw)
    return KgenDxContext(**base)


def caps3090(**kw):
    from sparse_pooling.backends.capabilities import synthetic

    return synthetic("NVIDIA GeForce RTX 3090", (8, 6), sm_count=82, **kw)


def resolve(caps, request="auto", profile="auto", c=None, compiled=COMPILED, **kw):
    from sparse_pooling.backends.dispatch import kgen_dx_resolve
    from sparse_pooling.backends.profiles import select

    return kgen_dx_resolve(caps, select(caps, profile), request, c or ctx(), compiled, LEGACY, **kw)


@pytest.fixture(autouse=True)
def _no_profile_env(monkeypatch):
    monkeypatch.delenv("SPARSE_POOL_PROFILE", raising=False)


# ------------------------------------------------------------------------------------------- host-only dispatch
@pytest.mark.parametrize("batch", [64, 512, 8192])
def test_rtx3090_auto_selects_wg4_inside_the_validated_scope(batch):
    d = resolve(caps3090(), c=ctx(batch=batch))
    assert (d.name, d.mt, d.nw, d.wg, d.legacy, d.source) == ("mt4_nw8_wg4", 4, 8, 4, False, "profile")
    assert d.status == "hardware-validated" and d.profile == "rtx3090" and d.profile_version == "2026.11.05"
    desc = d.describe()
    assert desc["tokens_per_cta"] == 64 and desc["threads_per_cta"] == 256 and desc["fp32_slab_copies"] == 2
    assert desc["shared_bytes"] == 43008 and desc["validated_scope"]["batches"] == (64, 512, 8192)


@pytest.mark.parametrize("field,value", [("batch", 1024), ("batch", 4096), ("precision", "bf16"), ("act", "identity"),
                                         ("D", 512), ("S", 32), ("H", 12_499_968), ("G", 512), ("R", 128),
                                         ("envelope_C", 128), ("envelope_K", 0), ("source", "generic"), ("C", 32)])
def test_out_of_scope_calls_fall_back_to_legacy_with_reason(field, value):
    d = resolve(caps3090(), c=ctx(**{field: value}))
    assert d.legacy and d.name == "legacy" and d.source == "fallback" and d.status == "generic fallback"
    assert field in d.reason or (field == "batch" and "batch M" in d.reason), d.reason
    assert (d.mt, d.nw, d.wg) == LEGACY


def test_forced_generic_profile_suppresses_the_3090_choice(monkeypatch):
    d = resolve(caps3090(), profile="generic_cuda")
    assert d.legacy and d.profile == "generic_cuda" and "no measured kgen dX entry" in d.reason
    monkeypatch.setenv("SPARSE_POOL_PROFILE", "generic_cuda")
    d = resolve(caps3090())
    assert d.legacy and d.profile == "generic_cuda"


@pytest.mark.parametrize("caps_kw", [dict(name="NVIDIA A40"), dict(name="NVIDIA GeForce RTX 3090 Ti"),
                                     dict(sm_count=84), dict(smem_per_block_optin=163 * 1024, cc=(8, 0)),
                                     dict(cc=(8, 9))])
def test_non_3090_devices_get_the_generic_choice(caps_kw):
    from sparse_pooling.backends.capabilities import synthetic

    kw = dict(name="NVIDIA GeForce RTX 3090", cc=(8, 6), sm_count=82)
    kw.update(caps_kw)
    caps = synthetic(kw.pop("name"), kw.pop("cc"), **kw)
    d = resolve(caps)
    assert d.legacy and d.profile == "generic_cuda" and d.status == "generic fallback"
    e = resolve(caps, request="mt4_nw8_wg4")
    assert not e.legacy and e.status == "explicit, unvalidated" and e.scope is None
    # forcing the 3090 profile elsewhere: its choice, labelled user-forced (no borrowed validation claim)
    f = resolve(caps, profile="rtx3090")
    if not f.legacy:
        assert f.status == "user-forced-unvalidated" and f.name == "mt4_nw8_wg4"


def test_explicit_requests_and_legacy_override():
    d = resolve(caps3090(), request="legacy")
    assert d.legacy and d.source == "explicit-legacy" and (d.mt, d.nw, d.wg) == LEGACY
    e = resolve(caps3090(), request="mt4_nw8_wg4")
    assert e.status == "explicit, inside the validated scope" and e.scope is not None
    e = resolve(caps3090(), request="mt2_nw4_wg1")
    assert e.status == "explicit, unvalidated" and (e.mt, e.nw, e.wg) == (2, 4, 1)
    e = resolve(caps3090(), request="mt4_nw8_wg4", c=ctx(batch=1024))  # outside the measured batches
    assert e.status == "explicit, unvalidated"


@pytest.mark.parametrize("req,msg", [("mt4_nw8_wg3", "must divide"), ("mt8_nw8_wg2", "not compiled"),
                                     ("mt4_nw8_wg1", "shared memory"), ("nonsense", "expected one of"),
                                     ("mt4_nw8_wg4_c16", "expected one of")])
def test_illegal_explicit_requests_raise_with_reason(req, msg):
    with pytest.raises(ValueError, match=msg):
        resolve(caps3090(), request=req)


def test_profile_choice_illegal_on_device_falls_back():
    caps = caps3090(smem_per_block_optin=40 * 1024)  # WG4 needs 42 KiB
    d = resolve(caps, profile="rtx3090")
    assert d.legacy and "cannot launch here" in d.reason and "shared memory" in d.reason
    d = resolve(caps3090(), compiled=[(2, 4, 0)])  # WG4's MT4 / NW8 entry missing from the loaded library
    assert d.legacy and "not compiled" in d.reason


def test_strict_determinism_is_rejected():
    with pytest.raises(ValueError, match="deterministic"):
        resolve(caps3090(), c=ctx(reduction="deterministic"))
    with pytest.raises(ValueError, match="deterministic"):
        resolve(caps3090(), request="legacy", c=ctx(reduction="deterministic"))


def test_k_other_than_16_keeps_its_only_tiling():
    d = resolve(caps3090(), c=ctx(K=8))
    assert d.legacy and d.source == "fixed"


def test_cache_identity_separates_device_profile_batch_and_build():
    from sparse_pooling.backends import dispatch

    dispatch._KGEN_DX_CACHE.clear()
    a = resolve(caps3090(), c=ctx(batch=8192))
    assert resolve(caps3090(), c=ctx(batch=8192)) is a  # cached
    assert resolve(caps3090(), c=ctx(batch=1024)).legacy  # batch is part of the key
    n = len(dispatch._KGEN_DX_CACHE)
    resolve(caps3090(index=1, uuid="11111111-0000-0000-0000-000000000000"), c=ctx(batch=8192), device_index=1)
    resolve(caps3090(), c=ctx(batch=8192), library_hash="other-build")
    resolve(caps3090(), profile="generic_cuda", c=ctx(batch=8192))
    assert len(dispatch._KGEN_DX_CACHE) == n + 3


def test_smem_formula_and_legacy_footprints():
    from sparse_pooling.experimental import kgen_dx_variants as DXV

    assert DXV.smem_bytes(1024, 64, 4, 8, 2) == 75776 and DXV.smem_bytes(1024, 64, 4, 8, 4) == 43008
    assert [DXV.smem_bytes(1024, 64, m, n, g) // 1024 for m, n, g in DXV.SCREEN] == [74, 42, 26, 70, 38, 74, 42, 38]


# ------------------------------------------------------------------------------------------- GPU
@requires_cuda_backend
def test_all_screened_tilings_equal_reference_and_legacy():
    from sparse_pooling.experimental import reference as REF

    lay, w, b = build()
    env, em, wm, bm, _ = masked_reference(w, b)
    for B in (8, 40, 64, 512):
        g = torch.Generator(device="cuda").manual_seed(B)
        x = torch.randn(B, D, device="cuda", generator=g).bfloat16().float()
        dy = torch.randn(B, G, device="cuda", generator=g)
        yr, dxr, _, _ = REF.grads(x.double(), env.indices.long(), wm.double(), bm.double(),
                                  env.layout.counts_tensor("cuda"), dy.double())
        y0, dx0, dw0, db0 = run(lay, x, dy, "legacy")
        assert maxrel(dx0, dxr) < 2e-5
        for v in SCREEN_NAMES:
            y, dx, dw, db = run(lay, x, dy, v)
            assert maxrel(y, y0) < 1e-6, v  # forward untouched by the dX choice
            assert torch.equal(dw, dw0) and torch.equal(db, db0), v  # dW / db launch untouched
            assert maxrel(dx, dxr) < 2e-5 and maxrel(dx, dx0) < 2e-5, (v, B)
            info = lay.last_info["kgen_dx"]
            assert info["variant"]["name"] == v and info["tokens"] == 16 * int(v[2])


@requires_cuda_backend
def test_executed_record_agrees_with_decision_and_launch_arguments():
    lay, _, _ = build()
    rt = lay._runtime()
    calls = []
    orig_dxv, orig = rt.backend.launch_dxv, rt.backend.launch

    def spy_dxv(a, act, split, mt, nw, impl, n_chunks, n_tiles, device):
        calls.append(("dxv", mt, nw, a.dx_group_warps, n_tiles, n_chunks))
        return orig_dxv(a, act, split, mt, nw, impl, n_chunks, n_tiles, device)

    def spy(K, kind, a, act, split, n_chunks, n_tiles, device):
        calls.append((kind, K, a.dx_group_warps, n_tiles, n_chunks))
        return orig(K, kind, a, act, split, n_chunks, n_tiles, device)

    rt.backend.launch_dxv, rt.backend.launch = spy_dxv, spy
    try:
        x = torch.randn(64, D, device="cuda").bfloat16().float()
        dy = torch.randn(64, G, device="cuda")
        for v in ("mt4_nw8_wg4", "legacy", "mt2_nw4_wg1"):
            calls.clear()
            run(lay, x, dy, v)
            info = lay.last_info["kgen_dx"]
            dec = lay.dx_variant_resolved(64)
            assert info["variant"] == dec.describe()
            if v == "legacy":
                dx = [c for c in calls if c[0] == "dx"]
                assert len(dx) == 1 and dx[0][2] == rt.dx_group_warps == info["launched_group_warps"]
                assert info["entry"] == "sp_kg_k16" and dx[0][3] == info["token_tiles"]
            else:
                dx = [c for c in calls if c[0] == "dxv"]
                assert len(dx) == 1 and dx[0][1:4] == (dec.mt, dec.nw, dec.wg) and info["entry"] == "sp_kg_k16_dxv"
                assert dx[0][4] == info["token_tiles"] == -(-64 // (16 * dec.mt))
            assert info["library_hash"] == rt.backend.native.ident["library_hash"]
            assert [c[0] for c in calls if c[0] not in ("dxv", "dx")] == ["fwd", "dw"]  # unchanged fwd / dW launches
    finally:
        rt.backend.launch_dxv, rt.backend.launch = orig_dxv, orig


@requires_cuda_backend
def test_auto_through_the_real_path_and_generic_override(monkeypatch):
    """Widen the RTX 3090 scope to this small case for the duration of the test: auto must then launch WG4 through
    the real dispatch / native path; generic_cuda (plan or environment) must suppress it on the same device."""
    from sparse_pooling.backends.capabilities import query
    from sparse_pooling.backends.profiles import rtx3090

    caps = query(torch.device("cuda"))
    if not rtx3090.is_rtx3090(caps)[0]:
        pytest.skip("the measured RTX 3090 profile applies to the RTX 3090 only")
    lay, _, _ = build()
    c = lay.kgen_dx_context(64)
    entry = dict(rtx3090.KGEN_DX_AUTO[0], D=c.D, G=c.G, H=c.H, R=c.R, batches=(64,))
    monkeypatch.setattr(rtx3090, "KGEN_DX_AUTO", (entry,))
    x = torch.randn(64, D, device="cuda").bfloat16().float()
    dy = torch.randn(64, G, device="cuda")
    run(lay, x, dy, "auto")
    v = lay.last_info["kgen_dx"]["variant"]
    assert v["name"] == "mt4_nw8_wg4" and v["source"] == "profile" and v["status"] == "hardware-validated"
    run(lay, torch.randn(72, D, device="cuda"), torch.randn(72, G, device="cuda"), "auto")  # batch outside scope
    assert lay.last_info["kgen_dx"]["variant"]["name"] == "legacy"
    assert "batch M = 72" in lay.last_info["kgen_dx"]["variant"]["reason"]
    gen = dataclasses.replace(lay.plan, tuning_profile="generic_cuda")
    lay.plan = gen
    run(lay, x, dy, "auto")
    v = lay.last_info["kgen_dx"]["variant"]
    assert v["name"] == "legacy" and v["profile"] == "generic_cuda"
    lay.plan = dataclasses.replace(gen, tuning_profile="auto")
    monkeypatch.setenv("SPARSE_POOL_PROFILE", "generic_cuda")
    lay._rt = None
    run(lay, x, dy, "auto")
    assert lay.last_info["kgen_dx"]["variant"]["name"] == "legacy"
    # explain(): pending without a batch, planned with one, never an executed claim
    monkeypatch.delenv("SPARSE_POOL_PROFILE")
    lay._rt = None
    ex = lay.explain()
    assert ex["dx"]["decision"].startswith("pending")
    assert lay.explain(64)["dx"]["planned"]["name"] == "mt4_nw8_wg4"


@requires_cuda_backend
def test_deterministic_plan_is_rejected_at_call_time():
    lay, _, _ = build()
    lay.plan = dataclasses.replace(lay.plan, reduction="deterministic")
    with pytest.raises(ValueError, match="deterministic"):
        lay(torch.randn(16, D, device="cuda").bfloat16())


@requires_cuda_backend
def test_interleaved_legacy_auto_and_explicit_launches_keep_their_shared_memory():
    """Regression: (MT 4, NW 8) is the legacy kernel symbol; interleaving legacy, auto and explicit launches at different
    D (different dynamic shared memory) must never lower the attribute below a later valid legacy launch."""
    from sparse_pooling.experimental import kgen_dx_variants as DXV
    from sparse_pooling.experimental import reference as REF

    small = build(D=256)
    big = build(D=1024, H=4 * 431 + 4)
    rt = big[0]._runtime()
    rt.backend.dxv_attributes(4, 8, 0, 1, 1, DXV.smem_bytes(1024, 64, 4, 8, 8))  # a small request: must not lower
    refs = {}
    for name, (lay, w, b) in (("small", small), ("big", big)):
        env, em, wm, bm, _ = masked_reference(w, b, H=lay.layout.hidden_dim, D=lay.layout.input_dim)
        refs[name] = (lay, env, wm, bm)
    seq = [("big", "mt4_nw8_wg8"), ("small", "legacy"), ("big", "mt4_nw8_wg2"), ("big", "legacy"),
           ("small", "mt4_nw8_wg4"), ("big", "auto"), ("small", "legacy"), ("big", "mt4_nw8_wg4"), ("big", "legacy")]
    for name, v in seq:
        lay, env, wm, bm = refs[name]
        L = env.layout
        x = torch.randn(40, L.input_dim, device="cuda").bfloat16().float()
        dy = torch.randn(40, L.groups, device="cuda")
        _, dx, _, _ = run(lay, x, dy, v)
        _, dxr, _, _ = REF.grads(x.double(), env.indices.long(), wm.double(), bm.double(), L.counts_tensor("cuda"),
                                 dy.double())
        assert maxrel(dx, dxr) < 2e-5, (name, v)


@requires_cuda_backend
def test_compiled_variant_attributes_readable():
    from sparse_pooling.experimental import kgen_dx_variants as DXV

    lay, _, _ = build(D=1024, H=4 * 431 + 4)
    rt = lay._runtime()
    assert sorted(rt.backend.dxv_compiled) == sorted(COMPILED)
    for m, n, g in DXV.SCREEN:
        assert DXV.reason(rt.caps, rt.backend, 1024, 64, 16, m, n, g) == "" or rt.caps.smem_per_block_optin < 75776
    at = rt.backend.dxv_attributes(4, 8, 0, 1, 1, DXV.smem_bytes(1024, 64, 4, 8, 4))
    assert at["registers_per_thread"] > 0 and at["active_ctas_per_sm"] >= 1
