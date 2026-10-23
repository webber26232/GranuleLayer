"""Plan resolution (backends/dispatch.py) on synthetic capability records: control flow only.

Nothing here runs a kernel; the device records are labelled ``source="synthetic"`` and are never
hardware evidence (docs/gpu_portability.md, support matrix)."""
import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import plan_matrix  # noqa: E402
from sparse_pooling import PhysicalPlan  # noqa: E402
from sparse_pooling.backends.capabilities import synthetic  # noqa: E402
from sparse_pooling.backends.dispatch import Resolver, UnsupportedError, matrix_decisions  # noqa: E402
from sparse_pooling.backends.profiles import select  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "data" / "plan_matrix_rtx3090.json.gz"
RTX3090 = synthetic("NVIDIA GeForce RTX 3090", (8, 6), sm_count=82)  # synthetic record of the measured board
# Synthetic fixtures: resource values are illustrative (A100 / Ada / Hopper-like opt-in limits from
# NVIDIA's published per-architecture tables), not measurements of any board.
A100_LIKE = synthetic("synthetic sm_80 (A100-like)", (8, 0), sm_count=108, smem_per_block_optin=163 * 1024)
ADA_LIKE = synthetic("synthetic sm_89 (Ada-like)", (8, 9), sm_count=128, smem_per_block_optin=99 * 1024)
HOPPER_LIKE = synthetic("synthetic sm_90 (Hopper-like)", (9, 0), sm_count=132, smem_per_block_optin=227 * 1024)
OTHER_SM86 = synthetic("NVIDIA GeForce RTX 3080", (8, 6), sm_count=68)
TURING = synthetic("synthetic sm_75", (7, 5), sm_count=40, smem_per_block_optin=64 * 1024)
SMALL_SMEM = synthetic("synthetic sm_80, 48 KiB opt-in", (8, 0), smem_per_block_optin=48 * 1024)


@pytest.fixture(autouse=True)
def _no_profile_env(monkeypatch):
    monkeypatch.delenv("SPARSE_POOL_PROFILE", raising=False)  # synthetic tests choose profiles explicitly


def _rows():
    return json.load(gzip.open(FIXTURE, "rt"))["rows"]


def test_rtx3090_reproduces_the_reviewed_decision_matrix():
    """Every dispatch decision of the reviewed implementation (2b9a84d, RTX 3090) is reproduced by
    the resolver under the automatically matched RTX 3090 profile (4,560 workload / override cases:
    grouped phases, scale rule, scatters, dW projection, deterministic mode, generic tilings,
    epilogues, staging, errors)."""
    d = json.load(gzip.open(FIXTURE, "rt"))
    assert d["commit"].startswith("2b9a84d") and d["impl"] == "legacy"
    bad = []
    for row in d["rows"]:
        got = json.loads(json.dumps(matrix_decisions(plan_matrix.make_rt(row["case"], None), row["case"]["B"], RTX3090)))
        if got != row["decisions"]:
            bad.append((row["case"], {k: (row["decisions"].get(k), got.get(k)) for k in set(got) | set(row["decisions"])
                                      if got.get(k) != row["decisions"].get(k)}))
    assert not bad, f"{len(bad)} differing cases, first: {bad[:3]}"


def test_profile_selection_and_status():
    s = select(RTX3090)
    assert s.profile.name == "rtx3090" and not s.forced and s.status == "hardware-validated"
    s = select(OTHER_SM86)  # same compute capability, not the measured board
    assert s.profile.name == "generic_cuda" and s.status == "capability-eligible, hardware-unvalidated"
    s = select(OTHER_SM86, "rtx3090")  # forced: legal, but explicitly unvalidated
    assert s.profile.name == "rtx3090" and s.forced and s.status == "user-forced-unvalidated"
    assert "does not match" in s.reason
    s = select(RTX3090, "generic_cuda")
    assert s.profile.name == "generic_cuda" and s.status == "numerics-validated, performance-unmeasured"
    for caps in (A100_LIKE, ADA_LIKE, HOPPER_LIKE):
        assert select(caps).profile.name == "generic_cuda"
    with pytest.raises(ValueError, match="unknown tuning profile"):
        select(RTX3090, "h100")


def test_profile_env_override(monkeypatch):
    monkeypatch.setenv("SPARSE_POOL_PROFILE", "generic_cuda")
    assert select(RTX3090).profile.name == "generic_cuda" and select(RTX3090).forced
    assert select(RTX3090, "rtx3090").profile.name == "rtx3090"  # an explicit plan value wins over the env


def _case(**kw):
    c = dict(C=8, S=64, ub=6.0, B=8192, D=1024, G=1024, H=100_000_000, precision="split", act="relu", det=False,
             pool="mean", plan={})
    c.update(kw)
    return c


def _rp(caps, case, part="bwd", profile="auto"):
    rt = plan_matrix.make_rt(case, None)
    return Resolver(caps, select(caps, profile)).resolve(rt, case["B"], part)


@pytest.mark.parametrize("caps", [A100_LIKE, ADA_LIKE, HOPPER_LIKE], ids=["sm80", "sm89", "sm90"])
def test_generic_policy_on_compatible_devices(caps):
    """Compatible devices get the conservative generic policy within THEIR shared-memory limit (no
    99 KiB constant), never a 3090 table."""
    rp = _rp(caps, _case())
    assert rp.profile == "generic_cuda" and rp.dx["kernel"] == "grouped" and rp.dx["scatter"] == "baseline"
    assert rp.dx["plan"] == "tiles" and rp.dw["staging"] == "full" and rp.dx["epilogue"] == "generic"
    assert rp.dx["smem"] <= caps.smem_per_block_optin and rp.dw["smem"] <= caps.smem_per_block_optin
    r = Resolver(caps, select(caps))
    assert r.limit == caps.smem_per_block_optin
    f = _rp(caps, _case(), "fwd")
    assert f.fwd["kernel"] == "grouped" and f.fwd["smem"] <= caps.smem_per_block_optin


def test_larger_shared_memory_admits_larger_tiles():
    """A tile the 99 KiB RTX 3090 cannot hold is legal on a 227 KiB device (S = 64, T = 64 dX with
    8 warp-private FP32 copies: 138 KiB)."""
    case = _case(S=64, plan=dict(cuda_grouped_dx_tokens=64, cuda_grouped_dx_reduction=1, cuda_input_supergroup_backend="grouped"))
    with pytest.raises(ValueError):
        _rp(RTX3090, case)
    assert _rp(HOPPER_LIKE, case).dx["tokens"] == 64


def test_low_shared_memory_shrinks_or_rejects():
    # the generic forward slab (D x 32 tokens, 64 KiB at D = 1024) does not fit 48 KiB: Triton forward
    assert _rp(SMALL_SMEM, _case(S=64), "fwd").fwd["backend"] == "triton"
    # at D = 256 the native forward is legal and the grouped tile fits the device
    f = _rp(SMALL_SMEM, _case(S=64, D=256, G=64, H=20_000_000), "fwd")
    assert f.fwd["kernel"] == "grouped" and f.fwd["smem"] <= 48 * 1024
    # the native backward needs the generic k_dx as its dX fallback (D * 96 B = 96 KiB at D = 1024):
    # it is unavailable there, so automatic plans run the Triton backward, with the reason recorded
    rp = _rp(SMALL_SMEM, _case(S=64))
    assert rp.bwd["backend"] == "triton" and "exceeds the device's shared memory" in rp.bwd["reason"]
    # ... and an explicit grouped request is refused rather than silently dropped
    with pytest.raises(UnsupportedError, match="native backward is unavailable"):
        _rp(SMALL_SMEM, _case(S=64, plan=dict(cuda_input_supergroup_backend="grouped")))
    # at D = 256 everything fits; an explicit tile that cannot fit raises (legality before overrides)
    rp = _rp(SMALL_SMEM, _case(S=64, D=256, G=64, H=20_000_000))
    assert rp.dx["kernel"] == "grouped" and rp.dx["smem"] <= 48 * 1024
    with pytest.raises(UnsupportedError):
        _rp(SMALL_SMEM, _case(S=64, D=256, G=64, H=20_000_000,
                              plan=dict(cuda_input_supergroup_backend="grouped", cuda_grouped_dx_tokens=128,
                                        cuda_grouped_dx_reduction=1)))


def test_device_without_required_instructions():
    r = Resolver(TURING, select(TURING))
    rt = plan_matrix.make_rt(_case(), None)
    assert "sm_80" in r.forward_reason(rt) and "mma.sync" in r.forward_reason(rt)
    assert r.backward_reason(rt) and r.sg_reason(rt, "dx", 8192)
    rp = _rp(TURING, _case(), "fwd")
    assert rp.fwd["backend"] == "triton"  # the existing BF16 Triton path; no silent precision change
    # explicit grouped on a device without the instructions raises
    with pytest.raises(UnsupportedError, match="sm_80"):
        _rp(TURING, _case(plan=dict(cuda_input_supergroup_backend="grouped")), "fwd")
    # strict determinism cannot be satisfied there (it needs the native grouped dX)
    rt.det = True
    assert r.det_reason(rt, 8192)


def test_precedence_determinism_over_overrides_and_profile():
    # 1. numerics: a deterministic request never inherits the fast shared-CAS plan ...
    rp = _rp(RTX3090, _case(det=True, ub=1.0))
    assert rp.dx["plan"] == "private_t32" and rp.dx["group_warps"] == 1
    # ... and rejects explicit unordered overrides instead of falling back
    for ov in (dict(cuda_grouped_dx_plan="shared_t64"), dict(cuda_grouped_dx_reduction=8)):
        with pytest.raises(UnsupportedError):
            _rp(RTX3090, _case(det=True, plan=ov))
    # also under a forced profile
    with pytest.raises(UnsupportedError):
        _rp(RTX3090, _case(det=True, plan=dict(cuda_grouped_dx_plan="shared_t64")), profile="generic_cuda")
    # 3. explicit valid overrides beat the profile
    rp = _rp(RTX3090, _case(plan=dict(cuda_grouped_dx_tokens=128, cuda_grouped_dx_warps=16)))
    assert (rp.dx["tokens"], rp.dx["warps"], rp.dx["plan"]) == (128, 16, "tiles")
    # 4. the profile's measured choice applies only without overrides
    assert _rp(RTX3090, _case(ub=6.0)).dx["plan"] == "private_t32"
    assert _rp(RTX3090, _case(ub=1.0)).dx["plan"] == "tiles"
    # 5. forcing a profile never bypasses legality: an illegal explicit scatter still raises
    with pytest.raises(UnsupportedError):
        _rp(RTX3090, _case(plan=dict(cuda_grouped_dx_scatter="feature_owner")), profile="rtx3090")


def test_insufficient_deterministic_workspace():
    r = Resolver(RTX3090, select(RTX3090))
    rt = plan_matrix.make_rt(_case(det=True, plan=dict(deterministic_workspace_bytes=1024)), None)
    with pytest.raises(RuntimeError, match="no atomic fallback"):
        r.det_window(rt, 1024, 64, 8192, "dx")


def test_resolved_plan_is_immutable_and_cached():
    import types

    rt = plan_matrix.make_rt(_case(), None)
    r = Resolver(RTX3090, select(RTX3090))
    rp = r.resolve(rt, 8192, "bwd")
    with pytest.raises(TypeError):
        rp.dx["tokens"] = 1
    with pytest.raises(Exception):
        rp.batch = 1
    assert isinstance(rp.dx, types.MappingProxyType) and rp.as_dict()["dx"]["scatter"]


def test_documented_profile_template_is_usable_and_unregistered():
    """docs/adding_a_gpu.md's example profile runs through the resolver as written (not registered,
    so it can never be selected automatically)."""
    import re

    from sparse_pooling.backends.profiles import ProfileSelection, registered

    doc = (Path(__file__).resolve().parents[2] / "docs" / "adding_a_gpu.md").read_text()
    code = re.search(r"## 4\..*?```python\n(.*?)```", doc, re.S).group(1)
    ns: dict = {}
    exec(code, ns)
    prof = ns["ExampleProfile"]()
    assert prof.name not in registered()
    caps = synthetic("NVIDIA Example Board", (8, 9), sm_count=128)
    assert prof.matches(caps)[0] and not prof.matches(RTX3090)[0]
    sel = ProfileSelection(prof, prof.name, True, "test", False, False)
    rp = Resolver(caps, sel).resolve(plan_matrix.make_rt(_case(), None), 8192, "bwd")
    assert rp.profile == "example_board" and rp.dx["kernel"] == "grouped"
    assert select(caps).profile.name == "generic_cuda"  # the template does not change auto selection


def test_explicit_legal_requests_beat_the_profile():
    """Precedence 3 > 4: a legal explicit request is honoured under any profile, and an explicit
    option that needs a particular (bitwise-equivalent) epilogue fixes the automatic epilogue."""
    # indexed k_dw staging at C = 64: generic prefers full, but an explicit request is legal there
    case = _case(C=64, S=0, pool="mean", plan=dict(cuda_dw_staging="indexed"))
    rt = plan_matrix.make_rt(case, None)
    for prof in ("generic_cuda", "rtx3090"):
        r = Resolver(RTX3090, select(RTX3090, prof))
        assert r.dw_staging(rt.plan, rt.layout, 1) == ("indexed", "")
    r = Resolver(RTX3090, select(RTX3090, "generic_cuda"))
    rt = plan_matrix.make_rt(_case(C=64, S=0, pool="mean"), None)
    assert r.dw_staging(rt.plan, rt.layout, 1) == ("full", "")  # auto: the profile's choice
    # compact dW projection / fused backward under the generic profile (auto epilogues = generic)
    rp = _rp(RTX3090, _case(plan=dict(cuda_grouped_dw_projection="compact", cuda_input_supergroup_backend="grouped")),
             profile="generic_cuda")
    assert rp.dw["projection"] == "compact" and rp.dw["epilogue"] == "fast"
    rp = _rp(RTX3090, _case(S=16, plan=dict(cuda_grouped_backward="fused", cuda_input_supergroup_backend="grouped")),
             profile="generic_cuda")
    assert rp.bwd["organization"] == "fused" and (rp.dx["epilogue"], rp.dw["epilogue"]) == ("mask", "fast")
    # an explicit generic epilogue still conflicts with compact (raises, never silently changed)
    with pytest.raises(UnsupportedError):
        _rp(RTX3090, _case(plan=dict(cuda_grouped_dw_projection="compact", cuda_dw_epilogue="generic",
                                     cuda_input_supergroup_backend="grouped")))


def test_documented_unknown_device_snippet_runs_as_written(capsys):
    """docs/adding_a_gpu.md §5 (synthetic resource-constrained device) runs and prints what it says."""
    import re

    doc = (Path(__file__).resolve().parents[2] / "docs" / "adding_a_gpu.md").read_text()
    code = re.search(r"## 5\..*?```python\n(.*?)```", doc, re.S).group(1)
    root = Path(__file__).resolve().parents[2]
    cwd = Path.cwd()
    try:
        import os

        os.chdir(root)  # the snippet's sys.path entry is relative to the repository root
        exec(code, {})
    finally:
        os.chdir(cwd)
    out = capsys.readouterr().out
    assert "capability-eligible, hardware-unvalidated generic_cuda triton" in out
    assert "k_dx slab of D = 1024 exceeds the device's shared memory" in out
