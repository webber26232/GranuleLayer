"""Build identity, bundles, target selection and cache publication (backends/cuda/build.py).

Host-only except where marked: the identity tests use a synthetic toolchain record, so they run
without nvcc or a GPU."""
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from sparse_pooling.backends import capabilities as C
from sparse_pooling.backends.cuda import build as B

TC = C.ToolchainInfo(nvcc="/opt/cuda/bin/nvcc", version="12.1", version_text="nvcc release 12.1, V12.1.66",
                     real_archs=("sm_75", "sm_80", "sm_86", "sm_87", "sm_89", "sm_90"),
                     virtual_archs=("compute_80", "compute_86", "compute_89", "compute_90"), host_compiler="gcc 12.3")
RTX = C.synthetic("NVIDIA GeForce RTX 3090", (8, 6), sm_count=82)


def test_every_source_unit_is_in_exactly_one_bundle_role():
    """No orphan or doubly assigned unit; production holds no experimental / diagnostic sources."""
    units = sorted(str(p.relative_to(B.SRC_DIR)) for p in B.SRC_DIR.rglob("*.cu"))
    listed = [u for b in B.BUNDLES.values() for u in b if u != "abi.cu"]
    assert sorted(listed + ["abi.cu"]) == units
    assert len(listed) == len(set(listed))
    assert not any(u.startswith(("experimental/", "diagnostics/")) for u in B.BUNDLES["production"])
    assert all("abi.cu" in b for b in B.BUNDLES.values())


def test_production_sources_hold_no_experimental_symbols():
    prod = "".join((B.SRC_DIR / u).read_text() for u in B.BUNDLES["production"])
    for sym in ("sp_sgxe_dx", "sp_sgo_dx", "sp_sgf_c8", "sp_sgx_layout", "sp_diag_proj", "sp_sg_dw_c8_exp",
                "sp_screen"):
        assert not re.search(rf'extern "C" int {sym}\w*\(', prod), sym


def _ident(monkeypatch, bundle="production", target="sm_86", extra="", tc=TC):
    monkeypatch.setenv(B.EXTRA_ENV, extra)
    return B.make_spec(bundle, target, tc).identity()


def test_identity_keys(monkeypatch):
    base = _ident(monkeypatch)
    assert base == _ident(monkeypatch)  # unchanged inputs reuse the artifact
    assert base["library_hash"] != _ident(monkeypatch, target="sm_80")["library_hash"]
    assert base["library_hash"] != _ident(monkeypatch, bundle="experimental")["library_hash"]
    tc2 = C.ToolchainInfo(**{**TC.__dict__, "version_text": "nvcc release 12.4, V12.4.131"})
    assert base["library_hash"] != _ident(monkeypatch, tc=tc2)["library_hash"]
    tc3 = C.ToolchainInfo(**{**TC.__dict__, "host_compiler": "gcc 13.2"})
    assert base["library_hash"] != _ident(monkeypatch, tc=tc3)["library_hash"]
    dev = _ident(monkeypatch, extra="-DSOMETHING=1")
    abl = _ident(monkeypatch, extra="-DSP_ABLATE_DX=1")
    assert dev["kind"] == "development" and abl["kind"] == "ablation" and base["kind"] == "production"
    assert len({base["library_hash"], dev["library_hash"], abl["library_hash"]}) == 3
    assert "-DSP_BUILD_KIND_ID=2" in abl["flags"] and "-DSP_BUILD_KIND_ID=0" in base["flags"]
    # the shared ABI unit gets a different object per bundle (bundle id is compiled in)
    assert base["unit_hashes"]["abi.cu"] != _ident(monkeypatch, bundle="diagnostics")["unit_hashes"]["abi.cu"]


def test_identity_covers_transitive_headers(monkeypatch, tmp_path):
    """Editing a header changes exactly the units that include it (directly or transitively)."""
    import shutil

    src = tmp_path / "cuda"
    shutil.copytree(B.SRC_DIR, src, ignore=shutil.ignore_patterns("build"))
    monkeypatch.setattr(B, "SRC_DIR", src)
    before = _ident(monkeypatch)["unit_hashes"]
    with open(src / "numerics.cuh", "a") as fh:
        fh.write("\n// edit\n")
    after = _ident(monkeypatch)["unit_hashes"]
    changed = {u for u in before if before[u] != after[u]}
    assert "sg_det.cu" not in changed  # does not include numerics.cuh
    assert {"fwd_mean.cu", "bwd_mean.cu", "sg_dx_c8_a_relu.cu", "abi.cu"} <= changed


def test_extra_flags_may_not_set_the_target(monkeypatch):
    monkeypatch.setenv(B.EXTRA_ENV, "-arch=sm_80")
    with pytest.raises(B.BuildError, match="target"):
        B.make_spec("production", "sm_86", TC)


@pytest.mark.parametrize("caps,override,want", [
    (RTX, None, "sm_86"),
    (C.synthetic("a", (8, 0)), None, "sm_80"),
    (C.synthetic("b", (8, 9)), None, "sm_89"),
    (C.synthetic("c", (9, 0)), None, "sm_90"),
    (RTX, "sm_80", "sm_80"),            # an sm_80 cubin runs on sm_86 (same major, newer minor)
    (C.synthetic("d", (8, 8)), None, "sm_87"),  # toolkit lacks sm_88: newest binary-compatible target
])
def test_target_resolution(caps, override, want):
    assert B.resolve_target(caps, TC, override)[0] == want


@pytest.mark.parametrize("caps,override,match", [
    (C.synthetic("turing", (7, 5)), None, "sm_80"),
    (C.synthetic("blackwell-like", (12, 0)), None, "cannot target"),   # future target, old toolkit
    (C.synthetic("e", (10, 0)), None, "cannot target"),
    (RTX, "sm_89", "cannot run"),       # newer minor than the device
    (RTX, "sm_90", "cannot run"),
    (RTX, "sm_8x", "invalid"),
    (C.synthetic("a", (8, 0)), "sm_86", "cannot run"),
])
def test_target_diagnostics(caps, override, match):
    with pytest.raises(B.BuildError, match=match):
        B.resolve_target(caps, TC, override)


def test_unsupported_target_in_toolchain():
    tc = C.ToolchainInfo(**{**TC.__dict__, "real_archs": ("sm_80",)})
    with pytest.raises(B.BuildError, match="cannot emit"):
        B.resolve_target(RTX, tc, "sm_86")
    assert B.resolve_target(RTX, tc)[0] == "sm_80"  # binary-compatible fallback, reported


def test_cache_location(monkeypatch, tmp_path):
    monkeypatch.setenv(B.CACHE_ENV, str(tmp_path / "c"))
    assert B.cache_root() == tmp_path / "c"
    monkeypatch.delenv(B.CACHE_ENV)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "x"))
    assert B.cache_root() == tmp_path / "x" / "granulelayer" / "cuda"
    assert not str(B.cache_root()).startswith(str(B.SRC_DIR))


def test_concurrent_publication_is_atomic(monkeypatch, tmp_path):
    """Two processes / threads building the same library: one compiles, the other waits on the
    lock and reuses it; nothing partial is ever published.  nvcc is replaced by a fake that is slow
    and writes its output in pieces."""
    fake = tmp_path / "nvcc"
    log = tmp_path / "calls"
    fake.write_text(f"""#!{sys.executable}
import sys, time
out = sys.argv[sys.argv.index('-o') + 1]
open({str(log)!r}, 'a').write(' '.join(sys.argv[1:3]) + '\\n')
with open(out, 'wb') as fh:
    fh.write(b'part1'); fh.flush(); time.sleep(0.2); fh.write(b'part2')
""")
    fake.chmod(0o755)
    tc = C.ToolchainInfo(**{**TC.__dict__, "nvcc": str(fake)})
    monkeypatch.setenv(B.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv(B.EXTRA_ENV, "")
    spec = B.make_spec("diagnostics", "sm_86", tc)
    results = []
    ts = [threading.Thread(target=lambda: results.append(B.build(spec)[0])) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(set(results)) == 1 and results[0].read_bytes() == b"part1part2"
    n_link = sum(1 for line in log.read_text().splitlines() if line.startswith("-shared"))
    assert n_link == 1  # built once, reused twice
    assert not list((tmp_path / "cache").rglob("*.tmp"))


@pytest.mark.skipif(not C.find_nvcc(), reason="nvcc not found")
def test_cli_info_runs_without_building(tmp_path):
    env = dict(os.environ, SPARSE_POOL_CACHE_DIR=str(tmp_path))
    r = subprocess.run([sys.executable, "-m", "sparse_pooling.backends.cuda.build", "--target", "sm_80"],
                       capture_output=True, text=True, env=env, cwd=Path(__file__).resolve().parents[2])
    assert r.returncode == 0, r.stderr
    assert "absent" in r.stdout and not list(tmp_path.rglob("*.so"))
