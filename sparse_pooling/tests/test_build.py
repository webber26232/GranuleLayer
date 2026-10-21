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


# ------------------------------------------------------------------------------------------------
# Target grammar with multi-digit majors (sm_100, sm_120, ...).  Resolver / parser tests on synthetic
# devices and synthetic or fake toolkits: control flow only, never evidence that such hardware works.
TC_BW = C.ToolchainInfo(nvcc="/opt/cuda-13/bin/nvcc", version="12.9", version_text="nvcc release 12.9, V12.9.86",
                        real_archs=("sm_80", "sm_86", "sm_89", "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_100f",
                                    "sm_103", "sm_120", "sm_120a"),
                        virtual_archs=("compute_80", "compute_90", "compute_100", "compute_120"),
                        host_compiler="gcc 13.3")
SM100 = C.synthetic("synthetic sm_100", (10, 0), sm_count=148, smem_per_block_optin=227 * 1024)
SM103 = C.synthetic("synthetic sm_103", (10, 3), sm_count=148, smem_per_block_optin=227 * 1024)
SM120 = C.synthetic("synthetic sm_120", (12, 0), sm_count=84, smem_per_block_optin=99 * 1024)
SM121 = C.synthetic("synthetic sm_121", (12, 1), sm_count=48, smem_per_block_optin=99 * 1024)


@pytest.mark.parametrize("arch,cc", [
    ("sm_80", (8, 0)), ("sm_86", (8, 6)), ("sm_89", (8, 9)), ("sm_90", (9, 0)),
    ("sm_100", (10, 0)), ("sm_103", (10, 3)), ("sm_120", (12, 0)), ("sm_121", (12, 1)),
])
def test_parse_arch_splits_major_and_last_digit_minor(arch, cc):
    assert B.parse_arch(arch) == cc and B.arch_name(cc) == arch
    assert C.synthetic("x", cc).arch == arch


def test_parsed_targets_order_numerically_not_lexicographically():
    archs = ["sm_120", "sm_90", "sm_100", "sm_86", "sm_121", "sm_80"]
    assert sorted(archs, key=B.parse_arch) == ["sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "sm_121"]
    assert B.parse_arch("sm_100") > B.parse_arch("sm_90")  # lexicographic order of the strings says otherwise


@pytest.mark.parametrize("bad", ["sm_", "sm_8", "sm_100junk", "sm_86 ", " sm_86", "sm_86\n", "sm86", "SM_86",
                                 "sm_086", "sm_8x", "compute_86", "10.0", "8.6", "sm_100a1", ""])
def test_parse_arch_rejects_incomplete_or_trailing_tokens(bad):
    with pytest.raises(B.BuildError, match="invalid CUDA target"):
        B.parse_arch(bad)


@pytest.mark.parametrize("suffixed", ["sm_90a", "sm_100a", "sm_100f", "sm_120a"])
def test_parse_arch_refuses_feature_suffixes_explicitly(suffixed):
    """'a' / 'f' targets are real but outside the kernels' grammar: refused by name, never stripped."""
    with pytest.raises(B.BuildError, match="architecture-specific"):
        B.parse_arch(suffixed)


def test_compiler_target_lists_keep_exact_tokens():
    out = "sm_80 sm_86\nsm_90 sm_90a sm_100 sm_100a sm_100f sm_103 sm_120 sm_120a junk sm_12x sm_ sm_1000junk\n"
    assert C.parse_gpu_list(out, "sm") == ("sm_80", "sm_86", "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_100f",
                                           "sm_103", "sm_120", "sm_120a")
    assert C.parse_gpu_list("compute_90 compute_90a compute_100 compute_120f sm_90 compute_x\n", "compute") == \
        ("compute_90", "compute_90a", "compute_100", "compute_120f")


def test_toolchain_query_parses_a_multi_digit_major_compiler(monkeypatch, tmp_path):
    """The public toolchain() query on a fake nvcc that lists Blackwell targets, then target
    resolution with the toolchain it returns."""
    fake = tmp_path / "nvcc"
    fake.write_text(f"""#!{sys.executable}
import sys
a = sys.argv[1]
if a == "--version":
    print("Cuda compilation tools, release 12.9, V12.9.86")
elif a == "--list-gpu-code":
    print("sm_75 sm_80 sm_86 sm_90 sm_90a sm_100 sm_100a sm_100f sm_120 sm_120a")
elif a == "--list-gpu-arch":
    print("compute_75 compute_80 compute_86 compute_90 compute_100 compute_120")
else:
    sys.exit(2)
""")
    fake.chmod(0o755)
    monkeypatch.setenv("NVCC", str(fake))
    tc = C.toolchain()
    assert tc.nvcc == str(fake.resolve()) and tc.version == "12.9"
    assert tc.real_archs == ("sm_75", "sm_80", "sm_86", "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_100f", "sm_120",
                             "sm_120a")
    assert tc.virtual_archs[-2:] == ("compute_100", "compute_120")
    assert B.resolve_target(SM100, tc)[0] == "sm_100"
    assert B.resolve_target(SM120, tc)[0] == "sm_120"
    t, why = B.resolve_target(SM121, tc)  # no sm_121 in this toolkit: same-major 12.0 cubin
    assert t == "sm_120" and "binary-compatible" in why
    assert B.resolve_target(SM120, tc, "sm_120")[0] == "sm_120"


@pytest.mark.parametrize("caps,override,want", [
    (SM100, None, "sm_100"),            # native target present
    (SM120, None, "sm_120"),
    (SM103, None, "sm_103"),
    (SM100, "sm_100", "sm_100"),        # explicit overrides on compatible devices
    (SM120, "sm_120", "sm_120"),
    (SM103, "sm_100", "sm_100"),        # same major, older minor
    (SM121, "sm_120", "sm_120"),
    (SM121, None, "sm_120"),            # toolkit lacks sm_121: newest same-major plain target
    (RTX, None, "sm_86"),               # one-digit majors unchanged under the Blackwell-capable toolkit
    (RTX, "sm_80", "sm_80"),
    (C.synthetic("c", (9, 0)), None, "sm_90"),  # sm_90a is listed too, but the plain target is chosen
])
def test_multi_digit_major_target_resolution(caps, override, want):
    assert B.resolve_target(caps, TC_BW, override)[0] == want


def test_multi_digit_major_override_through_the_environment(monkeypatch):
    monkeypatch.setenv(B.ARCH_ENV, "sm_120")
    t, why = B.resolve_target(SM121, TC_BW)
    assert t == "sm_120" and why == f"explicit {B.ARCH_ENV}=sm_120"
    monkeypatch.setenv(B.ARCH_ENV, "sm_120junk")
    with pytest.raises(B.BuildError, match="invalid CUDA target"):
        B.resolve_target(SM120, TC_BW)


@pytest.mark.parametrize("caps,override,tc,match", [
    (SM100, "sm_120", TC_BW, "cannot run"),         # other major
    (SM120, "sm_100", TC_BW, "cannot run"),         # older major: no cross-major cubin
    (SM120, "sm_121", TC_BW, "cannot run"),         # newer minor than the device
    (SM100, "sm_90", TC_BW, "cannot run"),
    (SM103, "sm_103", TC, "cannot emit"),           # compatible, but the toolkit cannot emit it
    (SM120, "sm_120", TC, "cannot emit"),
    (SM100, "sm_100a", TC_BW, "architecture-specific"),  # listed by the toolkit, still refused
    (SM120, "sm_120a", TC_BW, "architecture-specific"),
    (SM100, "sm_100junk", TC_BW, "invalid CUDA target"),
    (SM100, "sm_10", TC_BW, "cannot run"),           # a well-formed token: compute capability 1.0, not 10.0
    (SM120, None, TC, "cannot target"),             # pre-Blackwell toolkit, no same-major target
    (C.synthetic("sm_110", (11, 0)), None, TC_BW, "cannot target"),  # never falls back across majors
    (C.synthetic("no-mma sm_120", (12, 0), features=C.features_for((12, 0)) - {"mma.sync.m16n8k16.bf16"}),
     "sm_120", TC_BW, "lacks mma.sync.m16n8k16.bf16"),
    (C.synthetic("turing", (7, 5)), "sm_75", TC_BW, "sm_80"),
])
def test_multi_digit_major_target_diagnostics(caps, override, tc, match):
    with pytest.raises(B.BuildError, match=match):
        B.resolve_target(caps, tc, override)


def test_ptx_is_never_selected_for_a_newer_device():
    """The toolkit's compute_* list is recorded but never used as a fallback: a device newer than
    every cubin target the toolkit can emit gets an error, not a PTX JIT target."""
    tc = C.ToolchainInfo(**{**TC_BW.__dict__, "real_archs": ("sm_80", "sm_90", "sm_100"),
                            "virtual_archs": ("compute_80", "compute_90", "compute_100", "compute_120")})
    with pytest.raises(B.BuildError, match="cannot target"):
        B.resolve_target(SM120, tc)


def test_build_identity_is_exact_per_target(monkeypatch, tmp_path):
    monkeypatch.setenv(B.CACHE_ENV, str(tmp_path))
    targets = ("sm_86", "sm_90", "sm_100", "sm_103", "sm_120", "sm_121")
    ids = {t: _ident(monkeypatch, target=t, tc=TC_BW) for t in targets}
    assert len({i["library_hash"] for i in ids.values()}) == len(ids)
    for t, i in ids.items():
        assert i["target"] == t and f"-arch={t}" in i["flags"]
        assert B.library_path(B.make_spec("production", t, TC_BW), i).relative_to(tmp_path).parts[0] == t
    for bad, match in (("sm_100a", "architecture-specific"), ("sm_100junk", "invalid"), ("sm_120 ", "invalid"),
                       ("sm_75", "sm_80")):
        with pytest.raises(B.BuildError, match=match):
            B.make_spec("production", bad, TC_BW)


def test_cli_rejects_malformed_targets(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(B.CACHE_ENV, str(tmp_path))
    assert B.main(["--target", "sm_100junk"]) == 1
    assert "invalid CUDA target" in capsys.readouterr().err
    assert B.main(["--target", "sm_100a"]) == 1
    assert "architecture-specific" in capsys.readouterr().err
