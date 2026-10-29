"""benchmarks/run_sanitizer.sh against a fake compute-sanitizer and minimal workloads (CPU only).

The fake behaves like Compute Sanitizer 2023.1 where it matters: it runs the workload as a child,
prints the tool's summary line in the real formats, returns the workload's exit status, and returns
the --error-exitcode value when it "detects" errors.  Each test injects one fault and asserts the
runner's real process exit status and its per-run summary.tsv.  Nothing here needs NVIDIA tools."""
import csv
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "benchmarks" / "run_sanitizer.sh"
WORKLOADS = {"generic": "sanitize_workload.py", "sg": "sanitize_sg_workload.py",
             "dx_scatter": "sanitize_dx_scatter_workload.py", "det": "sanitize_det_workload.py"}
TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")

pytestmark = pytest.mark.skipif(not (shutil.which("bash") and shutil.which("timeout")),
                                reason="needs bash and coreutils timeout")

FAKE_SANITIZER = r'''#!PYTHON
import os, subprocess, sys, time
args = sys.argv[1:]
if args == ["--version"]:
    print("Compute Sanitizer fake\nVersion 2023.1.0"); sys.exit(0)
VALUED = {"--tool", "--racecheck-report", "--kernel-regex-exclude", "--error-exitcode", "--target-processes",
          "--print-limit"}
opts, i = {}, 0
while i < len(args) and args[i].startswith("--"):
    opts[args[i]] = args[i + 1]; i += 2
prog = args[i:]
tool, errexit = opts["--tool"], int(opts.get("--error-exitcode", "0"))
wl = os.path.basename(prog[-1])
wl = {"sanitize_workload.py": "generic"}.get(wl, wl[len("sanitize_"):-len("_workload.py")])
modes = dict(kv.split("=", 1) for kv in os.environ.get("FAKE_SAN", "").split(",") if kv)
mode = modes.get(f"{wl}.{tool}", modes.get(tool, modes.get("*", "ok")))
open(os.path.join(os.environ["FAKE_PIDS"], f"san.{wl}.{tool}.{os.getpid()}"), "w").close()
summary = ("========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" if tool == "racecheck"
           else "========= ERROR SUMMARY: 0 errors")
print("========= COMPUTE-SANITIZER", flush=True)
if not errexit:
    print("fake: runner did not pass --error-exitcode", flush=True); sys.exit(99)
if mode == "crash":          # sanitizer dies: nonzero, no summary, no traceback
    sys.exit(3)
if mode == "badstart":       # real message for an invalid option, exit 255
    print("the argument for option '--tool' is invalid", flush=True); sys.exit(255)
env = dict(os.environ, FAKE_WL_MODE="sleep") if mode == "sleep" else os.environ
rc = subprocess.run(prog, env=env).returncode
if mode == "exit42":         # the reviewed reproduction: zero summary, then exit 42
    print(summary, flush=True); sys.exit(42)
if mode == "truncated":      # output stops before the summary, exit 0
    sys.exit(0)
if mode == "errors":         # errors reported, exit 0 (as without --error-exitcode)
    print("========= Invalid __global__ write of size 4 bytes", flush=True)
    print("=========     at 0x80 in k(float *)", flush=True)
    print("========= ERROR SUMMARY: 3 errors" if tool != "racecheck" else
          "========= RACECHECK SUMMARY: 3 hazards displayed (3 errors, 0 warnings)", flush=True)
    sys.exit(0)
if mode == "errors_exit":    # errors reported and --error-exitcode honoured
    print("========= ERROR SUMMARY: 2 errors", flush=True); sys.exit(errexit)
if mode == "warnings":       # racecheck warnings only
    print("========= RACECHECK SUMMARY: 2 hazards displayed (0 errors, 2 warnings)", flush=True); sys.exit(0)
if mode == "mixed":          # a failure line AND a zero-error summary
    print("========= Error: Potential RAW hazard detected at __shared__ 0x10 in block (0,0,0) :", flush=True)
    print(summary, flush=True); sys.exit(rc)
if mode == "two_summaries":
    print(summary, flush=True); print(summary, flush=True); sys.exit(rc)
if mode == "other_tool":     # a summary, but of another tool
    print("========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" if tool != "racecheck"
          else "========= ERROR SUMMARY: 0 errors", flush=True)
    sys.exit(rc)
print(summary, flush=True)
sys.exit(rc)
'''

FAKE_WORKLOAD = r'''import os, sys, time
wl = "WL"
modes = dict(kv.split("=", 1) for kv in os.environ.get("FAKE_WL", "").split(",") if kv)
mode = os.environ.get("FAKE_WL_MODE") or modes.get(wl, "ok")
open(os.path.join(os.environ["FAKE_PIDS"], f"wl.{wl}.{os.getpid()}"), "w").close()
print("ok", wl, flush=True)
if mode == "sleep":
    time.sleep(300)
if mode == "traceback":
    raise RuntimeError("workload failure")
if mode == "exit1":            # nonzero without a traceback
    sys.exit(1)
if mode == "incomplete":       # stops early but exits 0
    sys.exit(0)
print(f"WORKLOAD COMPLETE {wl}", flush=True)
'''


@pytest.fixture
def env(tmp_path):
    bindir, wdir, pids = tmp_path / "bin", tmp_path / "workloads", tmp_path / "pids"
    for d in (bindir, wdir, pids):
        d.mkdir()
    fake = bindir / "compute-sanitizer"
    fake.write_text(FAKE_SANITIZER.replace("#!PYTHON", f"#!{sys.executable}"))
    fake.chmod(0o755)
    for wl, script in WORKLOADS.items():
        (wdir / script).write_text(FAKE_WORKLOAD.replace('"WL"', repr(wl)))
    e = dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}", GRANULE_SANITIZER_WORKLOADS=str(wdir),
             FAKE_PIDS=str(pids), FAKE_SAN="", FAKE_WL="")
    e.pop("COMPUTE_SANITIZER", None)
    return dict(env=e, tmp=tmp_path, pids=pids, fake=fake)


def run(env, *workloads, fake_san="", fake_wl="", extra=(), out=None, timeout=120):
    out = out or env["tmp"] / "out"
    e = dict(env["env"], FAKE_SAN=fake_san, FAKE_WL=fake_wl)
    r = subprocess.run(["bash", str(RUNNER), "--jobs", "4", "--out", str(out), *extra, sys.executable, *workloads],
                       capture_output=True, text=True, env=e, timeout=timeout)
    rows = []
    if (out / "summary.tsv").exists():
        rows = list(csv.DictReader(open(out / "summary.tsv"), delimiter="\t"))
    return r, {(row["workload"], row["tool"]): row for row in rows}


def failed(rows):
    return {k for k, v in rows.items() if v["result"] == "FAIL"}


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie left for its parent to reap is not a running process
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[1][0] != "Z"
    except (FileNotFoundError, IndexError):
        return False


def test_all_tools_pass_with_complete_successful_runs(env):
    r, rows = run(env)  # default workload list: all four
    assert r.returncode == 0, r.stdout + r.stderr
    assert set(rows) == {(w, t) for w in WORKLOADS for t in TOOLS}
    assert all(v["result"] == "PASS" and v["exit"] == "0" for v in rows.values())
    assert "16/16 runs passed" in r.stdout


def test_zero_error_summary_then_exit_42_fails(env):
    """The reviewed reproduction: 'ERROR SUMMARY: 0 errors' and exit status 42."""
    r, rows = run(env, "sg", "det", fake_san="sg.memcheck=exit42")
    assert r.returncode == 1
    assert failed(rows) == {("sg", "memcheck")}
    assert rows[("sg", "memcheck")]["exit"] == "42" and "nonzero exit 42" in rows[("sg", "memcheck")]["reason"]
    # every other launched job was collected and passed
    assert len(rows) == 8 and all(v["exit"] == "0" for k, v in rows.items() if k != ("sg", "memcheck"))


@pytest.mark.parametrize("fake_san,fake_wl,reason", [
    ("det.synccheck=crash", "", "nonzero exit 3"),                         # sanitizer dies, no traceback
    ("", "det=exit1", "nonzero exit 1"),                                   # workload exits 1, no traceback
    ("", "det=traceback", "nonzero exit 1"),                               # workload traceback
    ("det.synccheck=errors", "", "ERROR SUMMARY: 3 errors"),               # errors, exit 0
    ("det.synccheck=errors_exit", "", "sanitizer reported errors (exit 97)"),
    ("det.racecheck=errors", "", "3 hazards displayed"),                   # hazards, exit 0
    ("det.racecheck=warnings", "", "0 errors, 2 warnings"),                # warnings are not accepted
    ("det.synccheck=truncated", "", "expected one completed synccheck summary, found 0"),
    ("det.synccheck=mixed", "", "sanitizer diagnostic: ========= Error: Potential RAW hazard"),
    ("det.racecheck=mixed", "", "sanitizer diagnostic"),
    ("det.synccheck=two_summaries", "", "found 2"),
    ("det.memcheck=other_tool", "", "found 0"),
    ("det.racecheck=other_tool", "", "found 0"),
    ("det.memcheck=badstart", "", "nonzero exit 255"),
    ("", "det=incomplete", "workload did not complete"),                   # exit 0, no completion marker
])
def test_each_injected_failure_fails_the_run(env, fake_san, fake_wl, reason):
    r, rows = run(env, "sg", "det", fake_san=fake_san, fake_wl=fake_wl)
    assert r.returncode == 1, r.stdout
    bad = failed(rows)
    assert bad and all(w == "det" for w, _ in bad)
    assert all(reason in rows[k]["reason"] for k in bad), {k: rows[k]["reason"] for k in bad}
    assert all(rows[("sg", t)]["result"] == "PASS" for t in TOOLS)  # statuses of all 8 jobs collected
    assert "FAIL" in r.stdout and str(env["tmp"] / "out") in r.stdout


def test_one_failed_background_job_among_many(env):
    r, rows = run(env, fake_san="dx_scatter.initcheck=crash", extra=("--jobs", "8"))
    assert r.returncode == 1
    assert failed(rows) == {("dx_scatter", "initcheck")} and len(rows) == 16
    assert "15/16 runs passed" in r.stdout


def test_missing_or_unstartable_sanitizer(env, tmp_path):
    e = dict(env["env"], COMPUTE_SANITIZER=str(tmp_path / "nope"))
    r = subprocess.run(["bash", str(RUNNER), sys.executable, "det"], capture_output=True, text=True, env=e, timeout=60)
    assert r.returncode == 2 and "sanitizer not found" in r.stderr
    env["fake"].chmod(0o644)  # present but not executable
    e = dict(env["env"], COMPUTE_SANITIZER=str(env["fake"]))
    r = subprocess.run(["bash", str(RUNNER), sys.executable, "det"], capture_output=True, text=True, env=e, timeout=60)
    assert r.returncode == 2 and "sanitizer not found" in r.stderr
    e = dict(env["env"], PATH="/usr/bin:/bin")  # no compute-sanitizer on PATH
    r = subprocess.run(["bash", str(RUNNER), sys.executable, "det"], capture_output=True, text=True, env=e, timeout=60)
    assert r.returncode == 2 and "sanitizer not found" in r.stderr


def test_usage_errors(env):
    for args in (["--jobs", "0", sys.executable], ["--timeout", "x", sys.executable], [sys.executable, "nosuch"],
                 ["--bogus", sys.executable], []):
        r = subprocess.run(["bash", str(RUNNER), *args], capture_output=True, text=True, env=env["env"], timeout=60)
        assert r.returncode == 2, (args, r.stdout, r.stderr)


def test_stale_successful_log_cannot_satisfy_a_new_run(env):
    out = env["tmp"] / "out"
    r, rows = run(env, "det", out=out)
    assert r.returncode == 0
    assert "ERROR SUMMARY: 0 errors" in (out / "det.synccheck.log").read_text()
    r, rows = run(env, "det", fake_san="det.synccheck=truncated", out=out)
    assert r.returncode == 1 and failed(rows) == {("det", "synccheck")}
    assert "ERROR SUMMARY" not in (out / "det.synccheck.log").read_text()  # truncated before the run
    r, rows = run(env, "det", fake_san="det.synccheck=crash", out=out)
    assert r.returncode == 1 and failed(rows) == {("det", "synccheck")}


def test_timeout_fails_and_kills_the_process_group(env):
    t0 = time.monotonic()
    r, rows = run(env, "det", fake_san="det.racecheck=sleep", extra=("--timeout", "3"))
    assert time.monotonic() - t0 < 60
    assert r.returncode == 1 and failed(rows) == {("det", "racecheck")}
    assert rows[("det", "racecheck")]["exit"] == "124" and "timed out" in rows[("det", "racecheck")]["reason"]
    pids = [int(p.name.rsplit(".", 1)[1]) for p in env["pids"].iterdir() if ".racecheck." in p.name or
            p.name.startswith("wl.det.")]
    time.sleep(0.5)
    assert pids and not any(alive(p) for p in pids)


def _fake_pids(env):
    return [int(f.name.rsplit(".", 1)[1]) for f in env["pids"].iterdir()]


def _start_interruptible(env, sigint):
    """The runner on 4 sleeping fake jobs, with SIGINT's disposition set EXPLICITLY (SIG_DFL or SIG_IGN): an inherited
    ignored SIGINT (pytest started with '&' from a non-interactive shell) cannot be trapped by bash, which made the
    interrupt test depend on how the suite was launched (results/TOPK16_COVERAGE_CONSTRAINED_TOPK.md section 2.3)."""
    e = dict(env["env"], FAKE_SAN="*=sleep")
    p = subprocess.Popen(["bash", str(RUNNER), "--jobs", "4", "--out", str(env["tmp"] / "out"), sys.executable,
                          "det"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=e,
                         preexec_fn=lambda: signal.signal(signal.SIGINT, sigint))
    deadline = time.monotonic() + 30
    while len([f for f in env["pids"].iterdir() if f.name.startswith("wl.")]) < 4:
        assert time.monotonic() < deadline, "fake jobs did not start"
        time.sleep(0.1)
    return p


def _teardown(env, p):
    """Bounded cleanup of a runner that did not finish: SIGTERM (its own handler kills its job groups), a grace
    period, then SIGKILL of the runner and of THIS test's recorded fake processes only, and reap."""
    if p.poll() is None:
        p.terminate()
        try:
            p.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
    for x in _fake_pids(env):
        if alive(x):
            os.kill(x, signal.SIGKILL)


@pytest.mark.parametrize("sig,code", [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_interrupt_terminates_children_and_exits_nonzero(env, sig, code):
    p = _start_interruptible(env, signal.SIG_DFL)
    try:
        p.send_signal(sig)
        _, err = p.communicate(timeout=30)
    finally:
        _teardown(env, p)
    assert p.returncode == code and "INTERRUPTED" in err
    pids = _fake_pids(env)
    time.sleep(0.5)
    assert len(pids) == 8 and not any(alive(x) for x in pids)  # 4 sanitizers + 4 workloads


def test_sigint_ignored_on_entry_is_reported_and_sigterm_still_cleans_up(env):
    """The cause of the K8-study suite failure: a runner whose SIGINT is ignored on entry ignores SIGINT (bash cannot
    trap it).  It must say so at startup, keep running, and still terminate every child on SIGTERM."""
    p = _start_interruptible(env, signal.SIG_IGN)
    try:
        p.send_signal(signal.SIGINT)
        time.sleep(2.5)  # the trap would have run within ~1.5 s
        assert p.poll() is None
        p.send_signal(signal.SIGTERM)
        _, err = p.communicate(timeout=30)
    finally:
        _teardown(env, p)
    assert "SIGINT was ignored when this runner started" in err
    assert p.returncode == 143 and "INTERRUPTED" in err
    pids = _fake_pids(env)
    time.sleep(0.5)
    assert len(pids) == 8 and not any(alive(x) for x in pids)
