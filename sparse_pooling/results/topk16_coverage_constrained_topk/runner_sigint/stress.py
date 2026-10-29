"""Stress the sanitizer runner's signal path with the test-suite fakes. Records, per trial: timestamps, signal,
delay after readiness, exit status, exit latency, and on a hang a process/fd snapshot + xtrace tail."""
import json, os, random, signal, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])  # tests dir
import test_sanitizer_runner as T
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
n = int(sys.argv[3]); mode = sys.argv[4]  # ready | startup
load = int(sys.argv[5]) if len(sys.argv) > 5 else 0
xtrace = os.environ.get("XTRACE") == "1"
rng = random.Random(1234)
hogs = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(load)]

def snapshot(root_pid, f):
    ps = subprocess.run(["ps", "-eo", "pid,ppid,pgid,sid,stat,wchan:20,etimes,cmd"], capture_output=True, text=True).stdout
    f.write(ps)
    for pid in os.listdir("/proc"):
        if not pid.isdigit(): continue
        try:
            fds = {fd: os.readlink(f"/proc/{pid}/fd/{fd}") for fd in os.listdir(f"/proc/{pid}/fd")}
        except Exception:
            continue
        pipes = {k: v for k, v in fds.items() if v.startswith("pipe:")}
        if pipes and "fake" in open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace") or str(pid) == str(root_pid):
            f.write(f"fds {pid}: {pipes}\n")
res = []
try:
    for i in range(n):
        tmp = out / f"t{i}"
        import shutil; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
        bindir, wdir, pids = tmp / "bin", tmp / "workloads", tmp / "pids"
        for d in (bindir, wdir, pids): d.mkdir()
        fake = bindir / "compute-sanitizer"
        fake.write_text(T.FAKE_SANITIZER.replace("#!PYTHON", f"#!{sys.executable}")); fake.chmod(0o755)
        for wl, script in T.WORKLOADS.items():
            (wdir / script).write_text(T.FAKE_WORKLOAD.replace('"WL"', repr(wl)))
        e = dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}", GRANULE_SANITIZER_WORKLOADS=str(wdir),
                 FAKE_PIDS=str(pids), FAKE_SAN="*=sleep", FAKE_WL="")
        e.pop("COMPUTE_SANITIZER", None)
        sig = rng.choice([signal.SIGINT, signal.SIGTERM])
        ign = len(sys.argv) > 6 and sys.argv[6] == "ign"
        cmd = ["bash"] + (["-x"] if xtrace else []) + [str(T.RUNNER), "--jobs", "4", "--out", str(tmp / "out"), sys.executable, "det"]
        if xtrace:
            e["PS4"] = "+ ${EPOCHREALTIME} ${BASHPID} "
        t0 = time.monotonic()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=e, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN if ign else signal.SIG_DFL))
        if mode == "startup":
            time.sleep(rng.uniform(0.0, 0.6)); t_ready = None
        else:
            while len([f for f in pids.iterdir() if f.name.startswith("wl.")]) < 4 and time.monotonic() - t0 < 60:
                time.sleep(0.01)
            t_ready = time.monotonic() - t0
            time.sleep(rng.uniform(0.0, 2.0))
        t_sig = time.monotonic()
        p.send_signal(sig)
        try:
            o, err = p.communicate(timeout=30); hung = False
        except subprocess.TimeoutExpired:
            hung = True
            with open(tmp / "hang_snapshot.txt", "w") as f:
                f.write(f"runner pid {p.pid}\n"); snapshot(p.pid, f)
            p.kill(); o, err = p.communicate()
        lat = time.monotonic() - t_sig
        time.sleep(0.5)
        kids = [int(x.name.rsplit(".", 1)[1]) for x in pids.iterdir()]
        surv = [k for k in kids if T.alive(k)]
        for k in surv:
            try: os.kill(k, 9)
            except ProcessLookupError: pass
        r = dict(trial=i, sigint_ignored_on_entry=ign, sig=sig.name, mode=mode, load=load, ready_s=t_ready, latency_s=round(lat, 3), hung=hung,
                 rc=p.returncode, interrupted_msg="INTERRUPTED" in err, children=len(kids), survivors=surv,
                 stderr_tail=err.strip().splitlines()[-3:] if not xtrace else None)
        if xtrace:
            (tmp / "xtrace.txt").write_text(err)
        res.append(r)
        print(json.dumps(r), flush=True)
        if not hung and not surv: shutil.rmtree(tmp, ignore_errors=True)
finally:
    for h in hogs: h.kill()
bad = [r for r in res if r["hung"] or r["survivors"] or r["rc"] not in (130, 143)]
print(json.dumps(dict(trials=len(res), bad=len(bad))))
