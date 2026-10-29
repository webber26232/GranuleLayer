"""Traced runner trials: xtrace with timestamps (trap entry / exit), runner SigIgn / SigCgt masks, process tree
(pid / ppid / pgid / sid / stat) of the runner's descendants before and after the signal, exit status, survivors."""
import json, os, signal, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import test_sanitizer_runner as T
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)

def tree(root):
    rows = subprocess.run(["ps", "-eo", "pid,ppid,pgid,sid,stat,cmd"], capture_output=True, text=True).stdout.splitlines()[1:]
    procs = {}
    for r in rows:
        f = r.split(None, 5)
        procs[int(f[0])] = dict(pid=int(f[0]), ppid=int(f[1]), pgid=int(f[2]), sid=int(f[3]), stat=f[4], cmd=f[5][:120])
    keep, frontier = [], [root]
    while frontier:
        p = frontier.pop()
        if p in procs:
            keep.append(procs[p]); frontier += [q for q, v in procs.items() if v["ppid"] == p]
    return keep

def masks(pid):
    try:
        st = open(f"/proc/{pid}/status").read().splitlines()
        return {l.split(":")[0]: l.split(":")[1].strip() for l in st if l.startswith(("SigIgn", "SigCgt", "SigBlk"))}
    except FileNotFoundError:
        return None

res = []
for disp in ("dfl", "ign"):
    tmp = out / disp
    import shutil; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
    bindir, wdir, pids = tmp / "bin", tmp / "workloads", tmp / "pids"
    for d in (bindir, wdir, pids): d.mkdir()
    fake = bindir / "compute-sanitizer"; fake.write_text(T.FAKE_SANITIZER.replace("#!PYTHON", f"#!{sys.executable}")); fake.chmod(0o755)
    for wl, script in T.WORKLOADS.items(): (wdir / script).write_text(T.FAKE_WORKLOAD.replace('"WL"', repr(wl)))
    e = dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}", GRANULE_SANITIZER_WORKLOADS=str(wdir),
             FAKE_PIDS=str(pids), FAKE_SAN="*=sleep", FAKE_WL="", PS4="+ ${EPOCHREALTIME} pid=${BASHPID} ")
    e.pop("COMPUTE_SANITIZER", None)
    t0 = time.time()
    p = subprocess.Popen(["bash", "-x", str(T.RUNNER), "--jobs", "4", "--out", str(tmp / "out"), sys.executable, "det"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=e,
                         preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN if disp == "ign" else signal.SIG_DFL))
    while len([f for f in pids.iterdir() if f.name.startswith("wl.")]) < 4: time.sleep(0.05)
    before = tree(p.pid); m = masks(p.pid)
    t_sig = time.time(); p.send_signal(signal.SIGINT)
    try:
        o, err = p.communicate(timeout=8); hung = False; after = None
    except subprocess.TimeoutExpired:
        hung = True; after = tree(p.pid)
        p.send_signal(signal.SIGTERM); o, err = p.communicate(timeout=30)
    time.sleep(0.5)
    kids = [int(x.name.rsplit(".", 1)[1]) for x in pids.iterdir()]
    lines = err.splitlines()
    trap_lines = [l for l in lines if "on_signal" in l or "INTERRUPTED" in l or "interrupted" in l or "note:" in l]
    r = dict(disposition=disp, signal="SIGINT", sent_at_s=round(t_sig - t0, 3), runner_pid=p.pid, runner_signal_masks=m,
             tree_before_signal=before, hung_8s=hung, tree_while_hung=after,
             followed_by="SIGTERM" if hung else None, exit=p.returncode, children=len(kids),
             survivors=[k for k in kids if T.alive(k)], trap_trace=trap_lines[:12])
    (tmp / "xtrace.txt").write_text(err)
    res.append(r)
    print(json.dumps({k: r[k] for k in ("disposition", "runner_signal_masks", "hung_8s", "exit", "survivors")}))
(out / "traced.json").write_text(json.dumps(res, indent=1))
