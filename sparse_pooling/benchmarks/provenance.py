"""Source / build / hardware provenance of a measurement (K64 large-C / index-free / L1 study).

``provenance(impl_dir)`` records, for the checkout that holds the implementation under test:

* commit, branch, the FULL ``git status --porcelain`` (tracked and untracked);
* every changed / untracked path classified as ``source`` or ``generated`` (``sparse_pooling/results/``,
  caches): a generated-result change never makes a record "source-dirty";
* ``source_diff``: ``git diff HEAD`` of the changed tracked SOURCE paths (empty when source-clean), and
  the untracked source paths;
* ``source_tree_sha256``: sha256 over (path, sha256 of the working-tree bytes) of every tracked source file
  (results excluded), so two records with the same value ran byte-identical sources even if dirty.

Unlike the earlier ``bench_supergroups.git_dirty`` (any tracked change, results included), ``source_dirty``
is false for a checkout whose only changes are generated results.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

GENERATED_PREFIXES = ("sparse_pooling/results/",)
GENERATED_PARTS = ("__pycache__/", ".pytest_cache/")


def _git(repo, *a) -> str:
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout


def is_generated(path: str) -> bool:
    return path.startswith(GENERATED_PREFIXES) or any(p in path for p in GENERATED_PARTS) or path.endswith(".pyc")


def provenance(impl_dir) -> dict:
    impl = Path(impl_dir).resolve()
    repo = _git(impl, "rev-parse", "--show-toplevel").strip()
    if not repo:
        return dict(error=f"{impl} is not in a git checkout")
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    lines = [ln for ln in status.splitlines() if ln.strip()]
    changed = []
    for ln in lines:
        code, path = ln[:2], ln[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        changed.append(dict(code=code, path=path, kind="generated" if is_generated(path) else "source"))
    src_tracked = [c["path"] for c in changed if c["kind"] == "source" and c["code"] != "??"]
    src_untracked = [c["path"] for c in changed if c["kind"] == "source" and c["code"] == "??"]
    diff = _git(repo, "diff", "HEAD", "--", *src_tracked) if src_tracked else ""
    files = [f for f in _git(repo, "ls-files").splitlines() if f and not is_generated(f)]
    h = hashlib.sha256()
    for f in sorted(files):
        fp = Path(repo) / f
        h.update(f.encode())
        h.update(hashlib.sha256(fp.read_bytes()).hexdigest().encode() if fp.is_file() else b"<missing>")
    return dict(repo=repo, commit=_git(repo, "rev-parse", "HEAD").strip(),
                branch=_git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip(), status_porcelain=lines,
                changed_paths=changed, source_dirty=bool(src_tracked or src_untracked),
                generated_dirty=any(c["kind"] == "generated" for c in changed),
                source_changed_tracked=src_tracked, source_untracked=src_untracked, source_diff=diff,
                source_tree_sha256=h.hexdigest(), source_files=len(files))


def source_file_hashes(impl_dir) -> dict:
    """{path: sha256} of every tracked source file (campaign manifests)."""
    repo = _git(Path(impl_dir).resolve(), "rev-parse", "--show-toplevel").strip()
    out = {}
    for f in sorted(x for x in _git(repo, "ls-files").splitlines() if x and not is_generated(x)):
        fp = Path(repo) / f
        out[f] = hashlib.sha256(fp.read_bytes()).hexdigest() if fp.is_file() else None
    return out


def hardware_state() -> dict:
    q = ("name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,clocks.max.sm,power.draw,power.limit,"
         "persistence_mode,memory.used,memory.total,utilization.gpu")
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=10).stdout.strip().split(", ")
        return dict(zip(q.split(","), r))
    except Exception as e:  # noqa: BLE001
        return dict(error=repr(e))
