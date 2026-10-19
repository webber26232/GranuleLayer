"""Streamed exact tensor records for repeatability validation (det_training.py).

A record set is a directory with one raw-bytes file per (step, tensor) and a ``manifest.json``:

    {"schema": SCHEMA, "complete": true, "identity": {...}, "provenance": {...},
     "records": {"s1_dW": {"dtype": "torch.float32", "shape": [...], "nbytes": n, "sha1": "...", "order": i}, ...}}

* ``identity`` is the experiment (graph / seeds / precision / optimizer mode / requested plan): a
  comparison against records of a different experiment is refused.
* ``provenance`` is the implementation and environment that wrote the records (code commit, build,
  toolchain, GPU).  It is reported, never required to match: an old-versus-new implementation
  comparison of the same experiment is a legitimate use.

``RecordWriter`` writes into a private temporary directory next to the destination and publishes it
with one ``rename`` after ``finalize()``; an interrupted run leaves no record set behind, and an
existing complete set is never replaced by a partial one.  ``RecordComparer`` checks the manifest
(schema, completeness, identity), the presence and exact size of every record file, dtype / shape and
the full byte content of each compared tensor (streamed in chunks; SHA-1 digests are recorded but
never substitute for the byte comparison), and which records were not visited.

Every structural problem is a failure (``ok`` false), also in characterization mode, where value
differences are measured instead of stopping the run.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

SCHEMA = "granule.det_records/2"
CHUNK = 1 << 26  # bytes per streamed chunk
_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
_NP = {1: np.uint8, 2: np.int16, 4: np.int32, 8: np.int64}


class RecordError(RuntimeError):
    """A record set that cannot be used (absent, incomplete, corrupt, or of another experiment)."""


def key(step: int, name: str) -> str:
    return f"s{step}_{name}"


def step_of(k: str) -> int:
    return int(k[1:k.index("_")])


def bits(t: torch.Tensor) -> torch.Tensor:
    """The raw bits of a tensor as a flat integer tensor of the same element size."""
    t = t.detach().contiguous().reshape(-1)
    return t.view(_INT[t.element_size()])


class RecordWriter:
    def __init__(self, dest, identity: Dict, provenance: Optional[Dict] = None, replace: bool = False):
        self.dest = Path(dest)
        if self.dest.exists() and not replace:
            raise RecordError(f"record directory {self.dest} already exists (refusing to overwrite; "
                              f"remove it or pass replace=True / --replace-records)")
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix=f".{self.dest.name}.partial-", dir=self.dest.parent))
        self.identity = identity
        self.provenance = dict(provenance or {})
        self.replace = replace
        self.records: Dict[str, Dict] = {}
        self.published = False

    def __call__(self, step: int, name: str, t: torch.Tensor):
        k = key(step, name)
        if k in self.records:
            raise RecordError(f"record {k} written twice")
        b = bits(t)
        h = hashlib.sha1()
        nbytes = 0
        per = max(1, CHUNK // b.element_size())
        with open(self.tmp / f"{k}.bin", "wb") as fh:
            for i in range(0, b.numel(), per):
                a = b[i:i + per].cpu().numpy().tobytes()
                fh.write(a)
                h.update(a)
                nbytes += len(a)
        self.records[k] = dict(dtype=str(t.dtype), shape=list(t.shape), nbytes=nbytes, sha1=h.hexdigest(),
                               order=len(self.records))
        return None

    def finalize(self, extra: Optional[Dict] = None) -> Path:
        """Write the manifest and publish the record set atomically."""
        man = dict(schema=SCHEMA, complete=True, identity=self.identity, provenance=self.provenance,
                   records=self.records, **(extra or {}))
        with open(self.tmp / "manifest.json", "w") as fh:
            json.dump(man, fh, indent=1, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        old = None
        if self.dest.exists():  # replace=True: swap only now that the new set is complete
            old = self.dest.with_name(f".{self.dest.name}.old-{os.getpid()}")
            os.rename(self.dest, old)
        os.rename(self.tmp, self.dest)
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
        self.published = True
        return self.dest

    def abort(self):
        if not self.published:
            shutil.rmtree(self.tmp, ignore_errors=True)


def load_manifest(d) -> Dict:
    d = Path(d)
    if not d.is_dir():
        raise RecordError(f"record directory {d} does not exist")
    mf = d / "manifest.json"
    if not mf.exists():
        raise RecordError(f"{d} has no manifest.json (incomplete or not a record set)")
    try:
        man = json.loads(mf.read_text())
    except (OSError, ValueError) as e:
        raise RecordError(f"{mf}: unreadable manifest ({e})") from e
    if not isinstance(man, dict) or man.get("schema") != SCHEMA:
        raise RecordError(f"{mf}: schema {man.get('schema') if isinstance(man, dict) else None!r} != {SCHEMA!r}")
    if man.get("complete") is not True:
        raise RecordError(f"{mf}: record set is not marked complete")
    recs = man.get("records")
    if not isinstance(recs, dict) or not recs:
        raise RecordError(f"{mf}: no records")
    for k, r in recs.items():
        if not all(f in r for f in ("dtype", "shape", "nbytes", "sha1")):
            raise RecordError(f"{mf}: record {k} lacks dtype / shape / nbytes / sha1")
    return man


def identity_mismatch(want: Dict, got: Dict, ignore: Iterable[str] = ()) -> Dict:
    """{field: (reference, current)} of experiment-identity fields that differ (except ``ignore``)."""
    ignore = set(ignore)
    return {f: (got.get(f), want.get(f)) for f in sorted(set(want) | set(got))
            if f not in ignore and want.get(f) != got.get(f)}


class RecordComparer:
    """Exact streamed comparison against a record set.  ``__call__`` returns a result dict per
    tensor with ``bitwise`` and, for any problem, ``error``; ``ok`` is the overall verdict."""

    def __init__(self, d, identity: Optional[Dict] = None, characterize: bool = False, ignore: Iterable[str] = ()):
        """``ignore``: identity fields allowed to differ (characterization only, e.g. a fast run measured
        against deterministic records)."""
        if ignore and not characterize:
            raise RecordError("ignoring experiment-identity fields is only allowed in characterization mode")
        self.d = Path(d)
        self.manifest = load_manifest(self.d)
        self.meta = self.manifest["records"]
        if identity is not None:
            bad = identity_mismatch(identity, self.manifest.get("identity") or {}, ignore)
            if bad:
                raise RecordError(f"records in {self.d} are of another experiment (field: reference, current): {bad}")
        for k, r in self.meta.items():  # every file present with exactly the recorded size
            f = self.d / f"{k}.bin"
            if not f.exists():
                raise RecordError(f"record file {f.name} is missing")
            if f.stat().st_size != r["nbytes"]:
                raise RecordError(f"record file {f.name} has {f.stat().st_size} bytes, manifest says {r['nbytes']}")
        self.characterize = characterize
        self.visited: set = set()
        self.errors: List[str] = []

    def __call__(self, step: int, name: str, t: torch.Tensor) -> Dict:
        k = key(step, name)
        if k in self.visited:
            return self._err(k, "compared twice")
        self.visited.add(k)
        r = self.meta.get(k)
        if r is None:
            return self._err(k, "no reference record (unexpected tensor or step)")
        if r["dtype"] != str(t.dtype) or list(r["shape"]) != list(t.shape):
            return self._err(k, f"reference {r['dtype']} {r['shape']} != current {t.dtype} {list(t.shape)}")
        b = bits(t)
        n = b.numel()
        if n * b.element_size() != r["nbytes"]:
            return self._err(k, f"reference holds {r['nbytes']} bytes, current tensor {n * b.element_size()}")
        per = max(1, CHUNK // b.element_size())
        first, ndiff = None, 0
        st = dict(max_abs=0.0, sq=0.0, ref_sq=0.0, nonfinite_mismatch=0)
        fl = t.is_floating_point()
        with open(self.d / f"{k}.bin", "rb") as fh:
            for i in range(0, n, per):
                a = b[i:i + per]
                raw = fh.read(a.numel() * a.element_size())
                if len(raw) != a.numel() * a.element_size():
                    return self._err(k, f"record truncated at element {i}")
                ref = torch.from_numpy(np.frombuffer(raw, dtype=_NP[a.element_size()]).copy()).to(a.device)
                ne = a != ref
                c = int(ne.sum())
                if c:
                    ndiff += c
                    if first is None:
                        first = i + int(torch.nonzero(ne)[0])
                    if self.characterize and fl:
                        x, y = a.view(t.dtype).double(), ref.view(t.dtype).double()
                        fin = torch.isfinite(x) & torch.isfinite(y)
                        dd = (x - y)[fin]
                        st["max_abs"] = max(st["max_abs"], float(dd.abs().max()) if dd.numel() else 0.0)
                        st["sq"] += float((dd * dd).sum())
                        st["nonfinite_mismatch"] += int((torch.isfinite(x) != torch.isfinite(y)).sum())
                if self.characterize and fl:
                    y = ref.view(t.dtype).double()
                    y = y[torch.isfinite(y)]
                    st["ref_sq"] += float((y * y).sum())
            if fh.read(1):
                return self._err(k, "record has bytes beyond the tensor")
        out = dict(bitwise=ndiff == 0, n=n, n_diff=ndiff)
        if first is not None:
            out["first_index"] = first
            if self.characterize and fl:
                out.update(max_abs=st["max_abs"], rel_l2=(st["sq"] / max(st["ref_sq"], 1e-300)) ** 0.5,
                           rms=(st["sq"] / n) ** 0.5, nonfinite_mismatch=st["nonfinite_mismatch"])
        return out

    def _err(self, k: str, msg: str) -> Dict:
        self.errors.append(f"{k}: {msg}")
        return dict(bitwise=False, error=msg)

    def unvisited(self, steps: Iterable[int]) -> List[str]:
        """Reference records of the given steps that were never compared (incomplete trajectory)."""
        steps = set(steps)
        return sorted(k for k in self.meta if step_of(k) in steps and k not in self.visited)
