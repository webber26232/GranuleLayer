"""benchmarks/det_records.py: the record store of the deterministic-training validation must fail on
every structural problem (CPU only; the end-to-end CLI / runner check is test_det_training_cli)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from det_records import SCHEMA, RecordComparer, RecordError, RecordWriter  # noqa: E402

ID = dict(case="T", seed=0, steps=2)


def tensors(step):
    g = torch.Generator().manual_seed(step)
    return {"y": torch.randn(5, 3, generator=g), "W": torch.randn(7, generator=g).bfloat16(),
            "n": torch.tensor([step], dtype=torch.int64)}


def write(d, steps=(1, 2), identity=ID):
    w = RecordWriter(d, identity, dict(commit="abc"))
    for s in steps:
        for n, t in tensors(s).items():
            w(s, n, t)
    w.finalize(dict(resolved_plan={"x": 1}))
    return d


def compare(d, steps=(1, 2), identity=ID, mutate=None, characterize=False):
    c = RecordComparer(d, identity, characterize)
    res = {}
    for s in steps:
        for n, t in tensors(s).items():
            if mutate:
                t = mutate(s, n, t)
            res[(s, n)] = c(s, n, t)
    return c, res


def ok(c, res, steps=(1, 2)):
    return all(r.get("bitwise", False) for r in res.values()) and not c.errors and not c.unvisited(steps)


def test_roundtrip_passes(tmp_path):
    d = write(tmp_path / "r")
    man = json.loads((d / "manifest.json").read_text())
    assert man["schema"] == SCHEMA and man["complete"] and man["provenance"] == {"commit": "abc"}
    c, res = compare(d)
    assert ok(c, res)
    assert not list(tmp_path.glob(".r.partial-*"))  # published by rename, nothing left behind


def test_value_difference_is_found(tmp_path):
    d = write(tmp_path / "r")

    def flip(s, n, t):
        if (s, n) == (2, "y"):
            t = t.clone()
            t.view(-1)[4] = torch.nextafter(t.view(-1)[4], torch.tensor(1e30))
        return t

    c, res = compare(d, mutate=flip)
    r = res[(2, "y")]
    assert r["bitwise"] is False and r["n_diff"] == 1 and r["first_index"] == 4 and "error" not in r
    assert not ok(c, res)
    c, res = compare(d, mutate=flip, characterize=True)
    assert res[(2, "y")]["max_abs"] > 0


def test_missing_reference_key_fails(tmp_path):
    """The original false pass: a tensor without a reference record was reported as equal."""
    d = write(tmp_path / "r")
    c = RecordComparer(d, ID)
    r = c(1, "not_recorded", torch.zeros(3))
    assert r["bitwise"] is False and "no reference record" in r["error"]
    assert c.errors


def test_unvisited_records_fail(tmp_path):
    d = write(tmp_path / "r")
    c, res = compare(d, steps=(1,))  # trajectory stopped after step 1
    assert c.unvisited((1, 2)) == ["s2_W", "s2_n", "s2_y"]
    assert not ok(c, res, steps=(1, 2))


@pytest.mark.parametrize("what", ["dtype", "shape"])
def test_dtype_shape_mismatch_fails(tmp_path, what):
    d = write(tmp_path / "r")
    mut = (lambda s, n, t: t.double() if n == "y" else t) if what == "dtype" else \
        (lambda s, n, t: t.reshape(3, 5) if n == "y" else t)
    c, res = compare(d, mutate=mut)
    assert "error" in res[(1, "y")] and not ok(c, res)


@pytest.mark.parametrize("edit", ["truncate", "extend", "delete"])
def test_record_file_size_checked_at_open(tmp_path, edit):
    d = write(tmp_path / "r")
    f = d / "s2_W.bin"
    if edit == "truncate":
        f.write_bytes(f.read_bytes()[:-1])
    elif edit == "extend":
        f.write_bytes(f.read_bytes() + b"\0")
    else:
        f.unlink()
    with pytest.raises(RecordError):
        RecordComparer(d, ID)


def test_size_consistent_manifest_and_file_edit_is_caught_in_compare(tmp_path):
    """A manifest claiming fewer bytes than the tensor (both edited consistently) still fails."""
    d = write(tmp_path / "r")
    m = json.loads((d / "manifest.json").read_text())
    m["records"]["s1_y"]["nbytes"] -= 4
    (d / "manifest.json").write_text(json.dumps(m))
    f = d / "s1_y.bin"
    f.write_bytes(f.read_bytes()[:-4])
    c, res = compare(d)
    assert "error" in res[(1, "y")] and not ok(c, res)


@pytest.mark.parametrize("edit", ["no_manifest", "incomplete", "schema", "garbage", "no_records", "field"])
def test_corrupt_manifest_rejected(tmp_path, edit):
    d = write(tmp_path / "r")
    mf = d / "manifest.json"
    m = json.loads(mf.read_text())
    if edit == "no_manifest":
        mf.unlink()
    elif edit == "garbage":
        mf.write_text("{not json")
    else:
        if edit == "incomplete":
            m["complete"] = False
        elif edit == "schema":
            m["schema"] = "granule.det_records/1"
        elif edit == "no_records":
            m["records"] = {}
        else:
            del m["records"]["s1_y"]["nbytes"]
        mf.write_text(json.dumps(m))
    with pytest.raises(RecordError):
        RecordComparer(d, ID)


def test_other_experiment_rejected_provenance_not_compared(tmp_path):
    d = write(tmp_path / "r")
    with pytest.raises(RecordError, match="another experiment"):
        RecordComparer(d, dict(ID, seed=1))
    # characterization may ignore named identity fields, strict mode may not
    RecordComparer(d, dict(ID, seed=1), characterize=True, ignore=("seed",))
    with pytest.raises(RecordError):
        RecordComparer(d, dict(ID, seed=1), ignore=("seed",))
    # a different implementation (provenance) of the same experiment is fine
    m = json.loads((d / "manifest.json").read_text())
    m["provenance"] = {"commit": "an older implementation"}
    (d / "manifest.json").write_text(json.dumps(m))
    c, res = compare(d)
    assert ok(c, res)


def test_writer_never_overwrites_or_leaves_partial(tmp_path):
    d = write(tmp_path / "r")
    with pytest.raises(RecordError, match="already exists"):
        RecordWriter(d, ID)
    # interrupted run: nothing published, the complete reference survives
    w = RecordWriter(tmp_path / "s", ID)
    w(1, "y", torch.zeros(2))
    w.abort()
    assert not (tmp_path / "s").exists() and not list(tmp_path.glob(".s.partial-*"))
    w = RecordWriter(d, ID, replace=True)
    w(1, "y", torch.zeros(2))
    w.abort()
    c, res = compare(d)
    assert ok(c, res)
    # replace swaps only a complete set
    w = RecordWriter(d, ID, replace=True)
    w(1, "z", torch.ones(1))
    w.finalize()
    assert sorted(json.loads((d / "manifest.json").read_text())["records"]) == ["s1_z"]


def test_duplicate_record_rejected(tmp_path):
    w = RecordWriter(tmp_path / "r", ID)
    w(1, "y", torch.zeros(2))
    with pytest.raises(RecordError):
        w(1, "y", torch.zeros(2))
    w.abort()
    d = write(tmp_path / "q")
    c = RecordComparer(d, ID)
    t = tensors(1)["y"]
    assert c(1, "y", t)["bitwise"]
    assert "error" in c(1, "y", t)


BENCH = Path(__file__).resolve().parents[1] / "benchmarks"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.skipif(os.environ.get("GRANULE_SKIP_SLOW") == "1", reason="GRANULE_SKIP_SLOW=1")
def test_det_training_cli_selftest(tmp_path):
    """End to end: det_training.py through run_det_training.sh passes a repeatable pair, and every
    corruption negative control (flipped byte, truncated / extra bytes, missing file / key, extra
    key, shape, dtype, incomplete, schema, no manifest, other experiment / plan, short trajectory,
    overwrite) exits nonzero; characterization reports status 'characterization'."""
    r = subprocess.run(["bash", str(BENCH / "run_det_training.sh"), sys.executable, str(tmp_path / "scr"), "selftest",
                        str(tmp_path / "out")], capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "all strict pairs passed" in r.stdout
    assert json.loads((tmp_path / "out" / "selftest_compare.json").read_text())["status"] == "pass"
    for neg in ("flipped_byte", "truncated", "missing_key", "other_plan", "short_trajectory"):
        assert (tmp_path / "out" / f"selftest_{neg}.log").exists()
    shutil.rmtree(tmp_path / "scr", ignore_errors=True)
