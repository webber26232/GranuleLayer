"""Fail-closed keyed pairing of per-seed learning records and the preregistered paired decision rule
(research results/INTERACTION_COVERAGE_K16_DX.md section 2.1, corrected by integration review 5.3).

Every observation is indexed by an explicit key -- never by its position in a list:

    ObsKey(graph, data_seed, sr_seed, support, treatment, protocol, steps, backend)

``support`` is the support identity (spec string plus, when recorded, the mask / support fingerprint), ``treatment``
the normalization, ``protocol`` the evaluated test protocol, ``backend`` the executed backend variant (``dense-masked``
for the architecture-quality path).  A cohort (one method) is the set of its observations over the declared seed
pairs (data_seed, sr_seed).

Rules (``Cohort`` / ``paired``):

* expected identities come from the DECLARATION, never from the data (integration review 5.3: the research version
  learned a cohort's identity from its first record, so an internally consistent cohort of the wrong protocol -- or two
  cohorts of the same wrong protocol -- passed).  A comparison declares an ``Experiment`` (graph, protocol / evaluation
  set, update count: the invariants both sides must share) and one ``Method`` per side (support / fingerprint,
  treatment, executed backend: the fields a comparison may vary).  ``Cohort(name, experiment, method)`` fixes the
  expected key identity before the first record;
* a duplicate key is rejected (``DuplicateKey``) -- two files of the same run are never averaged or both counted;
* every observation of a cohort must carry the declared graph / support / treatment / protocol / steps / backend;
  a record of another protocol (or steps, graph, ...) is rejected (``WrongIdentity``), even if every record of the
  cohort agrees with it;
* ``paired`` checks that both cohorts belong to the declared experiment and to their declared methods (a cohort built
  for another experiment is rejected), and that the methods differ only in fields the comparison declares;
* paired differences join the two cohorts by the seed key (data_seed, sr_seed); positions never matter;
* a confirmatory comparison requires the DECLARED seed set exactly, once per side; if either side is missing a seed
  or holds an undeclared one, the comparison is ``incomplete``: an inventory and descriptive rows on the seeds both
  sides hold are emitted, never a verdict;
* n and df come from the validated cohort; family size and alpha are the preregistered ones (an incomplete or
  not-estimable member keeps its share of alpha: nothing is recycled).
"""
from __future__ import annotations

import dataclasses
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


class PairingError(ValueError):
    pass


class DuplicateKey(PairingError):
    pass


class WrongIdentity(PairingError):
    pass


@dataclass(frozen=True)
class Experiment:
    """Invariants shared by every cohort of a comparison (from the manifest / method definitions)."""

    graph: int
    protocol: str
    steps: int


@dataclass(frozen=True)
class Method:
    """Per-method identity (from the method definition): support identity, treatment, executed backend."""

    support: str
    treatment: str
    backend: str = "dense-masked"


@dataclass(frozen=True)
class ObsKey:
    graph: int
    data_seed: int
    sr_seed: int
    support: str
    treatment: str
    protocol: str
    steps: int
    backend: str = "dense-masked"

    def seed(self) -> Tuple[int, int]:
        return (self.data_seed, self.sr_seed)

    def identity(self) -> tuple:
        return (self.graph, self.support, self.treatment, self.protocol, self.steps, self.backend)


def expected_identity(experiment: Experiment, method: Method) -> tuple:
    return (experiment.graph, method.support, method.treatment, experiment.protocol, experiment.steps, method.backend)


def key_of_learn_record(j: dict, support_id: Optional[str] = None, backend: Optional[str] = None,
                        protocol_field: str = "final_test_set") -> ObsKey:
    """ObsKey of an rsn_learn / icdx backend-training record (explicit fields only; nothing inferred from file names).
    ``support_id`` defaults to the record's spec plus its mask fingerprint when present."""
    sup = support_id
    if sup is None:
        sr = j.get("support_record") or {}
        fp = sr.get("mask_sha256") or sr.get("support_fingerprint")
        sup = j["support"] + (f"@{fp[:16]}" if fp else "")
    be = backend or (j.get("backend") or {}).get("variant") or "dense-masked"
    return ObsKey(int(j["graph_seed"]), int(j["data_seed"]), int(j["sr_seed"]), sup, str(j["norm"]),
                  str(j.get(protocol_field)), int(j["steps"]), be)


@dataclass
class Cohort:
    """Observations of one DECLARED method in one DECLARED experiment, keyed by their seed pair."""

    name: str
    experiment: Experiment
    method: Method
    obs: Dict[Tuple[int, int], float] = field(default_factory=dict)
    sources: Dict[Tuple[int, int], str] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.experiment, Experiment) or not isinstance(self.method, Method):
            raise TypeError(f"{self.name}: a cohort needs its declared Experiment and Method (no identity is inferred "
                            "from the records)")

    @property
    def identity(self) -> tuple:
        return expected_identity(self.experiment, self.method)

    def add(self, key: ObsKey, value: float, source: str = "") -> None:
        if key.identity() != self.identity:
            raise WrongIdentity(f"{self.name}: record {source or key} has identity {key.identity()}, declared "
                                f"{self.identity}")
        s = key.seed()
        if s in self.obs:
            raise DuplicateKey(f"{self.name}: seed key {s} twice ({self.sources.get(s)!r} and {source!r})")
        if not (isinstance(value, (int, float)) and math.isfinite(value)):
            raise PairingError(f"{self.name}: non-finite value at {s}")
        self.obs[s] = float(value)
        self.sources[s] = source

    def seeds(self) -> set:
        return set(self.obs)


def build_cohort(name: str, experiment: Experiment, method: Method,
                 records: Iterable[Tuple[ObsKey, float, str]]) -> Cohort:
    c = Cohort(name, experiment, method)
    for k, v, src in records:
        c.add(k, v, src)
    return c


def tq(p: float, df: int) -> float:
    from scipy.stats import t

    return float(t.ppf(p, df))


def inventory(c: Cohort, declared: Iterable[Tuple[int, int]]) -> dict:
    d = set(map(tuple, declared))
    have = c.seeds()
    return dict(name=c.name, declared=sorted(d), present=sorted(have & d), missing=sorted(d - have),
                undeclared=sorted(have - d), complete=have == d)


def check_comparison(a: Cohort, b: Cohort, experiment: Experiment, method_a: Method, method_b: Method) -> None:
    """Both cohorts belong to the declared experiment and methods (raises WrongIdentity otherwise)."""
    for c, m in ((a, method_a), (b, method_b)):
        if c.experiment != experiment:
            raise WrongIdentity(f"{c.name}: cohort of experiment {c.experiment}, comparison declares {experiment}")
        if c.method != m:
            raise WrongIdentity(f"{c.name}: cohort of method {c.method}, comparison declares {m}")


def paired(a: Cohort, b: Cohort, declared: Iterable[Tuple[int, int]], *, experiment: Experiment, method_a: Method,
           method_b: Method, family_size: int, alpha: float, margin: float) -> dict:
    """Paired difference a - b joined by seed key, for the DECLARED experiment and methods (check_comparison first).
    Complete (both sides exactly the declared set): deltas, n, df, Bonferroni one-sided upper bound and verdict.
    Otherwise: status 'incomplete', inventories and descriptive deltas on the common declared seeds, verdict None."""
    check_comparison(a, b, experiment, method_a, method_b)
    declared = sorted(set(map(tuple, declared)))
    ia, ib = inventory(a, declared), inventory(b, declared)
    common = sorted(set(declared) & a.seeds() & b.seeds())
    deltas = {s: a.obs[s] - b.obs[s] for s in common}
    out = dict(a=a.name, b=b.name, experiment=dataclasses.asdict(experiment),
               methods=dict(a=dataclasses.asdict(method_a), b=dataclasses.asdict(method_b)),
               declared_seeds=declared, inventory=dict(a=ia, b=ib),
               deltas={f"{s[0]}/{s[1]}": d for s, d in deltas.items()}, family_size=family_size, alpha=alpha,
               margin=margin)
    vals = [deltas[s] for s in common]
    if len(vals) >= 2:
        m, sd = statistics.mean(vals), statistics.stdev(vals)
        h = tq(0.975, len(vals) - 1) * sd / math.sqrt(len(vals))
        out["descriptive"] = dict(n=len(vals), mean=m, sd=sd, pointwise95=[m - h, m + h])
    elif vals:
        out["descriptive"] = dict(n=1, mean=vals[0], sd=None, pointwise95=None)
    if not (ia["complete"] and ib["complete"]):
        out.update(status="incomplete", verdict=None, upper=None,
                   note="confirmatory verdict withheld: the declared seed set is not present exactly once on both "
                        "sides")
        return out
    n = len(vals)
    df = n - 1
    tcrit = tq(1 - alpha / family_size, df)
    m, sd = statistics.mean(vals), statistics.stdev(vals)
    up = m + tcrit * sd / math.sqrt(n)
    verdict = "improvement" if up < 0 else ("noninferior" if up < margin else "inconclusive")
    out.update(status="complete", n=n, df=df, t_crit=tcrit, mean=m, sd=sd, upper=up, verdict=verdict)
    return out


def not_estimable(a: str, b: str, family_size: int, alpha: float, reason: str) -> dict:
    return dict(a=a, b=b, status="not estimable", verdict=None, family_size=family_size, alpha=alpha, reason=reason,
                note="its share of alpha is not recycled")
