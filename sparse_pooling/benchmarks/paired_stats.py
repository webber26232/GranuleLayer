"""Fail-closed keyed pairing of per-seed learning records and the preregistered paired decision rule
(results/INTERACTION_COVERAGE_K16_DX.md section 2.1).

Every observation is indexed by an explicit key -- never by its position in a list:

    ObsKey(graph, data_seed, sr_seed, support, treatment, protocol, steps, backend)

``support`` is the support identity (spec string plus, when recorded, the mask / support fingerprint), ``treatment``
the normalization, ``protocol`` the evaluated test protocol, ``backend`` the executed backend variant (``dense-masked``
for the architecture-quality path).  A cohort (one method) is the set of its observations over the declared seed
pairs (data_seed, sr_seed).

Rules (``Cohort`` / ``paired``):

* a duplicate key is rejected (``DuplicateKey``) -- two files of the same run are never averaged or both counted;
* every observation of a cohort must carry the cohort's graph / support / treatment / protocol / steps / backend;
  a record of another protocol (or steps, graph, ...) is rejected (``WrongIdentity``);
* paired differences join the two cohorts by the seed key (data_seed, sr_seed); positions never matter;
* a confirmatory comparison requires the DECLARED seed set exactly, once per side; if either side is missing a seed
  or holds an undeclared one, the comparison is ``incomplete``: an inventory and descriptive rows on the seeds both
  sides hold are emitted, never a verdict;
* n and df come from the validated cohort; family size and alpha are the preregistered ones (an incomplete or
  not-estimable member keeps its share of alpha: nothing is recycled).
"""
from __future__ import annotations

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
    """Observations of one method (fixed identity), keyed by their seed pair."""

    name: str
    identity: Optional[tuple] = None
    obs: Dict[Tuple[int, int], float] = field(default_factory=dict)
    sources: Dict[Tuple[int, int], str] = field(default_factory=dict)

    def add(self, key: ObsKey, value: float, source: str = "") -> None:
        if self.identity is None:
            self.identity = key.identity()
        elif key.identity() != self.identity:
            raise WrongIdentity(f"{self.name}: record {source or key} has identity {key.identity()}, cohort "
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


def build_cohort(name: str, records: Iterable[Tuple[ObsKey, float, str]]) -> Cohort:
    c = Cohort(name)
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


def paired(a: Cohort, b: Cohort, declared: Iterable[Tuple[int, int]], *, family_size: int, alpha: float,
           margin: float) -> dict:
    """Paired difference a - b joined by seed key.  Complete (both sides exactly the declared set): deltas, n, df,
    Bonferroni one-sided upper bound and verdict.  Otherwise: status 'incomplete', inventories and descriptive
    deltas on the common declared seeds, verdict None."""
    declared = sorted(set(map(tuple, declared)))
    ia, ib = inventory(a, declared), inventory(b, declared)
    common = sorted(set(declared) & a.seeds() & b.seeds())
    deltas = {s: a.obs[s] - b.obs[s] for s in common}
    out = dict(a=a.name, b=b.name, declared_seeds=declared, inventory=dict(a=ia, b=ib),
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
