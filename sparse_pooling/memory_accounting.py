"""Unique-storage memory accounting (integration review 5.1).

Two rules:

* storage is counted ONCE: tensors are grouped by their underlying storage (``untyped_storage().data_ptr()``); a view
  or an alias (e.g. the work lists' ``feat`` = the layer's ``sg_feat``) adds nothing, overlapping views of one storage
  count that storage's bytes once (the whole allocation they share, not the sum of the views);
* bytes and logical counts are separate fields: element counts (``elements``) are never added to bytes.

``historical_persistent_bytes`` re-derives the categories of the research records' ``persistent_bytes`` dicts
(research commit f4c538a, benchmarks/icdx_report.py ``memory``) with an EXPLICIT key table.  The research table summed
every numeric field outside four core keys into "index / map / work lists / metadata", which double counted
``index_records_int16`` (the same storage as ``indices``) and ``compact_values`` (the same storage as ``weights``) and
added ``active_tiles`` (a tile COUNT, not bytes).  Corrected graph-0 values: indexed K16 43.8672 MiB (reported
67.7), U16 wide 10.6524 MiB (reported 392.9); measured allocation peaks and timings are unaffected.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import torch

MiB = 1 << 20

# historical persistent_bytes keys -> category; "alias:<key>" = the same storage as <key> (never added); "count" = a
# logical count (not bytes); "info" = descriptive (non-numeric)
HISTORICAL_KEYS = {
    "weights": "weights", "bias": "bias", "main_grad": "main_grad", "adam_moments": "moments",
    "indices": "metadata", "graph_metadata": "metadata", "work_lists_cached_last_runtime": "metadata",
    "canonical_envelope_map_uint8": "metadata", "sparse_metadata": "metadata",
    "index_records_int16": "alias:indices", "compact_values": "alias:weights",
    "active_tiles": "count", "sparse_format": "info",
}
CATEGORIES = ("weights", "bias", "main_grad", "moments", "metadata")


def historical_persistent_bytes(pb: Dict) -> Dict:
    """Category sums of a research ``persistent_bytes`` dict with aliases and counts separated (unknown numeric keys
    raise: every key must be classified explicitly)."""
    out = {c: 0 for c in CATEGORIES}
    aliases, counts, info = {}, {}, {}
    for k, v in pb.items():
        cat = HISTORICAL_KEYS.get(k)
        if cat is None:
            if isinstance(v, (int, float)):
                raise KeyError(f"unclassified numeric memory field {k!r}: add it to HISTORICAL_KEYS")
            info[k] = v
            continue
        if cat.startswith("alias:"):
            src = cat.split(":", 1)[1]
            if src in pb and pb[src] != v:
                raise ValueError(f"{k} = {v} is declared an alias of {src} = {pb[src]} but differs")
            aliases[k] = dict(alias_of=src, bytes=v)
        elif cat == "count":
            counts[k] = v
        elif cat == "info":
            info[k] = v
        else:
            out[cat] += int(v)
    return dict(bytes=out, metadata_mib=out["metadata"] / MiB, aliases_not_added=aliases, counts=counts, info=info)


def unique_storage(tensors: Iterable[torch.Tensor]) -> Dict:
    """Bytes of the distinct storages behind ``tensors`` and their total logical elements (separate fields)."""
    seen = {}
    elements = 0
    views = 0
    for t in tensors:
        if t is None:
            continue
        elements += t.numel()
        st = t.untyped_storage()
        key = (t.device, st.data_ptr())
        if key in seen:
            views += 1
            continue
        seen[key] = st.nbytes()
    return dict(bytes=int(sum(seen.values())), storages=len(seen), elements=int(elements), aliased_views=views)


def layer_storage(groups: Dict[str, List[torch.Tensor]]) -> Dict:
    """Per-category unique storage; a storage already counted in an earlier category is not counted again."""
    counted = set()
    out = {}
    for cat, ts in groups.items():
        fresh = []
        for t in ts:
            if t is None:
                continue
            key = (t.device, t.untyped_storage().data_ptr())
            if key not in counted:
                fresh.append(t)
                counted.add(key)
        u = unique_storage(fresh)
        u["elements"] = int(sum(t.numel() for t in ts if t is not None))
        out[cat] = u
    out["total_bytes"] = int(sum(v["bytes"] for v in out.values() if isinstance(v, dict)))
    return out
