"""Unique-storage memory accounting (sparse_pooling.memory_accounting; integration review 5.1): aliases and views are
counted once, counts are never bytes, every historical field is classified explicitly, and the corrected values of the
research graph-0 records are reproduced (indexed K16 43.8672 MiB, U16 wide 10.6524 MiB of metadata)."""
import pytest
import torch

from sparse_pooling import memory_accounting as MA

# research records (interaction-coverage-k16-dx f4c538a, results/interaction_coverage_k16_dx/raw/confirm, graph 0,
# M 8192, legacy): persistent_bytes as recorded
INDEXED = {"weights": 400031744, "bias": 25001984, "indices": 25001984, "main_grad": 850067456,
           "adam_moments": 850067456, "graph_metadata": 1310720, "work_lists_cached_last_runtime": 7184384,
           "index_records_int16": 25001984, "canonical_envelope_map_uint8": 12500992}
U16_WIDE = {"weights": 400002560, "bias": 25165824, "indices": 0, "main_grad": 850336768, "adam_moments": 850336768,
            "graph_metadata": 1310720, "work_lists_cached_last_runtime": 6713392, "sparse_metadata": 3145728,
            "sparse_format": "tilew16 (16 x 16 tiles)", "compact_values": 400002560, "active_tiles": 781255}


def test_historical_records_corrected():
    a = MA.historical_persistent_bytes(INDEXED)
    assert round(a["metadata_mib"], 4) == 43.8672
    assert a["aliases_not_added"]["index_records_int16"]["alias_of"] == "indices"
    u = MA.historical_persistent_bytes(U16_WIDE)
    assert round(u["metadata_mib"], 4) == 10.6524
    assert u["counts"] == {"active_tiles": 781255} and u["aliases_not_added"]["compact_values"]["alias_of"] == "weights"
    assert u["bytes"]["weights"] == 400002560  # compact_values not added a second time
    # the research table's sum over every numeric non-core field (the erratum)
    core = ("weights", "bias", "main_grad", "adam_moments")
    wrong = lambda pb: sum(v for k, v in pb.items() if k not in core and isinstance(v, (int, float))) / 2 ** 20
    assert round(wrong(INDEXED), 1) == 67.7 and round(wrong(U16_WIDE), 1) == 392.9


def test_unclassified_numeric_field_and_inconsistent_alias_raise():
    with pytest.raises(KeyError, match="unclassified"):
        MA.historical_persistent_bytes(dict(INDEXED, mystery_bytes=5))
    with pytest.raises(ValueError, match="alias"):
        MA.historical_persistent_bytes(dict(INDEXED, index_records_int16=1))


def test_unique_storage_dedups_views_and_aliases_and_separates_counts():
    base = torch.zeros(1000, dtype=torch.int16)
    v1, v2 = base[:500], base[250:]  # overlapping views of one storage
    other = torch.zeros(10, dtype=torch.float32)
    u = MA.unique_storage([base, v1, v2, base.view(10, 100), other])
    assert u["bytes"] == 2000 + 40 and u["storages"] == 2 and u["aliased_views"] == 3
    assert u["elements"] == 1000 + 500 + 750 + 1000 + 10  # logical elements, a separate field
    g = MA.layer_storage(dict(weights=[base], metadata=[v1, other]))
    assert g["weights"]["bytes"] == 2000 and g["metadata"]["bytes"] == 40  # v1's storage already counted
    assert g["total_bytes"] == 2040
