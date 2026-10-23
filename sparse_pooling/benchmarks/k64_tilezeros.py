"""Structure of the exact zeros of learned (proximal L1) checkpoints: how many complete 16-neuron x 16-input
tiles (the tile-sparse format's unit) and complete neurons are all zero, against the alpha = 0 checkpoint
(where only padding is zero).  A learned support helps the tile format only through all-zero tiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import common  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.experimental import kgx_sparse as SP
    from provenance import provenance

    rows = []
    for p in a.checkpoints:
        rec, w, _, _ = SP.load_checkpoint(p)
        w = w.cuda()
        nz = (w != 0).view(-1, 16, 4, 16)  # [n16, neurons, input blocks, inputs]
        tile_any = nz.any(3).any(1)  # [n16, 4]
        neuron_any = nz.view(-1, 64).any(1)
        rows.append(dict(checkpoint=Path(p).name, tag=rec.get("meta", {}).get("tag"), elements=w.numel(),
                         nonzero=int(nz.sum(dtype=torch.int64)), tiles=tile_any.numel(),
                         nonzero_tiles=int(tile_any.sum()), neuron_slots=neuron_any.numel(),
                         nonzero_neurons=int(neuron_any.sum())))
        print(json.dumps(rows[-1]), flush=True)
        del w, nz
        torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(rows=rows, provenance=provenance(Path(common.__file__).resolve().parent.parent)),
                                      indent=1, default=str))


if __name__ == "__main__":
    main()
