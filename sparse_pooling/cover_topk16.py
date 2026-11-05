"""Cover-TopK/16: a frozen 16-input support per 16-neuron row group, selected once from an original K64 envelope,
executed compactly at K = 16, C = 16 (docs/cover_topk16.md).

Notation (``CoverTopK16Layer.metadata``):

* original envelope   K64 / C256 (logical record = 256 neurons sharing one 64-input record), S64 candidate sets
                      (every record's 64 inputs are one input supergroup ``feat[q]``), R partition rounds;
* executed            K16 / C16: every valid 16-neuron row group u = (output g, record j, c // 16) of the envelope
                      becomes one compact record of 16 neurons sharing 16 indices.

Semantics (one-time preprocessing; the support stays frozen during ordinary training):

1. importance a[u, j] = sum over the VALID neurons c of row group u of W_pre[u, c, j]^2, FP64, sequential adds over
   c = 0..15 (experimental/local_supports.feature_importance; chunk / order independent);
2. TopK: each valid row group keeps the 16 largest a[u, j] of its record's 64 inputs (ties: lower local coordinate);
3. cover repair: inside every original record each of the 64 candidates must be kept by >= 1 valid row group (a
   UNION constraint -- not equal-frequency quotas); while an input is uncovered the swap of minimum energy loss with a
   donor counted >= 2 times is applied (ties lexicographic; experimental/local_supports.cover_repair); infeasible
   records (m 16 < 64) raise ValueError;
4. every valid neuron keeps exactly 16 trainable weights (its pre-pruning values at the kept coordinates, sorted
   index / value pairs) and its own bias; neuron / output / record / input-set identity and the ORIGINAL per-output
   valid-neuron count (the Mean denominator) are unchanged;
5. optional one-time energy restoration (``normalization="energy"``, experimental/row_normalize): per neuron row
   s_i = min(8, sqrt(E_dense_i / max(E_kept_i, 1e-6 E_dense_i))) on the canonical K64 values BEFORE lowering, weights
   only (bias scaling off), rounded to BF16 once; recorded in ``metadata["normalization"]`` and never re-applied
   (the packed layer has no pre-pruning values to restore from; ``from_envelope`` refuses an envelope already marked
   normalized).

Execution: Mean(ReLU), BF16 inputs / parameters, FP32 accumulation and FP32 ``main_grad``, split (BF16 hi + lo) dZ,
FP32-atomic reductions (not bitwise deterministic; reduction="deterministic" is rejected), recompute-based backward
(no M x H activation tensor).  The K16 dX launch is ``dx_variant="auto"`` by default: the selected tuning profile's
measured tiling inside its validated scope (RTX 3090: mt4_nw8_wg4 for the primary 200M-slot model at M 64 / 512 /
8192), otherwise the legacy launch; ``dx_variant="legacy"`` forces the original launch; explicit
"mt<MT>_nw<NW>_wg<WG>" names are capability-checked.  ``last_info["kgen_dx"]`` records what actually ran.

Nothing here touches CUDA or builds kernels at import; the kgen library is built / loaded on first use.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import torch

from .config import PhysicalPlan
from .experimental.kgen_lowered import Envelope, LoweredLocal, lower_local
from .supergroups import ConnectivityPlan

POLICY = "Cover-TopK"
FAN_IN = 16
SCHEMA = "granule.cover_topk16/1"
PACKED_SCHEMA = "granule.cover_topk16_packed/1"
CKPT_SCHEMA = "granule.k64_checkpoint/1"  # the research K64 envelope checkpoints (experimental/kgx_sparse.py)
NORMALIZATIONS = ("none", "energy")
VALIDATED_MODEL = dict(input_dim=1024, hidden_dim=12_500_000, output_dim=1024, envelope_K=64, envelope_C=256, S=64,
                       R=256, Q=4096, records=49_152, valid_slots=200_000_000)


def tensor_sha256(t: torch.Tensor) -> str:
    """sha256 of the raw bytes (the research checkpoints' weight_sha256 / bias_sha256 use the int16 view)."""
    h = hashlib.sha256()
    v = t.detach().contiguous().view(-1)
    for i in range(0, v.numel(), 1 << 28):
        h.update(v[i:i + (1 << 28)].cpu().numpy().tobytes())
    return h.hexdigest()


class CoverTopK16Layer(LoweredLocal):
    """The named Cover-TopK/16 layer (module docstring).  Build with ``from_envelope`` / ``from_checkpoint``, restore
    with ``load_packed``; ``forward`` accepts [..., D] (leading dims flattened, e.g. batch x time; the flattened
    token count must be a multiple of 8) and returns [..., G]."""

    graph_family = "cover-topk16"
    dx_variant = "auto"

    # ------------------------------------------------------------------ construction
    @classmethod
    @torch.no_grad()
    def from_envelope(cls, weight: torch.Tensor, bias: torch.Tensor, *, input_dim: int, hidden_dim: int,
                      output_dim: int, connectivity: ConnectivityPlan, envelope_C: int = 256,
                      normalization: str = "none", plan: Optional[PhysicalPlan] = None, device=None,
                      dx_variant: str = "auto", envelope: Optional[Envelope] = None) -> "CoverTopK16Layer":
        """Select the Cover-TopK/16 support from the K64 envelope values and lower it.

        ``weight`` [G, NB, envelope_C, 64] / ``bias`` [G, NB, envelope_C] BF16 pre-pruning values in KLayout order
        (experimental/kgen_layout.KLayout(input_dim, hidden_dim, output_dim, 64, envelope_C)); ``connectivity`` the
        envelope's input-supergroup plan (S = 64 candidate sets, R rounds, generation, seed): it regenerates the
        envelope graph deterministically.  The caller's tensors are not modified.  ``envelope`` (advanced) reuses an
        already generated ``Envelope`` of the same values."""
        from .experimental import local_supports as LS
        from .experimental import row_normalize as RN

        if normalization not in NORMALIZATIONS:
            raise ValueError(f"normalization {normalization!r} (one of {NORMALIZATIONS})")
        if connectivity.input_supergroup_dim != 64:
            raise ValueError(f"Cover-TopK/16 selects from S = 64 candidate sets (connectivity S = "
                             f"{connectivity.input_supergroup_dim})")
        plan = plan or PhysicalPlan(grad_precision="split")
        t0 = time.perf_counter()
        env = envelope or Envelope.generate(input_dim, hidden_dim, output_dim, envelope_C, connectivity, weight, bias,
                                            plan=plan, device=device or weight.device)
        if env.graph_stats.get("normalized"):
            raise ValueError("this envelope was already energy-normalized: refusing to apply a second normalization")
        L = env.layout
        dev = env.weight.device
        gen_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        sup = LS.build(POLICY, FAN_IN, L, env.kg_buckets, w=env.weight, graph_seed=connectivity.seed, layout_seed=0,
                       device=dev, Q=int(env.sg_feat.shape[0]))
        em = sup.element_mask(L)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        sel_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        if normalization == "energy":
            w_new, b_new, nrec = RN.apply(env.weight, env.bias, em, L.valid_mask(dev), "energy", scale_bias=False)
            nrec = dict(nrec, applied=True)
        else:
            w_new, b_new = env.weight * em.to(env.weight.dtype), env.bias.clone()
            nrec = dict(treatment="none", applied=False)
        menv = Envelope(L, w_new, b_new, env.sg_feat, env.kg_buckets, env.spec, connectivity, plan, mask=em,
                        graph_stats=dict(env.graph_stats, normalized=normalization == "energy"))
        lay = lower_local(menv, sup, plan, cls=cls)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        low_s = time.perf_counter() - t0
        lay.dx_variant = dx_variant
        stats = {k: v for k, v in sup.stats.items() if k != "selection"}
        lay.cover_meta = dict(
            schema=SCHEMA, policy=POLICY, policy_version=LS.SCHEMA, rng_schema=LS.RNG_SCHEMA, fan_in=FAN_IN,
            envelope=dict(K=64, C=L.block, S=connectivity.input_supergroup_dim, R=connectivity.partition_rounds,
                          Q=int(env.sg_feat.shape[0]), generation=connectivity.generation, seed=connectivity.seed,
                          spec=json.loads(env.spec.to_json())),
            executed=dict(K=FAN_IN, C=16, records_16=lay.graph_stats["records_16"]),
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            valid_slots=int(sup.enabled_slots()), biases=hidden_dim,
            support_fingerprint=sup.fingerprint(), support_stats=stats,
            normalization=nrec, normalization_scale_bias=False,
            construction_seconds=dict(envelope_graph=gen_s, selection=sel_s, normalization_and_lowering=low_s,
                                      note="one-time preprocessing incl. device synchronization; outside training"))
        if lay.cover_meta["valid_slots"] != FAN_IN * hidden_dim:
            raise AssertionError("valid slots != 16 H")
        return lay

    @classmethod
    def from_checkpoint(cls, path, *, normalization: str = "none", plan: Optional[PhysicalPlan] = None,
                        device=None, dx_variant: str = "auto", input_dim: int = 1024, output_dim: int = 1024,
                        verify: bool = True) -> "CoverTopK16Layer":
        """``from_envelope`` on a K64 envelope checkpoint of the research format (CKPT_SCHEMA: {rec, weight, bias,
        mask_bits}; rec carries spec / C / H and the weight / bias sha256, verified when ``verify``).  The path is
        user-supplied; no machine-specific location is assumed."""
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        rec = blob["rec"]
        if rec.get("schema") != CKPT_SCHEMA:
            raise ValueError(f"{path}: schema {rec.get('schema')!r}, expected {CKPT_SCHEMA!r}")
        w, b = blob["weight"], blob["bias"]
        mask_note = "no mask"
        if blob.get("mask_bits") is not None:  # accepted only if it enables every valid slot (an unpruned envelope)
            from .experimental.kgen_layout import KLayout

            bits = blob["mask_bits"]
            m = ((bits[:, None] >> torch.arange(8, dtype=torch.uint8)) & 1).reshape(-1)[:w.numel()].view(w.shape)
            valid = KLayout(input_dim, int(rec["H"]), output_dim, 64, int(rec["C"])).valid_mask("cpu")
            if not bool(m[valid[..., None].expand_as(m)].all()):
                raise ValueError(f"{path}: the checkpoint's mask disables valid slots; Cover-TopK/16 selects from the "
                                 "unpruned envelope values")
            mask_note = "mask present, every valid slot enabled (unpruned envelope)"
            del m, valid
        if verify and (tensor_sha256(w.view(torch.int16)) != rec["weight_sha256"] or
                       tensor_sha256(b.view(torch.int16)) != rec["bias_sha256"]):
            raise ValueError(f"{path}: weight / bias sha256 differ from the checkpoint record")
        sp = rec["spec"]
        if sp["K"] != 64:
            raise ValueError(f"{path}: envelope K = {sp['K']} (expected 64)")
        conn = ConnectivityPlan(sp["S"], sp["R"], sp.get("generation", "balanced_permutation"), sp["seed"])
        dev = torch.device(device or "cuda")
        env = Envelope.generate(input_dim, int(rec["H"]), output_dim, int(rec["C"]), conn, w.to(dev), b.to(dev),
                                plan=plan or PhysicalPlan(grad_precision="split"), device=dev)
        if json.loads(env.spec.to_json()) != sp:
            raise ValueError(f"{path}: checkpoint graph spec {sp} != regenerated {env.spec}")
        del blob, w, b
        lay = cls.from_envelope(env.weight, env.bias, input_dim=input_dim, hidden_dim=int(rec["H"]),
                                output_dim=output_dim, connectivity=conn, envelope_C=int(rec["C"]),
                                normalization=normalization, plan=env.plan, device=dev, dx_variant=dx_variant,
                                envelope=env)
        lay.cover_meta["source_checkpoint"] = dict(weight_sha256=rec["weight_sha256"], bias_sha256=rec["bias_sha256"],
                                                   tag=(rec.get("meta") or {}).get("tag"), verified=verify,
                                                   mask=mask_note)
        return lay

    # ------------------------------------------------------------------ persistence
    def save_packed(self, path) -> dict:
        """Save the frozen packed layer: compact indices / values / bias, envelope map, sets, buckets, graph spec,
        metadata (policy / version / normalization / support fingerprint) and immutable-buffer fingerprints.  The
        optimizer state is saved separately (``optimizer.state_dict()``)."""
        L = self.layout
        blob = dict(schema=PACKED_SCHEMA, meta=self.cover_meta,
                    layout=dict(input_dim=L.input_dim, hidden_dim=L.hidden_dim, output_dim=L.groups, fan_in=L.fan_in,
                                C=L.block),
                    connectivity=json.loads(self.connectivity.to_json()), spec=self.spec.to_json(),
                    envelope_KC=list(self.envelope_KC), fingerprints=self.immutable_fingerprints(),
                    state={k: v.detach().cpu() for k, v in self.state_dict().items()})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, path)
        return dict(path=str(path), fingerprints=blob["fingerprints"])

    @classmethod
    def load_packed(cls, path, *, device=None, plan: Optional[PhysicalPlan] = None,
                    dx_variant: str = "auto") -> "CoverTopK16Layer":
        """Restore a ``save_packed`` file: no support selection and no envelope / research checkpoint needed."""
        from .experimental import kgen_graph as KG

        blob = torch.load(Path(path), map_location="cpu", weights_only=True)
        if blob.get("schema") != PACKED_SCHEMA:
            raise ValueError(f"{path}: schema {blob.get('schema')!r}, expected {PACKED_SCHEMA!r}")
        lo, st = blob["layout"], blob["state"]
        spec = KG.GraphSpec.from_json(blob["spec"])
        dev = torch.device(device or "cuda")
        lay = cls(lo["input_dim"], lo["hidden_dim"], lo["output_dim"], lo["fan_in"], lo["C"],
                  ConnectivityPlan(**blob["connectivity"]), device=dev, plan=plan or PhysicalPlan(grad_precision="split"),
                  graph=(st["indices"], st["sg_feat"], st["kg_buckets"], spec), weights=(st["weight"], st["bias"]))
        lay.register_buffer("env", st["env"].to(dev).contiguous())
        lay.load_state_dict(st)
        lay.envelope_KC = tuple(blob["envelope_KC"])
        lay.cover_meta = blob["meta"]
        lay.dx_variant = dx_variant
        if lay.immutable_fingerprints() != blob["fingerprints"]:
            raise ValueError(f"{path}: immutable buffers differ from their recorded fingerprints")
        return lay

    # ------------------------------------------------------------------ introspection
    @property
    def metadata(self) -> dict:
        return self.cover_meta

    @property
    def normalization(self) -> dict:
        return self.cover_meta["normalization"]

    def extra_repr(self):
        m = self.cover_meta
        e = m["envelope"]
        return (f"Cover-TopK/16: in={m['input_dim']}, hidden={m['hidden_dim']:,}, out={m['output_dim']}, "
                f"envelope K{e['K']}/C{e['C']} S{e['S']}/R{e['R']}, executed K16/C16, "
                f"normalization={m['normalization']['treatment']}, dx_variant={self.dx_variant!r}")
