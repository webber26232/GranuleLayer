"""Bounded interaction diagnostic of the width-preserving study (assignment section 7; a validated synthetic
diagnostic, NOT an application-accuracy claim).

Fixed input mixing: the layer reads the raw inputs x ~ N(0, 1) [M, D] directly (no trainable projection that
could route interactions around the connectivity restriction); every output g has its own affine read-out
yhat_g = a_g y_g + c_g (no mixing across outputs; a / c calibrated on one batch, then trained).

Targets (fixed by the graph, identical for every compared support): every valid record (g, j) of output g
(set feat[q], 64 local inputs = 4 local 16-input blocks) carries one term over k inputs, one from each of
k distinct 16-input blocks (a_{g,j,i} = local input 16 i + h(g, j, i) mod 16, h a fixed seeded hash):

    f_k = (ReLU(x_1 + ... + x_k - 1) - mu_k) / sigma_k

(a single-index term: exactly one ReLU neuron that sees the k inputs; mu_k / sigma_k fixed Monte-Carlo
constants).  For k >= 2 a part of its variance is NOT additive across the k blocks: the best additive
approximation sum_i E[f_k | x_i] (closed form, Gaussian inputs) leaves the fraction ``additive_floor_k`` of
the variance unexplained.  A support in which no single neuron sees >= 2 of the k blocks (one-tile
supports) is additive across the blocks, so its normalized MSE cannot go below that floor; a support whose
neurons see all k inputs can approach 0.  The output target is t_g = n_rec(g)^-1/2 sum_j f_k(record j),
order k = 1 + (g mod 4) (order 1 = a linear learnability control, floor 0).  Reported: normalized held-out
MSE per order and the explained share of the interaction variance, 1 - nMSE / floor_k (<= 0: no interaction
captured).  (Earlier designs, products of Gaussians and sign-parity ReLU sums, which are PURE interactions,
were not learned by the dense control within the budget for k >= 3; they are not used.)

Model and training (a REDUCED, labelled diagnostic configuration, identical for every compared support):
D = G = 1024, K = 64, S = 64, R = 256, logical C = 256 and H = 262,144 (n_g = 256: ONE record per output,
so every neuron of an output works on its own record's term; at the full H = 12.5M each term is diluted
over 48 records and the dense control did not learn orders >= 2 within the budget); fresh by-neuron init
(seed 17, zero bias); the support of (policy, rho, score, seed) selected by the same rules from the init
weights (dense masked execution); L1AdamW lr 3e-3 (no L1, stochastic rounding), read-out AdamW lr 1e-2;
1000 updates of M = 512 (seeds 50_000 + i), held-out 4 x 512 tokens (seeds 49_000..49_003).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import common  # noqa: F401
import wps_cases as WC

M_TRAIN = 512
HASH_SEED = 777
TRAIN_SEED, HELD_SEEDS = 50_000, (49_000, 49_001, 49_002, 49_003)


def xbatch(seed, dev, M=M_TRAIN):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(M, WC.D, generator=g).to(dev)


@torch.no_grad()
def target_inputs(lay, terms=1):
    """ids [G, NBv, T, 4] global inputs of every record's T terms, rec_valid [G, NBv], order [G], loc [G, NBv, T, 4]."""
    L = lay.layout
    dev = lay.weight.device
    b = lay.kg_buckets.to(torch.int64)
    NB = L.blocks_per_group
    rb = torch.repeat_interleave(torch.arange(len(b), device=dev), b[:, 3])
    kb = torch.arange(len(rb), device=dev) - torch.repeat_interleave(torch.cumsum(b[:, 3], 0) - b[:, 3], b[:, 3])
    bq = torch.full((L.groups * NB,), -1, dtype=torch.int64, device=dev)
    bq[b[rb, 1] * NB + b[rb, 2] + kb] = b[rb, 0]
    q = bq.view(L.groups, NB)
    valid = q >= 0
    gen = torch.Generator(device="cpu").manual_seed(HASH_SEED)
    off = torch.randint(0, 16, (L.groups, NB, terms, 4), generator=gen).to(dev)
    loc = 16 * torch.arange(4, device=dev)[None, None, None, :] + off  # one input per local 16-block, per term
    ids = lay.sg_feat.to(torch.int64)[q.clamp_min(0)[..., None, None], loc]  # [G, NB, T, 4]
    order = 1 + torch.arange(L.groups, device=dev) % 4
    return ids, valid, order, loc


SIGMA_SEED, SIGMA_N = 4321, 1 << 22
OFFSET = 1.0


def _f(xs, k):
    """Unnormalized term over the first k inputs of xs [..., >= k]."""
    if k == 1:
        return xs[..., 0]
    return torch.relu(xs[..., :k].sum(-1) - OFFSET)


def sigmas(dev):
    """{k: (mu_k, sigma_k, additive_floor_k)} by Monte Carlo (fixed seed); the additive part uses the closed form
    E[ReLU(u + Y - 1)] = (u - 1) Phi((u - 1) / s) + s phi((u - 1) / s), Y ~ N(0, s^2), s^2 = k - 1."""
    g = torch.Generator(device="cpu").manual_seed(SIGMA_SEED)
    z = torch.randn(SIGMA_N, 4, generator=g, dtype=torch.float64).to(dev)
    out = {}
    nrm = torch.distributions.Normal(torch.tensor(0.0, dtype=torch.float64, device=dev),
                                     torch.tensor(1.0, dtype=torch.float64, device=dev))
    for k in (1, 2, 3, 4):
        f = _f(z, k)
        mu, sd = float(f.mean()), float(f.std())
        if k == 1:
            out[k] = (mu, sd, 0.0)
            continue
        s = (k - 1) ** 0.5
        add = torch.zeros_like(f)
        for i in range(k):
            u = (z[:, i] - OFFSET) / s
            add += (z[:, i] - OFFSET) * nrm.cdf(u) + s * torch.exp(nrm.log_prob(u)) - mu
        out[k] = (mu, sd, float(((f - mu - add) ** 2).mean()) / sd ** 2)
    return out


def targets(x, ids, valid, order, sig):
    G, NB, T, _ = ids.shape
    t = torch.zeros(x.shape[0], G, device=x.device)
    for j in range(NB):
        for u in range(T):
            xs = x[:, ids[:, j, u]]  # [M, G, 4]
            p = torch.zeros(x.shape[0], G, device=x.device)
            for k in (1, 2, 3, 4):
                sel = order == k
                p[:, sel] = (_f(xs[:, sel], k) - sig[k][0]) / sig[k][1]
            t += p * valid[:, j][None, :]
    return t / (valid.sum(1).clamp_min(1) * T).sqrt()[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--policy", default="dense")
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--score", default="magnitude")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--terms", type=int, default=1, help="independent terms per record (capacity stress: T > 1 "
                    "terms of the same order with independently drawn inputs; target normalized by sqrt(T n_rec))")
    ap.add_argument("--hidden", type=int, default=262_144, help="H of the diagnostic (default: one C256 record "
                    "per output, n_g = 256; the full-size H dilutes each record's term over 48 records and was not "
                    "learned by the dense control)")
    ap.add_argument("--init", default="fresh", choices=["fresh", "checkpoint"],
                    help="fresh: by-neuron init seed 17, zero bias (any H); checkpoint: the alpha = 0 checkpoint "
                         "(full H only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.experimental import kgx_sparse as SP
    from sparse_pooling.experimental import l1 as L1
    from sparse_pooling.experimental import width_masks as WM
    from sparse_pooling.experimental.kgx_layer import KgxLayer
    from sparse_pooling.supergroups import ConnectivityPlan
    from provenance import provenance

    dev = torch.device("cuda")
    t0 = time.perf_counter()
    ck = WC.ckpt(a.C, a.graph_seed) if a.init == "checkpoint" else None
    if ck:
        rec, w, b, _ = SP.load_checkpoint(ck)
        lay = KgxLayer(WC.D, a.hidden, WC.G, a.C, ConnectivityPlan(WC.S, WC.R, seed=a.graph_seed), variant="idf",
                       tile=128, device=dev, weights=(w, b), keep_indices=False)
        del w, b
    else:
        rec = dict(weight_sha256=None)
        lay = KgxLayer(WC.D, a.hidden, WC.G, a.C, ConnectivityPlan(WC.S, WC.R, seed=a.graph_seed), variant="idf",
                       tile=128, device=dev, seed=WC.INIT_SEED, keep_indices=False)
    lay.enable_main_grad()
    ids, rvalid, order, loc = target_inputs(lay, a.terms)
    sig = sigmas(dev)
    mask, ms = None, None
    res = dict(C=a.C, graph_seed=a.graph_seed, policy=a.policy, rho_target=a.rho, score=a.score, seed=a.seed,
               steps=a.steps, lr_layer=a.lr, lr_readout=1e-2, M_train=M_TRAIN, hidden=a.hidden, init=a.init,
               terms=a.terms,
               checkpoint=dict(path=ck, weight_sha256=rec["weight_sha256"]), hash_seed=HASH_SEED,
               provenance=provenance(Path(common.__file__).resolve().parent.parent),
               target="single-index ReLU terms f_k (module docstring)",
               target_constants={k: dict(mu=v[0], sigma=v[1], additive_floor=v[2]) for k, v in sig.items()})
    if a.policy != "dense":
        ms = WM.build(a.policy, a.rho, lay.weight.detach(), lay.layout, score=a.score, seed=a.seed)
        mask = ms.element_mask(lay.layout)
        with torch.no_grad():
            lay.weight.mul_(mask.to(lay.weight.dtype))
        res["mask_stats"] = {k: v for k, v in ms.stats.items() if "hist" not in k}
        res["mask_sha256"] = SP.mask_fingerprint(mask)
        # neurons that see ALL k inputs of their own record's product (support of the row group)
        L = lay.layout
        m = mask.view(L.groups, L.blocks_per_group, L.block, 64).bool()
        seen = m.gather(3, loc[:, :, None, 0, :].expand(-1, -1, L.block, 4))  # [G, NB, C, 4] (first term)
        k = order[:, None, None, None]
        need = torch.arange(4, device=dev)[None, None, None, :] < k
        allk = ((seen | ~need).all(3)) & L.valid_mask(dev)
        res["neurons_seeing_their_whole_product"] = {f"order{o}": float(allk[order == o].sum((1, 2)).double().mean())
                                                     for o in (1, 2, 3, 4)}
        res["neurons_per_output"] = float(L.valid_mask(dev).sum((1, 2)).double().mean())
    a_g = torch.ones(WC.G, device=dev, requires_grad=True)
    c_g = torch.zeros(WC.G, device=dev, requires_grad=True)
    opt = L1.L1AdamW([lay.weight, lay.bias], lr=a.lr)
    if mask is not None:
        L1.apply_mask_(opt, lay.weight, mask)
    opt2 = torch.optim.AdamW([a_g, c_g], lr=1e-2, weight_decay=0.0)
    with torch.no_grad():  # calibrate the read-out on one batch: centered, scale 0.1 / std (start MSE ~ 1)
        y = lay(xbatch(TRAIN_SEED - 1, dev).bfloat16()).float()
        sd = y.std(0).clamp_min(1e-12)
        a_g.copy_(0.1 / sd)
        c_g.copy_(-0.1 * y.mean(0) / sd)

    def predict(x):
        return a_g * lay(x.bfloat16()).float() + c_g

    @torch.no_grad()
    def evaluate():
        se = torch.zeros(WC.G, device=dev, dtype=torch.float64)
        var = torch.zeros(WC.G, device=dev, dtype=torch.float64)
        n = 0
        for s in HELD_SEEDS:
            x = xbatch(s, dev)
            t = targets(x, ids, rvalid, order, sig)
            se += ((predict(x) - t) ** 2).sum(0).double()
            var += (t ** 2).sum(0).double()
            n += x.shape[0]
        nm = se / var  # normalized MSE per output (E t = 0)
        r = {f"order{o}": float(nm[order == o].mean()) for o in (1, 2, 3, 4)}
        r.update({f"explained_interaction_order{o}": 1.0 - r[f"order{o}"] / sig[o][2] for o in (2, 3, 4)})
        return r

    hist = [dict(step=0, **evaluate())]
    for i in range(a.steps):
        x = xbatch(TRAIN_SEED + i, dev)
        loss = ((predict(x) - targets(x, ids, rvalid, order, sig)) ** 2).mean()
        loss.backward()
        opt2.step()
        opt2.zero_grad(set_to_none=True)
        opt.step()
        if (i + 1) % 50 == 0 or i + 1 == a.steps:
            hist.append(dict(step=i + 1, train=float(loss), **evaluate()))
            print(f"[{a.policy} {a.rho:g}] {i + 1}: " + " ".join(f"{k} {v:.4f}" for k, v in hist[-1].items()
                                                                  if k.startswith("order") or k.startswith("expl")),
                  flush=True)
    res["history"] = hist
    res["final"] = hist[-1]
    res["wall_seconds"] = time.perf_counter() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
