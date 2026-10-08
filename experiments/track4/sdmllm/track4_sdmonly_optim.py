"""SDMONLY optimisers: row-wise Adam for the store's value tables, Muon for the body matrices, and their composite.

<claudes_code_comments>
** Function List **
group_of(name, p, body_opt, store_opt) - the optimiser group label of one parameter, by its name and shape
assign_groups(model, body_opt, store_opt) - {parameter name: group label} for every trainable parameter
newton_schulz(G, steps) - the nearest semi-orthogonal matrix to G, by 5 quintic Newton-Schulz iterations
Muon.step() - momentum, Nesterov, orthogonalise, step; one group of 2-D matrices
row_grads(store, slot_range, chunk) - gradient of only the woken value rows: (row ids, U by d gradient)
RowAdam.materialise() - form the row gradients of this step once (the clip and the step both read them)
RowAdam.step() - Adam on the woken rows only, each row on its own clock
Composite - several optimisers behind the interface the S0 loop uses (param_groups, zero_grad, step, state_dict)
build_optimizer(model, lr, ...) - the optimiser for a model; with both options at their defaults it is S0's AdamW
clip_grad_norm_(parameters, max_norm, opt) - torch's global-norm clip, counting the row gradients as well
peak_lr_from_opt_state(sd) - the schedule's learning rate at a checkpoint, read from its optimiser state
selftest() - runs track4_sdmonly_optim_test.py

** Technical Review **
- Importing this file has no side effect: it builds nothing and patches nothing. track4_sdmonly_train.py does the
  patching of the S0 trainer; any other trainer can call build_optimizer and clip_grad_norm_ directly.
- Groups (labels): `adamw_decay` (2-D body matrices: wx, wq[h], readout, the dense control's mlps; weight decay
  0.1), `adamw_nodecay` (the embedding/head and every norm weight), `adamw_store` (the store's sub-keys, and its
  value tables when the store is dense), `muon` (the 2-D body matrices when body_opt is muon), `rowadam_store`
  (the value tables when store_opt is sparse). The rule for a body matrix is S0's rule for its decay group:
  2-D, no "emb" in the name, not under "store.".
- STORE-OPT SPARSE, and how it differs from dense AdamW. Dense AdamW updates every row of a value table every step:
  a row that no token woke still has its two moment averages decayed and still moves by its old momentum. RowAdam
  touches only the rows woken in the step. A row that is not woken gets NO update, NO momentum decay step and NO
  weight decay; its moments and its clock stand still until it is next woken. Each row runs Adam on its own clock
  (bias correction uses the number of steps in which that row was woken), so a row's update is exactly Adam over
  that row's own gradient sequence. Weight decay, if set, is decoupled and applied to woken rows only. Sub-keys and
  query maps stay dense. The forward pass and the gradients to every other parameter are unchanged.
- How the row gradient is formed. With ProductKeyStore.row_grad on, the read is r = sum_c w_c v_rows(c) / (k heads)
  + tap, with v detached and tap a zero leaf. So tap.grad = g = dLoss/dr (N by d), and the gradient of value row m
  is sum over the (token, candidate) pairs that selected m of w g / (k heads). row_grads() adds these up with
  one sparse matmul W^T g on CUDA (W is tokens by woken rows, 2k entries a token), or with index_add_ a few candidate
  columns at a time on other devices, into a U by d tensor (U = distinct woken rows). Nothing of size M by d is made.
  The test checks both against autograd's dense gradient. Measured on the GB10 under load, one store of 262,144 rows
  at d 768: the sparse matmul 25 ms, index_add_ 118 ms, autograd's scatter 80 ms, and AdamW over the table 259 ms.
- The gain depends on the fraction of rows woken in a step. B*T*2k = 524,288 selections a store a step: at 65,536
  rows about a third of the table is woken and the row path does not win; at 262,144 rows about a fifth is.
- The clip counts the row gradients in the global norm, so the clip threshold means what it means on the dense path.
- MUON (Jordan et al. 2024): buf = momentum * buf + (1 - momentum) * grad; u = Nesterov mix; O = Newton-Schulz(u);
  W -= lr * sqrt(max(1, rows / cols)) * O. Its learning rate is its own (--muon-lr, default 0.02) and follows the
  run's schedule through lr_mult = muon_lr / lr. Weight decay on the Muon group is decoupled (default 0).
- Composite.param_groups is the live list of the parts' own group dicts, so the S0 loop's per-step
  `group["lr"] = lr * group["lr_mult"]` reaches every part. state_dict()/load_state_dict() carry every part, so
  checkpoints, resume and the cooldown fork keep the moments.
- NOT covered: DistributedDataParallel averages p.grad of parameters only. The row gradients live outside p.grad,
  so --store-opt sparse is single-process until a trainer all-reduces row_grads() itself. Muon and the groups are
  DDP-safe (they read p.grad).
Docs: TRAINING_SDMONLY_HOW_WE_TRAIN.md
</claudes_code_comments>
"""
import os
import sys

import torch

GROUPS = ("adamw_decay", "adamw_nodecay", "adamw_store", "muon", "rowadam_store")


def group_of(name, p, body_opt="adamw", store_opt="dense"):
    if name.startswith("store."):
        return "rowadam_store" if store_opt == "sparse" and name.endswith(".values") else "adamw_store"
    if p.dim() == 2 and "emb" not in name:
        return "muon" if body_opt == "muon" else "adamw_decay"
    return "adamw_nodecay"


def assign_groups(model, body_opt="adamw", store_opt="dense"):
    return {n: group_of(n, p, body_opt, store_opt) for n, p in model.named_parameters() if p.requires_grad}


def newton_schulz(G, steps=5):
    """Quintic Newton-Schulz with the Muon coefficients; singular values land near 1 (roughly 0.7 to 1.2)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16 if G.is_cuda else torch.float32)
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.t()
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * A @ A
        X = a * X + B @ X
    return (X.t() if tall else X).to(G.dtype)


class Muon:
    def __init__(self, params, lr=0.02, lr_mult=1.0, weight_decay=0.0, momentum=0.95, ns_steps=5):
        self.params = list(params)
        assert all(p.dim() == 2 for p in self.params), "Muon takes 2-D matrices only"
        self.momentum, self.ns_steps = momentum, ns_steps
        self.param_groups = [{"params": self.params, "lr": lr, "lr_mult": lr_mult, "weight_decay": weight_decay, "kind": "muon"}]
        self.bufs = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self):
        g0 = self.param_groups[0]
        lr, wd = g0["lr"], g0["weight_decay"]
        for p, buf in zip(self.params, self.bufs):
            if p.grad is None:
                continue
            buf.lerp_(p.grad, 1 - self.momentum)
            u = newton_schulz(p.grad.lerp(buf, self.momentum), self.ns_steps)
            if wd:
                p.mul_(1 - lr * wd)
            p.add_(u, alpha=-lr * max(1.0, p.size(0) / p.size(1)) ** 0.5)

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    def state_dict(self):
        return {"bufs": self.bufs, "group": {k: v for k, v in self.param_groups[0].items() if k != "params"}}

    def load_state_dict(self, sd):
        for buf, s in zip(self.bufs, sd["bufs"]):
            buf.copy_(s)
        self.param_groups[0].update(sd["group"])


@torch.no_grad()
def row_grads(store, chunk=8, method="auto"):
    """Gradient of the woken rows of store.values for the step just back-propagated: (uniq row ids, U by d_v).
    method: "spmm" (one sparse matmul, CUDA), "index_add" (any device), "auto" (spmm on CUDA, index_add elsewhere)."""
    rows = torch.cat([r.reshape(r.shape[0], -1) for r in store.row_rows], 0)  # (slots*N, heads*c)
    w = torch.cat([x.reshape(x.shape[0], -1) for x in store.row_w], 0).float()
    g = torch.cat([t.grad for t in store.row_taps], 0).float() / (store.k * store.heads)
    uniq, inv = torch.unique(rows, return_inverse=True)
    n, c = rows.shape
    if method == "spmm" or (method == "auto" and g.is_cuda):
        # W is (tokens by woken rows) with c entries a token; the row gradients are W^T g
        crow = torch.arange(0, n * c + 1, c, device=g.device)
        W = torch.sparse_csr_tensor(crow, inv.reshape(-1), w.reshape(-1), (n, uniq.numel()), check_invariants=False)
        return uniq, torch.sparse.mm(W.t(), g)
    G = torch.zeros(uniq.numel(), g.shape[1], device=g.device, dtype=torch.float32)
    for j in range(0, c, chunk):
        G.index_add_(0, inv[:, j:j + chunk].reshape(-1), (w[:, j:j + chunk, None] * g[:, None, :]).reshape(-1, g.shape[1]))
    return uniq, G


class RowAdam:
    """Adam on the woken rows of each store's value table. `stores` are ProductKeyStore modules with row_grad on."""

    def __init__(self, stores, lr=1e-3, lr_mult=1.0, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8):
        self.stores = list(stores)
        self.params = [s.values for s in self.stores]
        self.betas, self.eps = betas, eps
        self.param_groups = [{"params": self.params, "lr": lr, "lr_mult": lr_mult, "weight_decay": weight_decay, "kind": "rowadam"}]
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = [torch.zeros(p.shape[0], device=p.device, dtype=torch.int32) for p in self.params]
        self.cache = None
        self.last_rows = [0] * len(self.params)

    def materialise(self):
        if self.cache is None:
            self.cache = [row_grads(s) if s.row_taps[0].grad is not None else None for s in self.stores]
        return self.cache

    @torch.no_grad()
    def step(self):
        g0 = self.param_groups[0]
        lr, wd, (b1, b2) = g0["lr"], g0["weight_decay"], self.betas
        for i, rg in enumerate(self.materialise()):
            if rg is None:
                continue
            uniq, G = rg
            p, m, v, t = self.params[i], self.m[i], self.v[i], self.t[i]
            tr = t[uniq] + 1
            t.index_copy_(0, uniq, tr)
            mr = m[uniq].mul_(b1).add_(G, alpha=1 - b1)
            vr = v[uniq].mul_(b2).addcmul_(G, G, value=1 - b2)
            m.index_copy_(0, uniq, mr)
            v.index_copy_(0, uniq, vr)
            trf = tr.float()[:, None]
            upd = (mr / (1 - b1 ** trf)) / ((vr / (1 - b2 ** trf)).sqrt_().add_(self.eps))
            pr = p[uniq]
            if wd:
                pr.mul_(1 - lr * wd)
            p.index_copy_(0, uniq, pr.add_(upd, alpha=-lr))
            self.last_rows[i] = int(uniq.numel())

    def zero_grad(self, set_to_none=True):
        self.cache = None
        for s in self.stores:
            for tap in s.row_taps:
                tap.grad = None
        for p in self.params:
            p.grad = None

    def state_dict(self):
        return {"m": self.m, "v": self.v, "t": self.t, "group": {k: v for k, v in self.param_groups[0].items() if k != "params"}}

    def load_state_dict(self, sd):
        for name in ("m", "v", "t"):
            for dst, src in zip(getattr(self, name), sd[name]):
                dst.copy_(src)
        self.param_groups[0].update(sd["group"])


class Composite:
    """Named optimisers behind the interface the S0 loop uses. parts: {label: optimiser}, in a fixed order."""

    def __init__(self, parts):
        self.parts = dict(parts)
        self.rowadam = next((o for o in self.parts.values() if isinstance(o, RowAdam)), None)

    @property
    def param_groups(self):
        # read live: torch's load_state_dict replaces an optimiser's group dicts, so a list built once would go stale
        return [g for o in self.parts.values() for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.parts.values():
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.parts.values():
            o.step()

    def state_dict(self):
        return {"sdmonly_composite": sorted(self.parts), "parts": {k: o.state_dict() for k, o in self.parts.items()}}

    def load_state_dict(self, sd):
        if "sdmonly_composite" not in sd:
            raise ValueError("this checkpoint's optimiser state is plain AdamW; resume it with --store-opt dense --body-opt adamw")
        if sorted(sd["parts"]) != sorted(self.parts):
            raise ValueError(f"optimiser parts differ: checkpoint {sorted(sd['parts'])}, this run {sorted(self.parts)}")
        for k, o in self.parts.items():
            o.load_state_dict(sd["parts"][k])


def build_optimizer(model, lr, store_lr_mult=1.0, store_wd=0.1, body_opt="adamw", store_opt="dense", muon_lr=0.02,
                    muon_wd=0.0, n_tokens=None, betas=(0.9, 0.95), eps=1e-8):
    """The optimiser for `model`. Defaults give the S0 trainer's AdamW (same groups, same order). n_tokens = B*T is
    needed for store_opt sparse (the size of the read taps); the model must already be on its device."""
    by = {g: [] for g in GROUPS}
    for n, p in model.named_parameters():
        if p.requires_grad:
            by[group_of(n, p, body_opt, store_opt)].append(p)
    groups = [{"params": by["adamw_decay"], "weight_decay": 0.1, "lr_mult": 1.0},
              {"params": by["adamw_nodecay"], "weight_decay": 0.0, "lr_mult": 1.0}]
    if by["adamw_store"]:
        groups.append({"params": by["adamw_store"], "weight_decay": store_wd, "lr_mult": store_lr_mult})
    if body_opt == "adamw" and store_opt == "dense":
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
    parts = {"adamw": torch.optim.AdamW([g for g in groups if g["params"]], lr=lr, betas=betas, eps=eps)}
    if by["muon"]:
        parts["muon"] = Muon(by["muon"], lr=muon_lr, lr_mult=muon_lr / lr, weight_decay=muon_wd)
    if by["rowadam_store"]:
        assert n_tokens, "store_opt sparse needs n_tokens = B * T"
        banks = list(model.store.banks)
        for b in banks:
            b.enable_row_grad(n_tokens, slots=model.hops if len(banks) == 1 else 1)
        parts["rowadam"] = RowAdam(banks, lr=lr * store_lr_mult, lr_mult=store_lr_mult, weight_decay=store_wd, betas=betas, eps=eps)
    return Composite(parts)


def clip_grad_norm_(parameters, max_norm, opt=None):
    """torch.nn.utils.clip_grad_norm_ over the parameters' gradients plus the row gradients held by `opt`."""
    ra = getattr(opt, "rowadam", None)
    if ra is None:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    grads = [p.grad for p in parameters if p.grad is not None] + [rg[1] for rg in ra.materialise() if rg is not None]
    total = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
    coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    for g in grads:
        g.mul_(coef.to(g.dtype))
    return total


def peak_lr_from_opt_state(sd):
    """The schedule's learning rate when the checkpoint was written: a group's lr divided by its lr_mult."""
    g = sd["parts"]["adamw"]["param_groups"][0] if "sdmonly_composite" in sd else sd["param_groups"][0]
    return g["lr"] / g.get("lr_mult", 1.0)


def selftest():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import track4_sdmonly_optim_test as T
    return T.run()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print(__doc__)
    print("NEXT -> python3 track4_sdmonly_optim.py --selftest")
