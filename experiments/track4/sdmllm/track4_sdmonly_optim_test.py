"""Tests for track4_sdmonly_optim.py and the opt-in schedule helpers of track4_sdmonly_train.py (CPU, seconds).

<claudes_code_comments>
** Function List **
tiny(arm, **kw) - a small SdmOnlyLM on CPU with non-zero value tables
loss_of(m, idx, tgt) - cross-entropy of the model on one batch
t_groups_by_name() - every parameter's optimiser group, by name, for each flag combination
t_default_is_s0_adamw() - with default options build_optimizer returns torch AdamW with S0's three groups
t_row_grad_forward_equal() - row_grad on gives the same logits and the same gradients to every other parameter
t_row_grads_equal_autograd() - the row gradients equal autograd's dense gradient on woken rows; other rows are zero
t_row_grads_shared_store() - the same with one store shared by every hop (one tap per hop)
t_row_grads_spmm_equals_index_add() - CUDA only: the sparse matmul and index_add_ give the same row gradients
t_rowadam_first_step() - woken rows move as dense Adam's first step; rows not woken do not move at all
t_rowadam_differs_from_dense_later() - control: dense AdamW moves a row that is no longer woken, RowAdam does not
t_clip_matches_torch() - the clip's total norm equals torch's on the dense gradients, and the scaling is applied
t_newton_schulz() - singular values land near 1; a tall matrix works; a zero gradient makes no step
t_muon_step() - Muon moves its matrices, and only its matrices
t_composite_lr_and_state() - the loop's lr rule reaches every part; save and load reproduce the next step
t_peak_lr() - the learning rate is read back from both kinds of optimiser state
t_schedules() - the cooldown line and the keep-checkpoint wrapper
run() - run every test, print PASS or FAIL per test
bench(argv) - --bench: dense and sparse (or adamw and muon) steps interleaved in one process on real batches, seconds a step

** Technical Review **
- One control per test function, so a failure names what broke. Run by `python3 track4_sdmonly_optim_test.py`
  or `python3 track4_sdmonly_optim.py --selftest`.
- --bench is the speed instrument for a shared GPU: the two arms take turns step by step in one process, so both see
  the same load, and the minimum and the median step time of each are printed with their ratio. It uses the S0
  loop's own pieces (seeded batches of the train shard, bf16 autocast, the compiled chunked loss, clip, step).
  Example: python3 track4_sdmonly_optim_test.py --bench --d 768 --hops 4 --n-sub 512 --steps 24
- The reference for the row gradient is autograd on an identical model with row_grad off: same seed, same weights.
</claudes_code_comments>
"""
import copy
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track4_sdmonly_models as M  # noqa: E402
import track4_sdmonly_optim as O  # noqa: E402

V, B, T = 300, 3, 20


def tiny(arm="sdmonly", **kw):
    torch.manual_seed(0)
    cfg = {"d": 32, "d_a": 16, "n_sub": 12, "k": 5, "hops": 3, "n_back": 4, "decays": (0.8, 0.97), **kw}
    m = M.build_sdmonly(arm, V, cfg)
    if arm == "sdmonly":
        for b in m.store.banks:
            torch.nn.init.normal_(b.values, std=0.1)
    return m


def batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, V, (B, T), generator=g), torch.randint(0, V, (B * T,), generator=g)


def loss_of(m, idx, tgt):
    return F.cross_entropy(m(idx).reshape(-1, V), tgt)


def t_groups_by_name():
    m = tiny(readout_f=24)
    want = {"emb.weight": "adamw_nodecay", "wx.weight": "adamw_decay", "nf.w": "adamw_nodecay", "rnorm.w": "adamw_nodecay",
            "readout.gate.weight": "adamw_decay", "readout.up.weight": "adamw_decay", "readout.down.weight": "adamw_decay"}
    for h in range(3):
        want.update({f"wq.{h}.weight": "adamw_decay", f"qnorms.{h}.w": "adamw_nodecay",
                     f"store.banks.{h}.keys1": "adamw_store", f"store.banks.{h}.keys2": "adamw_store",
                     f"store.banks.{h}.values": "adamw_store"})
    for j in range(6):
        want[f"fnorms.{j}.w"] = "adamw_nodecay"
    assert O.assign_groups(m) == want, "default groups"
    body = {n for n, g in want.items() if g == "adamw_decay"}
    mu = O.assign_groups(m, body_opt="muon")
    assert {n for n, g in mu.items() if g == "muon"} == body and all(mu[n] == want[n] for n in want if n not in body)
    sp = O.assign_groups(m, body_opt="muon", store_opt="sparse")
    assert {n for n, g in sp.items() if g == "rowadam_store"} == {f"store.banks.{h}.values" for h in range(3)}
    assert all(sp[f"store.banks.{h}.keys{i}"] == "adamw_store" for h in range(3) for i in (1, 2))
    assert sp["emb.weight"] == "adamw_nodecay" and sp["wx.weight"] == "muon" and sp["wq.1.weight"] == "muon"
    dn = O.assign_groups(tiny("sdmonly_dense"), body_opt="muon")
    assert dn["mlps.0.gate.weight"] == "muon" and dn["emb.weight"] == "adamw_nodecay"
    # the built optimiser holds exactly the parameters its labels say
    opt = O.build_optimizer(m, 3e-3, 3.0, 0.0, "muon", "sparse", n_tokens=B * T)
    names = {id(p): n for n, p in m.named_parameters()}
    assert {names[id(p)] for p in opt.parts["muon"].params} == body
    assert {names[id(p)] for p in opt.parts["rowadam"].params} == {f"store.banks.{h}.values" for h in range(3)}
    in_adamw = {names[id(p)] for g in opt.parts["adamw"].param_groups for p in g["params"]}
    assert in_adamw == {n for n, g in sp.items() if g.startswith("adamw")}


def t_default_is_s0_adamw():
    m = tiny()
    opt = O.build_optimizer(m, 3e-3, 3.0, 0.0)
    assert type(opt) is torch.optim.AdamW and len(opt.param_groups) == 3
    assert [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.0, 0.0]
    assert [g["lr_mult"] for g in opt.param_groups] == [1.0, 1.0, 3.0]
    assert not any(b.row_grad for b in m.store.banks), "the default path must not switch row_grad on"


def pair(**kw):
    a, b = tiny(**kw), tiny(**kw)
    b.load_state_dict(a.state_dict())
    shared = len(b.store.banks) == 1
    for bank in b.store.banks:
        bank.enable_row_grad(B * T, slots=b.hops if shared else 1)
    return a, b


def t_row_grad_forward_equal():
    a, b = pair()
    idx, tgt = batch()
    la, lb = loss_of(a, idx, tgt), loss_of(b, idx, tgt)
    assert torch.equal(a(idx), b(idx)), "logits differ"
    la.backward()
    lb.backward()
    pb = dict(b.named_parameters())
    for n, p in a.named_parameters():
        if n.endswith(".values"):
            assert pb[n].grad is None, "autograd must not fill the value table's gradient"
        else:
            assert torch.allclose(p.grad, pb[n].grad, atol=1e-7), n
    b.eval()
    with torch.no_grad():
        assert torch.equal(a.eval()(idx), b(idx)), "eval path differs"


def check_rows(a, b):
    idx, tgt = batch()
    loss_of(a, idx, tgt).backward()
    loss_of(b, idx, tgt).backward()
    for sa, sb in zip(a.store.banks, b.store.banks):
        uniq, G = O.row_grads(sb, chunk=3)
        dense = sa.values.grad
        assert torch.allclose(G, dense[uniq], atol=1e-6, rtol=1e-4), "woken rows differ from autograd"
        rest = torch.ones(dense.shape[0], dtype=torch.bool)
        rest[uniq] = False
        assert float(dense[rest].abs().sum()) == 0.0, "autograd has gradient on a row the row gradients miss"
        assert 0 < uniq.numel() < dense.shape[0] and float(G.abs().sum()) > 0, "vacuous"


def t_row_grads_equal_autograd():
    check_rows(*pair(n_sub=30))
    check_rows(*pair(n_sub=40, heads=2))


def t_row_grads_shared_store():
    check_rows(*pair(n_sub=60, share_store=True))


def t_row_grads_spmm_equals_index_add():
    if not torch.cuda.is_available():
        print("SKIP t_row_grads_spmm_equals_index_add: no CUDA here (the sparse matmul path is CUDA only)")
        return
    for kw in ({"n_sub": 30}, {"n_sub": 40, "heads": 2}, {"n_sub": 60, "share_store": True}):
        _, b = pair(**kw)
        b = b.cuda()
        for bank in b.store.banks:
            bank.enable_row_grad(B * T, slots=b.hops if len(b.store.banks) == 1 else 1)
        idx, tgt = batch()
        loss_of(b, idx.cuda(), tgt.cuda()).backward()
        for sb in b.store.banks:
            u1, G1 = O.row_grads(sb, method="index_add")
            u2, G2 = O.row_grads(sb, method="spmm")
            assert torch.equal(u1, u2) and torch.allclose(G1, G2, atol=1e-6, rtol=1e-4) and float(G1.abs().sum()) > 0, kw


def t_rowadam_first_step():
    a, b = pair()
    idx, tgt = batch()
    ref = torch.optim.Adam([s.values for s in a.store.banks], lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
    ra = O.RowAdam(b.store.banks, lr=1e-2)
    before = [s.values.detach().clone() for s in b.store.banks]
    loss_of(a, idx, tgt).backward()
    loss_of(b, idx, tgt).backward()
    ref.step()
    ra.step()
    for i, (sa, sb) in enumerate(zip(a.store.banks, b.store.banks)):
        uniq = ra.cache[i][0]
        rest = torch.ones(sb.values.shape[0], dtype=torch.bool)
        rest[uniq] = False
        assert torch.allclose(sb.values[uniq], sa.values[uniq], atol=2e-4), "woken rows differ from Adam's first step"
        assert torch.equal(sb.values[rest], before[i][rest]) and float(ra.m[i][rest].abs().sum()) == 0.0
        assert int(ra.t[i][uniq].min()) == 1 and int(ra.t[i][rest].max()) == 0
        assert float((sb.values.detach()[uniq] - before[i][uniq]).abs().max()) > 1e-3, "vacuous: woken rows did not move"


def t_rowadam_differs_from_dense_later():
    a, b = pair()
    da = torch.optim.AdamW([s.values for s in a.store.banks], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0)
    ra = O.RowAdam(b.store.banks, lr=1e-2)
    woke = []
    for seed in (1, 2):
        idx, tgt = batch(seed)
        da.zero_grad()
        ra.zero_grad()
        loss_of(a, idx, tgt).backward()
        loss_of(b, idx, tgt).backward()
        snap_a, snap_b = a.store.banks[0].values.detach().clone(), b.store.banks[0].values.detach().clone()
        da.step()
        ra.step()
        woke.append(set(ra.cache[0][0].tolist()))
    only_first = torch.tensor(sorted(woke[0] - woke[1]))
    assert only_first.numel() > 0, "need a row woken in step 1 and not in step 2"
    assert torch.equal(b.store.banks[0].values[only_first], snap_b[only_first]), "RowAdam moved a row that was not woken"
    assert float((a.store.banks[0].values.detach()[only_first] - snap_a[only_first]).abs().max()) > 1e-4, "control: dense AdamW should move it"


def t_clip_matches_torch():
    a, b = pair()
    idx, tgt = batch()
    opt = O.build_optimizer(b, 3e-3, 3.0, 0.0, store_opt="sparse", n_tokens=B * T)
    (loss_of(a, idx, tgt) * 50).backward()
    (loss_of(b, idx, tgt) * 50).backward()
    wq_before = b.wq[0].weight.grad.clone()
    ta = torch.nn.utils.clip_grad_norm_(a.parameters(), 1.0)
    tb = O.clip_grad_norm_(list(b.parameters()), 1.0, opt)
    assert float(ta) > 1.0, "vacuous: the clip must be active"
    assert abs(float(ta) - float(tb)) < 1e-4 * float(ta), (float(ta), float(tb))
    assert torch.allclose(b.wq[0].weight.grad, a.wq[0].weight.grad, atol=1e-7) and not torch.allclose(b.wq[0].weight.grad, wq_before)
    uniq, G = opt.rowadam.cache[0]
    assert torch.allclose(G, a.store.banks[0].values.grad[uniq], atol=1e-6, rtol=1e-4), "row gradients were not scaled by the clip"
    # no row optimiser: the call is torch's own
    c = tiny()
    (loss_of(c, idx, tgt) * 50).backward()
    assert abs(float(O.clip_grad_norm_(list(c.parameters()), 1.0, None)) - float(ta)) < 1e-4 * float(ta)


def t_newton_schulz():
    torch.manual_seed(0)
    for shape in ((24, 60), (60, 24), (32, 32)):
        s = torch.linalg.svdvals(O.newton_schulz(torch.randn(*shape)))
        assert 0.3 < float(s.min()) and float(s.max()) < 1.6, (shape, float(s.min()), float(s.max()))
    assert float(torch.linalg.svdvals(torch.randn(24, 60)).max()) > 1.6, "control: a raw matrix is not near-orthogonal"
    assert float(O.newton_schulz(torch.zeros(8, 16)).abs().sum()) == 0.0


def t_muon_step():
    m = tiny()
    opt = O.build_optimizer(m, 3e-3, 3.0, 0.0, body_opt="muon", muon_lr=0.02)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    idx, tgt = batch()
    loss_of(m, idx, tgt).backward()
    opt.parts["muon"].step()
    groups = O.assign_groups(m, body_opt="muon")
    for n, p in m.named_parameters():
        moved = not torch.equal(p, before[n])
        assert moved == (groups[n] == "muon"), n
    w = m.wx.weight
    step = (w - before["wx.weight"]).norm()
    assert 0.3 * 0.02 * 32 ** 0.5 < float(step) < 1.6 * 0.02 * 32 ** 0.5, float(step)  # lr * about sqrt(rank)


def t_composite_lr_and_state():
    def make():
        m = tiny()
        return m, O.build_optimizer(m, 3e-3, 3.0, 0.0, "muon", "sparse", muon_lr=0.02, n_tokens=B * T)

    def one(m, opt, seed, lr):
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)
        idx, tgt = batch(seed)
        opt.zero_grad(set_to_none=True)
        loss = loss_of(m, idx, tgt)
        loss.backward()
        O.clip_grad_norm_(list(m.parameters()), 1.0, opt)
        opt.step()
        return float(loss)

    m, opt = make()
    assert all("lr_mult" in g for g in opt.param_groups) and len(opt.param_groups) == 4
    assert abs(opt.parts["muon"].param_groups[0]["lr_mult"] - 0.02 / 3e-3) < 1e-9
    assert opt.parts["rowadam"].param_groups[0]["lr_mult"] == 3.0
    losses = [one(m, opt, s, 3e-3) for s in (1, 2, 3, 4, 5, 6)]
    assert losses[-1] < losses[0] or min(losses) < losses[0], "the composite does not train"
    with tempfile.TemporaryDirectory() as td:
        torch.save({"model": m.state_dict(), "opt": opt.state_dict()}, os.path.join(td, "c.pt"))
        ck = torch.load(os.path.join(td, "c.pt"))
    m2, opt2 = make()
    m2.load_state_dict(ck["model"])
    opt2.load_state_dict(ck["opt"])
    one(m, opt, 7, 1e-3)
    one(m2, opt2, 7, 1e-3)
    for (n, p), q in zip(m.named_parameters(), m2.parameters()):
        assert torch.allclose(p, q, atol=1e-7), f"resume differs at {n}"
    m3, opt3 = make()  # control: without the optimiser state the next step is a different step
    m3.load_state_dict(ck["model"])
    one(m3, opt3, 7, 1e-3)
    assert not torch.allclose(m3.wx.weight, m.wx.weight, atol=1e-7), "control: the optimiser state should matter"
    try:
        opt2.load_state_dict(torch.optim.AdamW(tiny().parameters()).state_dict())
        raise AssertionError("a plain AdamW state must be refused")
    except ValueError:
        pass


def t_peak_lr():
    m = tiny()
    for opt in (O.build_optimizer(m, 3e-3, 3.0, 0.0), O.build_optimizer(m, 3e-3, 3.0, 0.0, "muon", "sparse", n_tokens=B * T)):
        for g in opt.param_groups:
            g["lr"] = 2.5e-3 * g["lr_mult"]
        assert abs(O.peak_lr_from_opt_state(copy.deepcopy(opt.state_dict())) - 2.5e-3) < 1e-12


def t_schedules():
    import argparse
    import track4_sdmonly_train as TR
    lr = TR.cooldown_lr(100, 3e-3)
    assert lr(100, 150, 9.9, 3) == 3e-3 and abs(lr(125, 150, 9.9, 3) - 1.5e-3) < 1e-12 and abs(lr(149, 150, 9.9, 3) - 3e-3 / 50) < 1e-12
    assert all(lr(s, 150, 0, 0) > lr(s + 1, 150, 0, 0) for s in range(100, 149))
    with tempfile.TemporaryDirectory() as td:
        os.environ["SDMLLM_CKPT"] = td
        os.makedirs(os.path.join(td, "r"))
        m = tiny()
        a = argparse.Namespace(B=2, T=10, keep_at=[35, 100], run="r", arm="sdmonly")
        cap = {"model": m, "opt": O.build_optimizer(m, 3e-3), "cfg": {"d": 32}}
        f = TR.with_keep(lambda s, t, p, w: 0.5, a, cap)
        assert [f(s, 9, 1, 1) for s in range(7)] == [0.5] * 7
        got = sorted(os.listdir(os.path.join(td, "r")))
        assert got == ["keep_100.pt", "keep_35.pt"], got  # 35 tokens -> step 2 (rounded up), 100 tokens -> step 5
        ck = torch.load(os.path.join(td, "r", "keep_35.pt"))
        assert ck["step"] == 2 and ck["tokens"] == 40 and set(ck) == {"model", "opt", "step", "cfg", "arm", "tokens"}
        del os.environ["SDMLLM_CKPT"]


TESTS = [t_groups_by_name, t_default_is_s0_adamw, t_row_grad_forward_equal, t_row_grads_equal_autograd,
         t_row_grads_shared_store, t_row_grads_spmm_equals_index_add, t_rowadam_first_step, t_rowadam_differs_from_dense_later, t_clip_matches_torch,
         t_newton_schulz, t_muon_step, t_composite_lr_and_state, t_peak_lr, t_schedules]


def run():
    ok = 0
    for t in TESTS:
        try:
            t()
            ok += 1
            print("PASS", t.__name__)
        except Exception as e:  # report every test, do not stop at the first failure
            print("FAIL", t.__name__, "-", type(e).__name__, e)
    print(f"{ok} of {len(TESTS)} tests pass")
    return ok == len(TESTS)


def bench(argv):
    import argparse
    import json
    import statistics
    import time
    import track4_sdmllm_train_one_arm as S0
    from track4_sdmllm24_fused_ce import ce_chunked
    p = argparse.ArgumentParser()
    p.add_argument("--bench", action="store_true")
    p.add_argument("--d", type=int, default=768)
    p.add_argument("--hops", type=int, default=4)
    p.add_argument("--n-sub", type=int, default=256)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--warm", type=int, default=4, help="steps dropped from the front (compile, allocator)")
    p.add_argument("--arms", default="dense,sparse", help="comma list of dense, sparse, muon, sparse+muon")
    p.add_argument("--no-compile", action="store_true")
    a = p.parse_args(argv)
    dev = S0.pick_device()
    sync = torch.cuda.synchronize if dev.type == "cuda" else (torch.mps.synchronize if dev.type == "mps" else (lambda: None))
    arr, Bb, Tt = S0.load_split("train"), 32, 256
    arms = {}
    for name in a.arms.split(","):
        torch.manual_seed(0)
        m = M.build_sdmonly("sdmonly", S0.V, {"d": a.d, "hops": a.hops, "n_sub": a.n_sub, "k": a.k}).to(dev)
        opt = O.build_optimizer(m, 3e-3, 3.0, 0.0, "muon" if "muon" in name else "adamw", "sparse" if "sparse" in name else "dense",
                                n_tokens=Bb * Tt)
        m.store.collect_stats = False
        f = (lambda mm: (lambda x, t: ce_chunked(mm.hidden(x), mm.emb.weight, t, 2048)))(m)
        arms[name] = {"m": m, "opt": opt, "loss": f if a.no_compile else torch.compile(f), "t": [], "rows": []}
    for step in range(a.steps + a.warm):
        xb = torch.from_numpy(S0.batch_for(step, 0, arr, Bb, Tt)).to(dev)
        for name, r in arms.items():
            sync()
            t0 = time.time()
            for g in r["opt"].param_groups:
                g["lr"] = 3e-3 * g.get("lr_mult", 1.0)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type in ("mps", "cuda")):
                loss = r["loss"](xb[:, :-1], xb[:, 1:])
            r["opt"].zero_grad(set_to_none=True)
            loss.backward()
            O.clip_grad_norm_(list(r["m"].parameters()), 1.0, r["opt"])
            r["opt"].step()
            sync()
            if step >= a.warm:
                r["t"].append(time.time() - t0)
                ra = getattr(r["opt"], "rowadam", None)
                if ra is not None:
                    r["rows"].append(sum(ra.last_rows) / (len(ra.last_rows) * r["m"].M))
            r["last"] = float(loss.detach())
    out = {"d": a.d, "hops": a.hops, "n_sub": a.n_sub, "rows_per_store": a.n_sub ** 2, "steps": a.steps, "stamp": S0.stamp()}
    for name, r in arms.items():
        out[name] = {"s_min": round(min(r["t"]), 4), "s_median": round(statistics.median(r["t"]), 4),
                     "tok_s_at_min": round(Bb * Tt / min(r["t"])), "tok_s_at_median": round(Bb * Tt / statistics.median(r["t"])),
                     "last_loss": round(r["last"], 4)}
        if r["rows"]:
            out[name]["woken_row_fraction"] = round(statistics.mean(r["rows"]), 4)
    names = list(arms)
    for n in names[1:]:
        out[f"speedup_{n}_over_{names[0]}"] = {"at_min": round(min(arms[names[0]]["t"]) / min(arms[n]["t"]), 3),
                                              "at_median": round(statistics.median(arms[names[0]]["t"]) / statistics.median(arms[n]["t"]), 3)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench(sys.argv[1:])
        sys.exit(0)
    sys.exit(0 if run() else 1)
