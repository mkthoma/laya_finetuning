import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("laya")
from laya.common import proper_reward  # noqa: E402

from laya_poc.loss import LossParts, upstream_loss  # noqa: E402

pytestmark = pytest.mark.torch


def _upstream_block(logits, act, batch, sigma, device, GROUP_SIZE=4, GRAD_ACCUM=1):
    """Inline copy of upstream_nb/cell08.py:152-173 (only `batch`/`device` plumbing kept)."""
    logits = logits.float()
    mask = batch["marker_mask"].to(device)
    k = mask.sum(-1, keepdim=True).float()
    target = batch["target"].to(device)

    eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)

    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()
    return loss, loss_ce, loss_rl, r


def _batch(ks=(10, 7, 2), qtypes=(0, 0, 0), seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    kmax = max(ks)
    mask = torch.zeros((len(ks), kmax), dtype=torch.bool)
    target = torch.zeros((len(ks), kmax))
    for i, k in enumerate(ks):
        mask[i, :k] = True
        t = torch.rand(k, generator=g) + 0.05
        target[i, :k] = t / t.sum()
    logits = torch.randn((len(ks), kmax), generator=g).masked_fill(~mask, -1e4).to(dtype).requires_grad_(True)
    act = torch.randn((len(ks), 2), generator=g).requires_grad_(True)
    return logits, act, {"marker_mask": mask, "target": target, "qtype": torch.tensor(qtypes)}


def _grads(*tensors):
    return [t.grad.clone() for t in tensors]


@pytest.mark.parametrize("qtypes,acc,sigma", [((0, 0, 0), 1, 0.4), ((0, 1, 0), 4, 0.1), ((1, 1, 1), 2, 0.3)])
def test_bit_for_bit_equal_to_upstream_block(qtypes, acc, sigma):
    logits, act, b = _batch(qtypes=qtypes)
    torch.manual_seed(1234)
    up_total, up_ce, up_rl, up_r = _upstream_block(logits, act, b, sigma, "cpu", GRAD_ACCUM=acc)
    up_total.backward()
    up_grads = _grads(logits, act)
    logits.grad = act.grad = None

    torch.manual_seed(1234)
    parts = upstream_loss(logits, act, b["marker_mask"], b["target"], b["qtype"], sigma, grad_accum=acc)
    parts.total.backward()

    assert isinstance(parts, LossParts)
    assert torch.equal(parts.total, up_total)
    assert torch.equal(parts.ce, up_ce) and torch.equal(parts.rl, up_rl)
    assert torch.equal(parts.reward, up_r.mean())
    for ours, theirs in zip(_grads(logits, act), up_grads):
        assert torch.equal(ours, theirs)


def test_noise_comes_from_the_global_rng():
    logits, act, b = _batch()
    torch.manual_seed(7)
    a = upstream_loss(logits, act, b["marker_mask"], b["target"], b["qtype"], 0.4)
    torch.manual_seed(7)
    same = upstream_loss(logits, act, b["marker_mask"], b["target"], b["qtype"], 0.4)
    other = upstream_loss(logits, act, b["marker_mask"], b["target"], b["qtype"], 0.4)
    assert torch.equal(a.rl, same.rl) and not torch.equal(a.rl, other.rl)
    assert torch.equal(a.ce, other.ce)  # CE is noise-free


def test_parts_are_consistent_and_ce_matches_soft_cross_entropy():
    logits, act, b = _batch()
    parts = upstream_loss(logits, act, b["marker_mask"], b["target"], b["qtype"], 0.4, ce_weight=0.5, grad_accum=4)
    assert parts.total.item() == pytest.approx((parts.rl.item() + 0.5 * parts.ce.item()) / 4, rel=1e-6)
    lp = torch.log_softmax(logits.detach().masked_fill(~b["marker_mask"], -1e4), -1)
    assert parts.ce.item() == pytest.approx(-(b["target"] * lp).sum(-1).mean().item(), rel=1e-6)
    assert parts.reward.requires_grad is False


def test_half_logits_are_upcast():
    logits, act, b = _batch(dtype=torch.float16)
    parts = upstream_loss(logits, act, b["marker_mask"], b["target"], b["qtype"], 0.4)
    assert parts.total.dtype == torch.float32


@pytest.mark.parametrize("kw,match", [(dict(sigma=0.0), "sigma"), (dict(grad_accum=0), "grad_accum"),
                                      (dict(group_size=1), "group_size")])
def test_rejects_bad_hyperparameters(kw, match):
    logits, act, b = _batch()
    args = dict(sigma=0.4) | kw
    sigma = args.pop("sigma")
    with pytest.raises(ValueError, match=match):
        upstream_loss(logits, act, b["marker_mask"], b["target"], b["qtype"], sigma, **args)


def test_rejects_non_bool_mask_and_shape_mismatch():
    logits, act, b = _batch()
    with pytest.raises(TypeError, match="bool"):
        upstream_loss(logits, act, b["marker_mask"].long(), b["target"], b["qtype"], 0.4)
    with pytest.raises(ValueError, match="shape"):
        upstream_loss(logits, act, b["marker_mask"], b["target"][:, :3], b["qtype"], 0.4)


def test_gradients_reach_encoder_and_head_and_act_head_gets_zero_grads(tiny_ckpt_dir):
    import laya
    from laya.common import collate_items

    from laya_poc.items import build_items
    from synth import rows_from_records, synthetic_records

    agent = laya.load(str(tiny_ckpt_dir), device="cpu")
    model, tok = agent.model, agent.tok
    model.train()
    rows = rows_from_records(synthetic_records(4, seed=3), "train")
    items = [it for r in rows for it in build_items(tok, r, 512, 192)]
    b = collate_items([items], tok.pad_token_id)
    torch.manual_seed(0)
    logits, act = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    parts = upstream_loss(logits, act, b["marker_mask"], b["target"], b["qtype"], 0.4, grad_accum=2)
    parts.total.backward()

    grads = {n: p.grad for n, p in model.named_parameters()}
    assert all(g is not None for g in grads.values()), [n for n, g in grads.items() if g is None]
    enc = [g for n, g in grads.items() if n.startswith("encoder.")]
    assert enc and any(g.abs().sum() > 0 for g in enc)
    assert grads["scorer.3.weight"].abs().sum() > 0
    act_grads = [g for n, g in grads.items() if n.startswith("act_head.")]
    assert act_grads and all(torch.count_nonzero(g) == 0 for g in act_grads)
    assert torch.isfinite(parts.total) and parts.rl.item() != 0.0


def _val_items(tok, n=23):
    from laya_poc.items import build_items
    from synth import rows_from_records, synthetic_records
    rows = rows_from_records(synthetic_records(n, seed=5), "val")
    return [it for r in rows for it in build_items(tok, r, 512, 192)]


def test_evaluate_items_batches_by_length_and_is_order_and_batch_size_invariant(tiny_ckpt_dir, monkeypatch):
    import math

    import laya

    from laya_poc import loss as LS

    agent = laya.load(str(tiny_ckpt_dir), device="cpu")
    model, pad = agent.model, agent.tok.pad_token_id
    items = _val_items(agent.tok)
    seen, real = [], LS.collate_items

    def spy(groups, pad_id):
        seen.append([len(it["ids"]) for it in groups[0]])
        return real(groups, pad_id)

    monkeypatch.setattr(LS, "collate_items", spy)
    model.train()
    ref = LS.evaluate_items(model, items, device="cpu", fp16=False, pad_id=pad, batch_size=5)
    assert model.training                                          # mode restored
    assert [len(b) for b in seen] == [5, 5, 5, 5, 3] and len(seen) == math.ceil(len(items) / 5)
    flat = [n for b in seen for n in b]
    assert flat == sorted(flat)                                    # length-sorted: little padding per batch
    for bs, order in ((64, items), (1, items[::-1]), (7, items[3:] + items[:3])):
        m = LS.evaluate_items(model, order, device="cpu", fp16=False, pad_id=pad, batch_size=bs)
        assert (m["val_acc"], m["val_macro_f1"], m["n"]) == (ref["val_acc"], ref["val_macro_f1"], ref["n"])
        assert m["val_ce"] == pytest.approx(ref["val_ce"], rel=1e-5)


def test_evaluate_items_rejects_a_bad_batch_size(tiny_ckpt_dir):
    import laya

    from laya_poc.loss import evaluate_items
    agent = laya.load(str(tiny_ckpt_dir), device="cpu")
    with pytest.raises(ValueError, match="batch_size"):
        evaluate_items(agent.model, _val_items(agent.tok, 2), device="cpu", fp16=False,
                       pad_id=agent.tok.pad_token_id, batch_size=0)
