import itertools
import math

import pytest

torch = pytest.importorskip("torch")
from laya_poc import schedule as S  # noqa: E402

pytestmark = pytest.mark.torch


# ---- sigma (upstream cell08.py:133-134: a per-epoch step function) ----

def test_sigma_steps_per_epoch_like_upstream():
    assert [S.sigma_for_epoch(e, 4, 0.4, 0.1) for e in range(4)] == pytest.approx([0.4, 0.3, 0.2, 0.1])


def test_sigma_is_constant_start_for_a_single_epoch():
    assert S.sigma_for_epoch(0, 1, 0.4, 0.1) == pytest.approx(0.4)


@pytest.mark.parametrize("epoch,epochs", [(-1, 4), (4, 4), (0, 0)])
def test_sigma_rejects_out_of_range_epoch(epoch, epochs):
    with pytest.raises(ValueError):
        S.sigma_for_epoch(epoch, epochs, 0.4, 0.1)


# ---- linear warm-up then linear decay to 0 (doc §5.9) ----

def test_warmup_then_linear_decay_to_zero():
    f = S.linear_warmup_lambda(100, 0.06)  # 6 warm-up steps
    assert f(0) == pytest.approx(1 / 6)
    assert f(5) == pytest.approx(1.0)
    assert f(6) == pytest.approx(1.0)
    assert f(53) == pytest.approx(47 / 94)
    assert f(99) == pytest.approx(1 / 94)
    assert f(100) == 0.0 and f(150) == 0.0
    factors = [f(s) for s in range(100)]
    assert factors[:6] == sorted(factors[:6])                 # rising
    assert factors[6:] == sorted(factors[6:], reverse=True)   # falling
    assert all(0.0 < v <= 1.0 for v in factors)


def test_no_warmup_when_fraction_rounds_to_zero():
    f = S.linear_warmup_lambda(10, 0.06)  # int(0.6) == 0
    assert f(0) == pytest.approx(1.0)
    assert f(9) == pytest.approx(0.1)


@pytest.mark.parametrize("total,frac", [(0, 0.06), (10, -0.1), (10, 1.0)])
def test_warmup_lambda_validates(total, frac):
    with pytest.raises(ValueError):
        S.linear_warmup_lambda(total, frac)


def test_lambda_drives_lambdalr_per_optimizer_step():
    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, S.linear_warmup_lambda(10, 0.2))
    seen = []
    for _ in range(10):
        seen.append(opt.param_groups[0]["lr"])  # lr used by this optimizer step
        opt.step()
        sched.step()
    assert seen == pytest.approx([0.5, 1.0, 1.0, 7 / 8, 6 / 8, 5 / 8, 4 / 8, 3 / 8, 2 / 8, 1 / 8])


# ---- param groups (doc §5.9: no weight decay on biases, norms and any 1-D param) ----

class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Embedding(5, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 4))
        self.scorer = torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Linear(4, 1))
        self.type_emb = torch.nn.Embedding(3, 4)
        self.scale = torch.nn.Parameter(torch.ones(4))       # 1-D, not a bias or norm by name
        self.frozen = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)


def _names(model, group):
    ids = {id(p) for p in group["params"]}
    return {n for n, p in model.named_parameters() if id(p) in ids}


def test_param_groups_partition_encoder_head_and_decay():
    m = _Toy()
    groups = S.param_groups(m, 2e-5, 1e-4, 0.01)
    by = {g["name"]: g for g in groups}
    assert set(by) == {"encoder", "encoder_no_decay", "head", "head_no_decay"}
    assert _names(m, by["encoder"]) == {"encoder.0.weight", "encoder.2.weight"}
    assert _names(m, by["encoder_no_decay"]) == {"encoder.1.weight", "encoder.1.bias", "encoder.2.bias"}
    assert _names(m, by["head"]) == {"scorer.1.weight", "type_emb.weight"}
    assert _names(m, by["head_no_decay"]) == {"scorer.0.weight", "scorer.0.bias", "scorer.1.bias", "scale"}
    assert by["encoder"]["lr"] == by["encoder_no_decay"]["lr"] == 2e-5
    assert by["head"]["lr"] == by["head_no_decay"]["lr"] == 1e-4
    assert by["encoder"]["weight_decay"] == by["head"]["weight_decay"] == 0.01
    assert by["encoder_no_decay"]["weight_decay"] == by["head_no_decay"]["weight_decay"] == 0.0
    assert all(g["group"] in ("encoder", "head") for g in groups)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assigned = [n for g in groups for n in _names(m, g)]
    assert sorted(assigned) == sorted(trainable)                # each exactly once, frozen excluded
    torch.optim.AdamW(groups)                                  # accepted as-is


def test_param_groups_drop_empty_groups():
    m = torch.nn.Module()
    m.encoder = torch.nn.Linear(2, 2, bias=False)
    groups = S.param_groups(m, 1e-5, 1e-4, 0.01)
    assert [g["name"] for g in groups] == ["encoder"]


# ---- per-micro-step seeds ----

def test_micro_seed_is_deterministic_and_collision_free():
    assert S.micro_seed(11, 0, 0) == S.micro_seed(11, 0, 0)
    grid = list(itertools.product([0, 1, 7, 999], [0, 1, 99_999, 100_000, 999_999]))
    seeds = {S.micro_seed(11, e, i) for e, i in grid}
    assert len(seeds) == len(grid)
    assert S.micro_seed(11, 0, 0) != S.micro_seed(22, 0, 0)
    assert S.micro_seed(11, 999, 999_999) < S.micro_seed(12, 0, 0)
    assert 0 <= S.micro_seed(99, 999, 999_999) < 2 ** 63


@pytest.mark.parametrize("seed,epoch,idx", [(11, 1000, 0), (11, 0, 10 ** 6), (11, -1, 0), (-1, 0, 0)])
def test_micro_seed_rejects_out_of_range(seed, epoch, idx):
    with pytest.raises(ValueError):
        S.micro_seed(seed, epoch, idx)


# ---- step planning (must match the trainer's boundary rule exactly) ----

def _simulate(n_items_per_epoch, mb, acc, cap):
    """Reference: the trainer's loop with its boundary rule, counted one micro-step at a time."""
    micro = opt = 0
    for n in n_items_per_epoch:
        nb = math.ceil(n / mb)
        for bi in range(nb):
            micro += 1
            at_cap = cap is not None and micro >= cap
            if (bi + 1) % acc == 0 or bi == nb - 1 or at_cap:
                opt += 1
            if at_cap:
                return micro, opt
    return micro, opt


def test_plan_steps_counts_partial_last_window_per_epoch():
    plan = S.plan_steps([2140], micro_batch=8, grad_accum=4, max_micro_steps=None)
    assert plan == {"total_micro": 268, "total_opt": 67}        # 268 batches, last window of 4 is full
    plan = S.plan_steps([2141, 2141], micro_batch=8, grad_accum=4, max_micro_steps=None)
    assert plan == {"total_micro": 2 * 268, "total_opt": 2 * 67}
    plan = S.plan_steps([10, 10], micro_batch=2, grad_accum=4, max_micro_steps=None)
    assert plan == {"total_micro": 10, "total_opt": 4}          # 5 batches -> windows 4 + 1, per epoch


def test_plan_steps_honours_the_micro_step_cap():
    assert S.plan_steps([2140], 8, 4, 250) == {"total_micro": 250, "total_opt": 63}
    assert S.plan_steps([10, 10], 2, 4, 7) == {"total_micro": 7, "total_opt": 3}
    assert S.plan_steps([10], 2, 4, 99) == {"total_micro": 5, "total_opt": 2}


@pytest.mark.parametrize("n,mb,acc,cap", [([64], 2, 2, 12), ([63, 65, 1], 2, 3, None), ([100, 37], 8, 4, 15),
                                          ([5], 8, 4, None), ([33, 33, 33], 4, 5, 20)])
def test_plan_steps_matches_trainer_simulation(n, mb, acc, cap):
    micro, opt = _simulate(n, mb, acc, cap)
    assert S.plan_steps(n, mb, acc, cap) == {"total_micro": micro, "total_opt": opt}


@pytest.mark.parametrize("kw", [dict(micro_batch=0), dict(grad_accum=0), dict(max_micro_steps=0)])
def test_plan_steps_validates(kw):
    args = dict(n_items_per_epoch=[10], micro_batch=2, grad_accum=2, max_micro_steps=None) | kw
    with pytest.raises(ValueError):
        S.plan_steps(**args)


# ---- run layout ----

def test_batching_defaults_and_overrides(cfg):
    assert S.batching(cfg, "CPU") == (2, 2)
    assert S.batching(cfg, "T4") == (8, 4)
    assert S.batching(cfg, "L4") == (32, 1)
    assert S.batching(cfg, "T4", micro_batch=4) == (4, 8)
    assert S.batching(cfg, "CPU", micro_batch=2, effective_batch=8) == (2, 4)
    with pytest.raises(ValueError, match="multiple"):
        S.batching(cfg, "T4", micro_batch=5)


def test_count_items_and_epochs(tmp_path):
    from synth import write_synthetic_data_dir
    data = write_synthetic_data_dir(tmp_path / "data", n_train=10, n_val=4, epochs=2)
    assert S.count_items(data / "train_e0.jsonl") == 10
    assert S.available_epochs(data) == 2
    assert S.resolve_epochs(None, data) == 2 and S.resolve_epochs(1, data) == 1
    with pytest.raises(FileNotFoundError, match="needs"):
        S.resolve_epochs(3, data)
    with pytest.raises(FileNotFoundError, match="build_data"):
        S.resolve_epochs(None, tmp_path)


def _opts(**kw):
    from types import SimpleNamespace
    base = dict(model="laya", seed=11, device="cpu", card="auto", micro_batch=None, effective_batch=None,
                grad_ckpt="auto", epochs=None, max_micro_steps=None, ckpt_every_micro_steps=None,
                ckpt_every_min=None, eval_every_opt_steps=None, keep_last=None)
    return SimpleNamespace(**(base | kw))


def test_resolve_settings_on_cpu_uses_cpu_defaults_and_config(cfg, tmp_path):
    from synth import write_synthetic_data_dir
    data = write_synthetic_data_dir(tmp_path / "data", n_train=4, n_val=2)
    s = S.resolve_settings(_opts(), cfg, data)
    assert (s.device, s.card, s.micro_batch, s.grad_accum) == ("cpu", "CPU", 2, 2)
    assert s.grad_ckpt is False and s.fp16 is False and s.epochs == 1
    assert (s.max_len, s.head_max_len) == (cfg["model"]["laya"]["max_len"], cfg["model"]["laya"]["head_max_len"])
    assert s.ckpt_every_min == cfg["train"]["ckpt_every_min"] and s.keep_last == cfg["train"]["keep_last"]
    assert s.eval_every_opt_steps == cfg["train"]["eval_every_opt_steps"] and s.ckpt_every_micro_steps is None
    s = S.resolve_settings(_opts(model="laya_ml", card="T4", grad_ckpt="off", eval_every_opt_steps=0,
                                 ckpt_every_min=0, keep_last=1), cfg, data)
    assert (s.card, s.micro_batch, s.grad_accum, s.grad_ckpt) == ("T4", 8, 4, False)
    assert s.eval_every_opt_steps is None and s.ckpt_every_min == 0.0 and s.keep_last == 1
    assert s.head_max_len == cfg["model"]["laya_ml"]["head_max_len"]


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA error path")
def test_resolve_settings_fails_fast_without_cuda(cfg, tmp_path):
    with pytest.raises(RuntimeError, match="CUDA"):
        S.resolve_settings(_opts(device="cuda"), cfg, tmp_path)


def test_resolve_settings_rejects_keep_last_below_one(cfg, tmp_path):
    from synth import write_synthetic_data_dir
    data = write_synthetic_data_dir(tmp_path / "data", n_train=4, n_val=2)
    with pytest.raises(ValueError, match="keep.last"):          # before any training compute is spent
        S.resolve_settings(_opts(keep_last=0), cfg, data)
    bad_cfg = {**cfg, "train": {**cfg["train"], "keep_last": 0}}
    with pytest.raises(ValueError, match="keep.last"):
        S.resolve_settings(_opts(), bad_cfg, data)
