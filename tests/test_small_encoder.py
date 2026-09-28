"""B4 small-encoder baseline (design §5.13 item 4, §7.10) on CPU with a tiny random ModernBERT: the REAL pinned
ModernBERT-base config.json (tiny dims) + tokenizer, saved as a masked-LM checkpoint like the Hub one, so the run's
seed initialises a fresh classifier head as in a real run. Skips when the Hub is unreachable. `build_tiny_encoder`,
`write_b4_data` and `hub_config_dir` are importable helpers (tools/dry_run_p4_local.py can reuse them)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.torch
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("laya")

from laya_poc import labels as L  # noqa: E402
from laya_poc.io_utils import read_jsonl, write_jsonl  # noqa: E402
from synth import rows_from_records, synthetic_records  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HUB_CONFIG_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
# Same tiny shape as tests/conftest.py TINY_ENCODER (3 layers, 64 wide; global attention every 3rd layer).
TINY_DIMS = dict(hidden_size=64, intermediate_size=96, num_attention_heads=4, num_hidden_layers=3,
                 layer_types=["full_attention", "sliding_attention", "sliding_attention"])
EVAL_KEYS = {"model", "ckpt", "split", "n", "n_unlabelled", "n_abstained_no_evidence", "temperature", "pre", "post",
             "abstention"}
STRIPPED_KEYS = {"abstain_rate_at_tau", "false_confident_rate", "false_confident_threshold", "gate_abstain_rate",
                 "no_gate"}
PRED_KEYS = {"id", "y", "p", "answer_confidence", "abstained"}


# ---------------------------------------------------------------- helpers (importable)

def hub_config_dir(cfg: dict, key: str) -> Path:
    """config.json + tokenizer files of a pinned phase4.small_encoders entry (no weights)."""
    from huggingface_hub import snapshot_download

    spec = cfg["phase4"]["small_encoders"][key]
    return Path(snapshot_download(spec["id"], revision=spec["revision"], allow_patterns=HUB_CONFIG_FILES))


def build_tiny_encoder(src: Path, dst: Path, seed: int = 0) -> Path:
    """A randomly initialised 3-layer ModernBERT masked-LM checkpoint (+ tokenizer) from a real config dir."""
    import torch
    from transformers import AutoConfig, AutoTokenizer, ModernBertForMaskedLM

    config = AutoConfig.from_pretrained(str(src), **TINY_DIMS)
    torch.manual_seed(seed)
    ModernBertForMaskedLM(config).save_pretrained(str(dst))
    AutoTokenizer.from_pretrained(str(src)).save_pretrained(str(dst))
    return dst


def _renumber(rows: list[dict], split: str, start: int = 0) -> list[dict]:
    return [{**r, "id": f"{split}-{start + i:06d}", "split": split} for i, r in enumerate(rows)]


def _stripped(rows: list[dict], scheme: str = "c10") -> list[dict]:
    """Country-only copies with the TRUE label kept and a uniform gold, as build_data writes stripped_test."""
    uniform = json.dumps(L.gold(None, scheme, 0.1), ensure_ascii=False)
    out = [{**r, "state": json.dumps({"country": json.loads(r["state"])["country"]}, separators=(",", ":")),
            "gold": uniform, "has_evidence": False, "aug": "stripped"} for r in rows]
    return _renumber(out, "stripped_test")


def write_b4_data(out: Path, *, epochs: int = 3, n_train: int = 96, n_eval: int = 32, seed: int = 0) -> Path:
    """train_e{k} (with a few uniform-target no-evidence rows), val, test_id, stripped_test, and
    <out>/extra/trap_candidates.jsonl."""
    out.mkdir(parents=True, exist_ok=True)
    for e in range(epochs):
        labelled = rows_from_records(synthetic_records(n_train, seed + e), "train")
        empty = rows_from_records([({"country": "GB"}, None)] * 4, "train")
        write_jsonl(out / f"train_e{e}.jsonl", _renumber(labelled + empty, "train"))
    write_jsonl(out / "val.jsonl", rows_from_records(synthetic_records(n_eval, seed + 100), "val"))
    test = rows_from_records(synthetic_records(n_eval, seed + 200), "test_id")
    write_jsonl(out / "test_id.jsonl", test)
    write_jsonl(out / "stripped_test.jsonl", _stripped(test[:12]))
    traps = rows_from_records(synthetic_records(12, seed + 300), "trap_candidates")
    write_jsonl(out / "extra" / "trap_candidates.jsonl", traps)
    return out


def small_config(tmp_path: Path, **train: object) -> Path:
    """The project config with CPU-sized B4 settings (tiny batches/lengths, a small order-invariance check)."""
    from laya_poc.config import load_config

    cfg = copy.deepcopy(load_config(ROOT / "config.yaml"))
    p4 = cfg["phase4"]
    cfg["phase4"] = {**p4, "small_encoder_train": {**p4["small_encoder_train"], "batch_size": 16, "max_length": 64,
                                                   **train}}
    cfg["eval"] = {**cfg["eval"], "batch_size": 16}
    cfg["phase3"] = {**cfg["phase3"], "order_invariance": {"split": "test_id", "n": 16, "perms": 2, "seed": 7}}
    path = tmp_path / "config_b4.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="session")
def mb_config_dir(cfg):
    try:
        return hub_config_dir(cfg, "modernbert_base")
    except Exception as exc:  # offline, rate-limited, ...
        pytest.skip(f"Hub unavailable: {exc}")


@pytest.fixture(scope="session")
def tiny_encoder_dir(mb_config_dir, tmp_path_factory):
    return build_tiny_encoder(mb_config_dir, tmp_path_factory.mktemp("tiny_encoder") / "enc")


@pytest.fixture(scope="session")
def b4_data(tmp_path_factory):
    return write_b4_data(tmp_path_factory.mktemp("b4") / "data")


def cli_args(run_dir: Path, data: Path, init: Path, config: Path, *extra: str) -> list[str]:
    return ["--model", "modernbert_base", "--seed", "11", "--scheme", "c10", "--data-dir", str(data),
            "--run-dir", str(run_dir), "--init", str(init), "--device", "cpu", "--config", str(config),
            "--splits", "val", "test_id", "stripped_test",
            "--extra", f"trap_candidates={data / 'extra' / 'trap_candidates.jsonl'}", *extra]


@pytest.fixture(scope="module")
def b4_run(tiny_encoder_dir, b4_data, tmp_path_factory):
    """One full CLI run (3 epochs, CPU) shared by the schema tests."""
    from laya_poc.small_encoder import main

    tmp = tmp_path_factory.mktemp("b4_run")
    run_dir = tmp / "fsq-c10-B4-modernbert_base-s11"
    assert main(cli_args(run_dir, b4_data, tiny_encoder_dir, small_config(tmp))) == 0
    return run_dir


# ---------------------------------------------------------------- pure / small pieces

def test_encoder_source_reads_pinned_hub_ids(cfg, tmp_path):
    from laya_poc.small_encoder_train import encoder_source

    for key in ("modernbert_base", "mmbert_small"):
        src = encoder_source(cfg, key)
        spec = cfg["phase4"]["small_encoders"][key]
        assert (src.repo_id, src.revision, src.init) == (spec["id"], spec["revision"], None)
    assert encoder_source(cfg, "mmbert_small", tmp_path).init == tmp_path
    with pytest.raises(ValueError, match="phase4.small_encoders"):
        encoder_source(cfg, "bert_tiny")
    with pytest.raises(ValueError, match="absolute"):
        encoder_source(cfg, "mmbert_small", Path("relative/dir"))


def test_train_config_follows_config_and_disables_fp16_on_cpu(cfg):
    from laya_poc.small_encoder_train import train_config

    st = cfg["phase4"]["small_encoder_train"]
    tc = train_config(cfg, seed=22, device="cpu")
    assert (tc.lr, tc.batch_size, tc.epochs, tc.warmup_frac, tc.weight_decay, tc.max_length) == (
        st["lr"], st["batch_size"], st["epochs"], st["warmup_frac"], st["weight_decay"], st["max_length"])
    assert tc.fp16 is False and tc.seed == 22 and tc.max_steps is None
    assert tc.eval_batch_size == cfg["eval"]["batch_size"]
    tc2 = train_config(cfg, seed=11, device="cpu", epochs=1, max_steps=5)
    assert (tc2.epochs, tc2.max_steps) == (1, 5)
    with pytest.raises(ValueError):
        train_config(cfg, seed=11, device="cpu", max_steps=0)


def test_soft_targets_are_the_gold_probabilities_in_option_order():
    from laya_poc.small_encoder_data import soft_targets

    rows = rows_from_records([({"name": "x"}, "health"), ({"country": "GB"}, None)], "train", smoothing=0.1)
    keys = L.option_keys("c10")
    T = soft_targets(rows, keys)
    assert T.shape == (2, 10) and np.allclose(T.sum(1), 1.0)
    assert T[0].argmax() == keys.index("health") and T[0].max() == pytest.approx(0.9)
    assert np.allclose(T[1], 0.1)  # the no-evidence row: uniform target
    with pytest.raises(ValueError, match="options"):
        soft_targets(rows, L.option_keys("c7"))


def test_soft_ce_matches_hard_ce_on_one_hot_and_uses_every_class():
    import torch
    from laya_poc.small_encoder_data import soft_ce

    logits = torch.tensor([[2.0, 0.5, -1.0], [0.0, 1.0, 3.0]])
    onehot = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert soft_ce(logits, onehot).item() == pytest.approx(
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).item(), abs=1e-6)
    uniform = torch.full((2, 3), 1 / 3)
    expected = -(torch.log_softmax(logits, -1).mean(-1)).mean().item()
    assert soft_ce(logits, uniform).item() == pytest.approx(expected, abs=1e-6)


def test_encode_truncates_with_special_tokens_and_counts(tiny_encoder_dir):
    from transformers import AutoTokenizer
    from laya_poc.small_encoder_data import collate, encode

    tok = AutoTokenizer.from_pretrained(str(tiny_encoder_dir))
    short = json.dumps({"country": "GB", "name": "Rosa's Cafe"}, separators=(",", ":"))
    long = json.dumps({"country": "GB", "name": "word " * 200}, separators=(",", ":"))
    ids, truncated = encode(tok, [short, long], 32)
    assert truncated == 1 and len(ids[1]) == 32 and ids[1][-1] == tok.sep_token_id and ids[1][0] == tok.cls_token_id
    assert ids[0] == tok(short)["input_ids"]
    input_ids, mask = collate([ids[0], ids[1]], tok.pad_token_id)
    assert tuple(input_ids.shape) == (2, 32) and int(mask[0].sum()) == len(ids[0]) and int(mask[1].sum()) == 32
    assert (input_ids[0, len(ids[0]):] == tok.pad_token_id).all()


def test_pinned_encoders_load_as_modernbert_sequence_classifiers(cfg):
    """VERIFY (spec §3): ModernBERT-base and mmBERT-small both map to ModernBertForSequenceClassification in the
    installed transformers (config + tokenizer only; tiny dims so no weights are needed)."""
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    from transformers import ModernBertForSequenceClassification

    for key in ("modernbert_base", "mmbert_small"):
        try:
            src = hub_config_dir(cfg, key)
        except Exception as exc:  # offline, rate-limited, ...
            pytest.skip(f"Hub unavailable: {exc}")
        config = AutoConfig.from_pretrained(str(src), num_labels=10, **TINY_DIMS)
        assert config.model_type == "modernbert"
        model = AutoModelForSequenceClassification.from_config(config)
        assert isinstance(model, ModernBertForSequenceClassification)
        tok = AutoTokenizer.from_pretrained(str(src))
        enc = tok(['{"country":"TH","name":"ร้านกาแฟ"}'], return_tensors="pt")
        with torch.no_grad():
            assert tuple(model(**enc).logits.shape) == (1, 10)


# ---------------------------------------------------------------- training

def _train(tiny_encoder_dir: Path, data: Path, run_dir: Path, config: Path, seed: int = 11, **kw) -> tuple:
    from laya_poc.config import load_config
    from laya_poc.small_encoder_data import val_set
    from laya_poc.small_encoder_train import encoder_source, load_encoder, train_config, train_encoder

    cfg, keys = load_config(config), L.option_keys("c10")
    tc = train_config(cfg, seed=seed, device="cpu", **kw)
    model, tok = load_encoder(encoder_source(cfg, "modernbert_base", tiny_encoder_dir), len(keys), seed)
    val = val_set(read_jsonl(data / "val.jsonl"), keys)
    files = [data / f"train_e{e}.jsonl" for e in range(tc.epochs)]
    summary = train_encoder(model, tok, tc, files, val, run_dir / "train", meta={"model": "modernbert_base"})
    return model, tok, summary


def test_training_reduces_loss(tiny_encoder_dir, tmp_path):
    data = write_b4_data(tmp_path / "data", epochs=4, n_train=160)
    _, _, s = _train(tiny_encoder_dir, data, tmp_path / "run", small_config(tmp_path, lr=2.0e-3, epochs=4))
    losses = [e["train_loss"] for e in s["per_epoch"]]
    ces = [e["val_ce"] for e in s["per_epoch"]]
    assert all(np.isfinite(losses)) and losses[-1] < 0.8 * losses[0], losses
    assert ces[-1] < ces[0], ces
    assert s["nonfinite"] == 0 and s["stop_reason"] == "epochs" and s["micro_steps"] == s["total_opt_steps"]


@pytest.mark.parametrize("scripted, best", [([0.2, 0.6, 0.4], 2), ([0.5, 0.5, 0.3], 1), ([0.1, 0.2, 0.3], 3)])
def test_best_epoch_weights_are_kept(tiny_encoder_dir, b4_data, tmp_path, monkeypatch, scripted, best):
    """The weights left in the model and in train/best are the best val macro-F1 epoch's (first on ties)."""
    import torch
    from transformers import AutoModelForSequenceClassification
    from laya_poc import small_encoder_train as T

    snaps: list[dict] = []

    def fake(model, tok, val, tc):
        snaps.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return {"val_macro_f1": scripted[len(snaps) - 1], "val_acc": 0.0, "val_ce": 1.0, "n": 1}

    monkeypatch.setattr(T, "val_metrics", fake)
    model, _, s = _train(tiny_encoder_dir, b4_data, tmp_path, small_config(tmp_path))
    assert s["best_epoch"] == best and s["best_val_macro_f1"] == scripted[best - 1]
    assert s["best_opt_step"] == s["per_epoch"][best - 1]["step"]
    assert [e["best"] for e in s["per_epoch"]] == [i == best - 1 for i in range(3)]
    want = snaps[best - 1]
    assert all(torch.equal(v, want[k]) for k, v in model.state_dict().items())
    saved = AutoModelForSequenceClassification.from_pretrained(str(tmp_path / "train" / "best")).state_dict()
    assert all(torch.equal(v, want[k]) for k, v in saved.items())


def test_a_skipped_update_does_not_advance_the_lr_schedule(tiny_encoder_dir, b4_data, tmp_path, monkeypatch):
    """A non-finite loss (fp32: no backward) applies no update, and the LR schedule does not move for it (the HF
    Trainer's rule for skipped steps), so the last planned step still runs at a non-zero rate."""
    from laya_poc import small_encoder_train as T

    calls, real = [], T.soft_ce

    def nan_on_third(logits, targets):
        calls.append(1)
        loss = real(logits, targets)
        return loss * float("nan") if len(calls) == 3 else loss

    monkeypatch.setattr(T, "soft_ce", nan_on_third)
    monkeypatch.setattr(T, "PRINTS_PER_RUN", 1000)  # log every step
    _, _, s = _train(tiny_encoder_dir, b4_data, tmp_path, small_config(tmp_path))
    assert (s["nonfinite"], s["opt_steps_skipped"], s["micro_steps"], s["opt_steps"]) == (1, 1, 21, 20)
    steps = [json.loads(x) for x in (tmp_path / "train" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    steps = [e for e in steps if e["event"] == "step"]
    assert steps[2]["finite"] is False and steps[2]["grad_norm"] is None and steps[2]["lr"] == steps[1]["lr"]
    assert steps[-1]["lr"] > 0


def test_max_steps_caps_training_mid_epoch(tiny_encoder_dir, b4_data, tmp_path):
    _, _, s = _train(tiny_encoder_dir, b4_data, tmp_path, small_config(tmp_path), max_steps=8)
    assert s["micro_steps"] == 8 and s["stop_reason"] == "max_steps"
    assert len(s["per_epoch"]) == 2 and s["per_epoch"][-1]["step"] == 8  # epoch 1 has 7 batches of 16 (100 rows)
    events = [json.loads(x) for x in (tmp_path / "train" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    start = next(e for e in events if e["event"] == "start")
    assert start["micro_batch"] == 16 and start["n_items_per_epoch"] == [100, 100, 100]
    assert events[-1]["event"] == "done"


# ---------------------------------------------------------------- CLI: layout, schema, determinism

def test_cli_writes_the_shared_run_layout(b4_run, b4_data):
    for split in ("val", "test_id", "stripped_test", "trap_candidates"):
        payload = json.loads((b4_run / "eval" / f"{split}.json").read_text(encoding="utf-8"))
        assert EVAL_KEYS <= set(payload) and payload["split"] == split and payload["model"] == "modernbert_base"
        preds = read_jsonl(b4_run / "preds" / f"{split}.jsonl")
        assert len(preds) == payload["n"] and all(set(p) == PRED_KEYS for p in preds)
    val = json.loads((b4_run / "eval" / "val.json").read_text(encoding="utf-8"))
    assert "tau" in val
    stripped = json.loads((b4_run / "eval" / "stripped_test.json").read_text(encoding="utf-8"))
    assert STRIPPED_KEYS <= set(stripped) and stripped["n_abstained_no_evidence"] == stripped["n"]
    no_gate = read_jsonl(b4_run / "preds" / "no_gate" / "stripped_test.jsonl")
    assert len(no_gate) == stripped["n"] and not any(p["abstained"] for p in no_gate)
    cal = json.loads((b4_run / "calibration.json").read_text(encoding="utf-8"))
    assert {"T", "clamped", "fitted_on", "n", "extra"} <= set(cal) and cal["fitted_on"] == "val"
    assert val["temperature"] == pytest.approx(cal["T"]) and cal["n"] == 32
    oi = json.loads((b4_run / "order_invariance.json").read_text(encoding="utf-8"))
    assert {"split", "n", "perms", "agreement_per_perm", "mean_agreement", "passed_99"} <= set(oi)
    assert (oi["split"], oi["n"], oi["perms"]) == ("test_id", 16, 2)
    assert (b4_run / "train" / "best" / "model.safetensors").is_file()
    assert not (b4_run / "done.json").exists()  # the matrix writes it


def test_cli_summary_carries_the_matrix_fields(b4_run):
    s = json.loads((b4_run / "train" / "summary.json").read_text(encoding="utf-8"))
    for k in ("model", "hub_id", "revision", "init", "seed", "scheme", "device", "fp16", "epochs", "micro_steps",
              "opt_steps", "total_opt_steps", "stop_reason", "best_epoch", "best_opt_step", "best_val_macro_f1",
              "per_epoch", "seconds", "hparams"):
        assert k in s, k
    assert s["epochs"] == 3 and len(s["per_epoch"]) == 3 and s["fp16"] is False and s["seed"] == 11
    assert s["micro_steps"] == 3 * 7 and s["stop_reason"] == "epochs"
    assert {"epoch", "step", "train_loss", "val_macro_f1", "val_acc", "val_ce", "seconds", "best"} <= set(
        s["per_epoch"][0])


def test_val_metrics_in_eval_match_the_preds(b4_run):
    """eval/val.json post macro-F1 is the macro-F1 of its own preds (every val row answered and labelled)."""
    from laya_poc.metrics import macro_f1

    val = json.loads((b4_run / "eval" / "val.json").read_text(encoding="utf-8"))
    preds = read_jsonl(b4_run / "preds" / "val.jsonl")
    y = np.array([p["y"] for p in preds])
    yhat = np.array([int(np.argmax(p["p"])) for p in preds])
    assert val["post"]["macro_f1"] == pytest.approx(macro_f1(y, yhat, labels=list(range(10))))


def test_cli_is_deterministic_and_reruns_overwrite_cleanly(b4_run, tiny_encoder_dir, b4_data, tmp_path):
    from laya_poc.small_encoder import main

    def same_outputs(a: Path, b: Path) -> None:
        for rel in ("preds/val.jsonl", "preds/test_id.jsonl", "preds/no_gate/stripped_test.jsonl",
                    "preds/trap_candidates.jsonl"):
            assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel
        ca, cb = (json.loads((d / "calibration.json").read_text(encoding="utf-8")) for d in (a, b))
        assert ca["T"] == cb["T"]

    run_b = tmp_path / "again"
    config = small_config(tmp_path)
    assert main(cli_args(run_b, b4_data, tiny_encoder_dir, config)) == 0
    same_outputs(b4_run, run_b)
    (run_b / "eval" / "stale.json").write_text("{}", encoding="utf-8")
    (run_b / "done.json").write_text("{}", encoding="utf-8")
    (run_b / "logs").mkdir()
    (run_b / "logs" / "b4.log").write_text("matrix tee", encoding="utf-8")
    assert main(cli_args(run_b, b4_data, tiny_encoder_dir, config)) == 0
    assert not (run_b / "eval" / "stale.json").exists() and not (run_b / "done.json").exists()
    assert (run_b / "logs" / "b4.log").read_text(encoding="utf-8") == "matrix tee"  # not ours: kept
    log = (run_b / "train" / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert sum(json.loads(x)["event"] == "start" for x in log) == 1
    same_outputs(b4_run, run_b)


def test_cli_fails_with_one_line_errors(tiny_encoder_dir, b4_data, tmp_path, capsys):
    from laya_poc.small_encoder import main

    config = small_config(tmp_path)
    assert main(cli_args(Path("relative/run"), b4_data, tiny_encoder_dir, config)) == 1
    assert "absolute" in capsys.readouterr().err
    assert main(cli_args(tmp_path / "r", b4_data, tiny_encoder_dir, config, "--epochs", "5")) == 1
    assert "train_e3.jsonl" in capsys.readouterr().err
    assert main(cli_args(tmp_path / "r2", b4_data, tiny_encoder_dir, config, "--splits", "test_id")) == 1
    assert "val" in capsys.readouterr().err
