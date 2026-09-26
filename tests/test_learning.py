"""The ported training loop must actually reduce the loss (smoke exit criterion: loss_ce drops).

Uses a tiny checkpoint with a non-degenerate init (the default fixture's option logits are all
equal, so its loss is pinned at ln(10) whatever the optimiser does) and raised learning rates,
because the doc's rates are sized for the pretrained 421M-parameter model.
"""
import json
import statistics

import pytest
import yaml

pytest.importorskip("laya")
pytestmark = pytest.mark.torch


@pytest.fixture(scope="module")
def learnable_ckpt(en_snapshot, tmp_path_factory):
    from conftest import build_tiny_checkpoint
    return build_tiny_checkpoint(en_snapshot, tmp_path_factory.mktemp("learnable") / "ckpt", init_scale=0.5)


def test_training_reduces_loss_ce(learnable_ckpt, cfg, tmp_path):
    from synth import write_synthetic_data_dir
    from laya_poc import train_single
    from laya_poc.config import with_overrides

    data = write_synthetic_data_dir(tmp_path / "data", n_train=400, n_val=40)
    fast = with_overrides(cfg, {"train.lr_encoder": 1e-3, "train.lr_head": 3e-3, "train.warmup_frac": 0.0})
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(fast), encoding="utf-8")
    run = tmp_path / "run"
    rc = train_single.main(["--run-dir", str(run), "--data-dir", str(data), "--config", str(tmp_path / "cfg.yaml"),
                            "--init", str(learnable_ckpt), "--device", "cpu", "--micro-batch", "8",
                            "--effective-batch", "8", "--max-micro-steps", "50", "--print-every", "1000"])
    assert rc in (0, None)
    ce = [e["loss_ce"] for e in map(json.loads, (run / "log.jsonl").read_text().splitlines()) if e["event"] == "micro"]
    first, last = statistics.mean(ce[:10]), statistics.mean(ce[-10:])
    assert last < 0.8 * first, (first, last)
