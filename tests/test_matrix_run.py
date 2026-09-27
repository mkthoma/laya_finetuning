"""`python -m laya_poc.matrix run` with the subprocess layer mocked: a fake runner records every command and
writes the outputs the real CLI would (train_single, export_check, evaluate multi-split). Covers step order and
arguments, skip-if-done, step-level resume, cleanup, done.json, runs.csv, failures and missing variants."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from laya_poc import matrix as M
from laya_poc import matrix_exec as X
from laya_poc.io_utils import write_jsonl

from synth import rows_from_records, synthetic_records

SPLITS = ("val", "test_id", "ood_country", "ood_script", "ood_brand", "stripped_test", "trap_candidates")


def opts(args: list[str]) -> dict[str, list[str]]:
    """--flag v1 v2 ... -> {flag: [v1, v2]}; repeated flags accumulate."""
    out: dict[str, list[str]] = {}
    key = None
    for tok in args:
        if tok.startswith("--"):
            key = tok
            out.setdefault(key, [])
        elif key is not None:
            out[key].append(tok)
    return out


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _metrics(f1: float, ece: float) -> dict:
    return {"n": 8, "macro_f1": f1, "macro_f1_9": f1 + 0.01, "acc": f1 + 0.02, "ece": ece}


class FakeRunner:
    """Stands in for matrix_exec.tee_run: (cmd, log_path) -> return code."""

    def __init__(self, fail: dict[str, int] | None = None, drop_split: str | None = None):
        self.calls: list[tuple[str, dict[str, list[str]], list[str], Path]] = []
        self.fail, self.drop_split = fail or {}, drop_split

    def modules(self) -> list[str]:
        return [c[0] for c in self.calls]

    def __call__(self, cmd, log_path) -> int:
        cmd = [str(c) for c in cmd]
        i = cmd.index("-m")
        module = cmd[i + 1].rsplit(".", 1)[-1]
        o = opts(cmd[i + 2:])
        self.calls.append((module, o, cmd, Path(log_path)))
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("$ " + " ".join(cmd) + "\nfake output\n", encoding="utf-8")
        if module in self.fail:
            return self.fail[module]
        getattr(self, module)(o)
        return 0

    def train_single(self, o) -> None:
        run = Path(o["--run-dir"][0])
        events = [{"t": 100.0, "event": "start", "n_items_per_epoch": [8, 8, 8, 8], "micro_batch": 2},
                  *({"t": 101.0 + e, "event": "micro", "micro_step": e + 1, "epoch": e} for e in range(4)),
                  {"t": 160.0, "event": "done", "micro_steps": 16, "opt_steps": 4, "stop_reason": "epochs"}]
        (run / "ckpt").mkdir(parents=True, exist_ok=True)
        (run / "ckpt" / "step0000004.pt").write_bytes(b"x" * 10)
        (run / "log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
        _json(run / "best" / "train_eval.json", {"opt_step": 4, "final_eval": {"val_acc": 0.5, "n": 8}})
        (run / "best" / "model.safetensors").write_bytes(b"w")
        _json(run / "summary.json", {"stop_reason": "epochs", "best_opt_step": 4, "epochs": 4, "micro_steps": 16,
                                     "opt_steps": 4, "model": o["--model"][0], "seed": int(o["--seed"][0])})

    def export_check(self, o) -> None:
        _json(Path(o["--out"][0]), {"T": 1.25, "clamped": False, "passed": True,
                                    "pre": {"ece": 0.08}, "post": {"ece": 0.03}})

    def evaluate(self, o) -> None:
        names = list(o["--splits"]) + [e.split("=", 1)[0] for e in o.get("--extra", [])]
        for split in dict.fromkeys(names):
            if split == self.drop_split:
                continue
            res = {"split": split, "temperature": 1.25, "pre": _metrics(0.5, 0.08), "post": _metrics(0.5, 0.03)}
            if split == "stripped_test":
                res = {**res, "false_confident_rate": 0.01}
            _json(Path(o["--out-dir"][0]) / f"{split}.json", res)
            write_jsonl(Path(o["--preds-dir"][0]) / f"{split}.jsonl", [{"id": "x", "y": 0}])
        _json(Path(o["--order-invariance-out"][0]), {"split": "test_id", "n": 8, "perms": 5,
                                                     "agreement_per_perm": [1.0] * 5, "mean_agreement": 1.0,
                                                     "passed_99": True})


def _rows(path: Path, n: int, split: str) -> None:
    write_jsonl(path, rows_from_records(synthetic_records(n), split))


def _data_dir(d: Path, *, epochs: int = 4, eval_splits: bool = True) -> None:
    for e in range(epochs):
        _rows(d / f"train_e{e}.jsonl", 8, "train")
    _rows(d / "val.jsonl", 8, "val")
    for split in SPLITS[1:-1] if eval_splits else ():
        _rows(d / f"{split}.jsonl", 8, split)


@pytest.fixture()
def work(tmp_path, cfg) -> Path:
    """A work dir with every data variant the config's matrix needs, and its config.yaml."""
    w = tmp_path / "work"
    _data_dir(w / "data")
    _data_dir(w / "data_c7")
    for n in (1000, 3000, 10000):
        _data_dir(w / f"data_lc{n}", eval_splits=False)
    _rows(w / "data_eval" / "trap_candidates.jsonl", 4, "trap_candidates")
    _rows(w / "data_eval" / "trap_candidates_c7.jsonl", 4, "trap_candidates")
    (w / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return w


def run(work: Path, runner: FakeRunner, *only: str, extra: tuple[str, ...] = ()) -> int:
    argv = ["run", "--only", *only, "--config", str(work / "config.yaml"), "--work", str(work),
            "--device", "cpu", "--card", "CPU", *extra]
    return M.main(argv, runner=runner)


def run_dir(work: Path, name: str) -> Path:
    return work / "runs" / "p3" / name


# ---------------------------------------------------------------- one trained run

def test_a_trained_run_trains_calibrates_evaluates_cleans_up_and_is_recorded(work):
    fake = FakeRunner()
    assert run(work, fake, "fsq-c10-E2-laya-s22") == 0
    assert fake.modules() == ["train_single", "export_check", "evaluate"]
    rd = run_dir(work, "fsq-c10-E2-laya-s22")
    train, export, evaluate = (c[1] for c in fake.calls)
    assert train["--run-dir"] == [str(rd / "train")] and train["--data-dir"] == [str(work / "data")]
    assert train["--model"] == ["laya"] and train["--seed"] == ["22"] and train["--epochs"] == ["4"]
    assert train["--device"] == ["cpu"] and train["--card"] == ["CPU"] and train["--init"] == ["hub"]
    assert "--save-best" in train and "--final-eval" in train
    assert "--freeze-encoder" not in train and "--eval-every-opt-steps" not in train
    assert train["--config"] == [str(rd / "run_config.yaml")]
    assert export["--ckpt"] == [str(rd / "train" / "best")] and export["--model"] == ["laya"]
    assert export["--rows"] == [str(work / "data" / "val.jsonl")]
    assert export["--out"] == [str(rd / "export_check.json")]
    assert export["--train-summary"] == [str(rd / "train" / "best" / "train_eval.json")]
    assert evaluate["--ckpt"] == [str(rd / "train" / "best")] and evaluate["--splits"] == list(SPLITS)
    assert evaluate["--data-dir"] == [str(work / "data")]
    assert evaluate["--extra"] == [f"trap_candidates={work / 'data_eval' / 'trap_candidates.jsonl'}"]
    assert evaluate["--out-dir"] == [str(rd / "eval")] and evaluate["--preds-dir"] == [str(rd / "preds")]
    assert evaluate["--order-invariance-out"] == [str(rd / "order_invariance.json")]
    assert (evaluate["--order-split"], evaluate["--order-n"], evaluate["--order-perms"], evaluate["--order-seed"])         == (["test_id"], ["1000"], ["5"], ["20260925"])
    # logs per step, cleanup keeps best/ and the small files, done.json per contract
    assert [c[3] for c in fake.calls] == [rd / "logs" / f"{s}.log" for s in ("train", "export_check", "evaluate")]
    assert not (rd / "train" / "ckpt").exists() and (rd / "train" / "best" / "model.safetensors").is_file()
    assert (rd / "train" / "summary.json").is_file() and (rd / "train" / "log.jsonl").is_file()
    done = json.loads((rd / "done.json").read_text(encoding="utf-8"))
    assert set(done) == {"run_name", "arm", "model", "scheme", "seed", "subset", "head_only", "card", "finished_at",
                         "seconds"}
    assert (done["run_name"], done["arm"], done["seed"], done["subset"], done["head_only"], done["card"]) == \
        ("fsq-c10-E2-laya-s22", "E2", 22, None, False, "CPU")
    rows = list(csv.DictReader((work / "results" / "runs.csv").open(encoding="utf-8")))
    assert [r["run_name"] for r in rows] == ["fsq-c10-E2-laya-s22"] and rows[0]["T"] == "1.25"


def test_run_config_carries_the_scheme_of_the_run(work):
    fake = FakeRunner()
    assert run(work, fake, "fsq-c7-E5-laya_ml-s11") == 0
    rd = run_dir(work, "fsq-c7-E5-laya_ml-s11")
    saved = yaml.safe_load((rd / "run_config.yaml").read_text(encoding="utf-8"))
    assert saved["labels"]["scheme"] == "c7" and saved["phase3"]["card"] == "G4"
    train, export, evaluate = (c[1] for c in fake.calls)
    assert train["--data-dir"] == [str(work / "data_c7")] and evaluate["--data-dir"] == [str(work / "data_c7")]
    assert export["--rows"] == [str(work / "data_c7" / "val.jsonl")]
    assert evaluate["--extra"] == [f"trap_candidates={work / 'data_eval' / 'trap_candidates_c7.jsonl'}"]
    assert all(c[1]["--config"] == [str(rd / "run_config.yaml")] for c in fake.calls)


def test_head_only_and_subset_runs(work):
    fake = FakeRunner()
    assert run(work, fake, "E4", "fsq-c10-E6-laya-s11-n3000") == 0
    trains = [c[1] for c in fake.calls if c[0] == "train_single"]
    head, sub = trains
    assert "--freeze-encoder" in head and "--eval-every-opt-steps" not in head
    assert "--freeze-encoder" not in sub and sub["--data-dir"] == [str(work / "data_lc3000")]
    assert sub["--eval-every-opt-steps"] == ["10"]  # CPU 2 x 2: 4 x ceil(8 / 2) / 2 = 8 opt steps -> max(10, 0)
    evals = [c[1] for c in fake.calls if c[0] == "evaluate"]
    assert evals[1]["--data-dir"] == [str(work / "data")]  # subsets are evaluated on the full splits


def test_train_extra_is_appended_so_it_wins(work):
    fake = FakeRunner()
    assert run(work, fake, "fsq-c10-E6-laya-s11-n1000",
               extra=("--train-extra=--max-micro-steps 4 --eval-every-opt-steps 2", "--train-extra=--print-every 1")) == 0
    cmd = fake.calls[0][2]
    assert cmd[-6:] == ["--max-micro-steps", "4", "--eval-every-opt-steps", "2", "--print-every", "1"]
    assert fake.calls[0][1]["--eval-every-opt-steps"] == ["10", "2"]  # argparse keeps the last value


# ---------------------------------------------------------------- zero-shot

def test_zero_shot_only_evaluates_the_hub_checkpoint(work):
    fake = FakeRunner()
    assert run(work, fake, "B2-laya_ml") == 0
    assert fake.modules() == ["evaluate"]
    ev = fake.calls[0][1]
    assert ev["--ckpt"] == ["hub"] and ev["--model"] == ["laya_ml"] and ev["--splits"] == list(SPLITS)
    rd = run_dir(work, "fsq-c10-B2-laya_ml-zs")
    done = json.loads((rd / "done.json").read_text(encoding="utf-8"))
    assert (done["arm"], done["seed"], done["head_only"]) == ("B2", None, False)
    rows = list(csv.DictReader((work / "results" / "runs.csv").open(encoding="utf-8")))
    assert rows[0]["T"] == "1.25" and rows[0]["epochs_run"] == "" and rows[0]["clamped"] == ""


def test_zero_shot_with_a_local_init_checkpoint(work, tmp_path):
    ckpt = tmp_path / "tiny"
    ckpt.mkdir()
    fake = FakeRunner()
    assert run(work, fake, "B2-laya", extra=("--init", str(ckpt))) == 0
    assert fake.calls[0][1]["--ckpt"] == [str(ckpt)]


def test_init_must_be_hub_or_an_existing_absolute_dir(work, capsys):
    assert run(work, FakeRunner(), "B2-laya", extra=("--init", "relative/ckpt")) == 1
    assert "--init" in capsys.readouterr().err


# ---------------------------------------------------------------- skip, resume, failures

def test_done_runs_are_skipped(work, capsys):
    assert run(work, FakeRunner(), "E4") == 0
    again = FakeRunner()
    assert run(work, again, "E4") == 0
    assert again.calls == [] and "skip" in capsys.readouterr().out


def test_finished_steps_are_not_rerun_after_an_interruption(work):
    first = FakeRunner(fail={"evaluate": 1})
    assert run(work, first, "E4") == 1
    rd = run_dir(work, "fsq-c10-E4-laya-s11-head")
    assert (rd / "train" / "ckpt").exists() and not (rd / "done.json").exists()  # no cleanup before success
    second = FakeRunner()
    assert run(work, second, "E4") == 0
    assert second.modules() == ["evaluate"]
    assert (rd / "logs" / "evaluate_2.log").is_file()  # the failed attempt's log is kept
    assert (rd / "done.json").is_file() and not (rd / "train" / "ckpt").exists()


def test_an_interrupted_training_is_rerun_so_train_single_resumes(work):
    rd = run_dir(work, "fsq-c10-E4-laya-s11-head")
    (rd / "train" / "ckpt").mkdir(parents=True)
    (rd / "train" / "ckpt" / "step0000002.pt").write_bytes(b"x")
    fake = FakeRunner()
    assert run(work, fake, "E4") == 0
    assert fake.modules() == ["train_single", "export_check", "evaluate"]
    assert fake.calls[0][1]["--run-dir"] == [str(rd / "train")]  # same run dir: train_single resumes from ckpt/


def test_a_done_run_missing_a_newly_configured_split_warns_and_reevaluates_after_done_is_removed(work, cfg, capsys):
    assert run(work, FakeRunner(), "E4") == 0
    grown = {**cfg, "phase3": {**cfg["phase3"], "eval_splits": [*SPLITS, "ood_extra"]}}
    (work / "config.yaml").write_text(yaml.safe_dump(grown, sort_keys=False), encoding="utf-8")
    (work / "data" / "ood_extra.jsonl").write_text((work / "data" / "val.jsonl").read_text(encoding="utf-8"),
                                                   encoding="utf-8")
    quiet = FakeRunner()
    assert run(work, quiet, "E4") == 0
    assert quiet.calls == [] and "WARNING" in capsys.readouterr().out
    (run_dir(work, "fsq-c10-E4-laya-s11-head") / "done.json").unlink()
    again = FakeRunner()
    assert run(work, again, "E4") == 0
    assert again.modules() == ["evaluate"] and "ood_extra" in again.calls[0][1]["--splits"]


def test_cleanup_always_keeps_best(work, cfg):
    bare = {**cfg, "phase3": {**cfg["phase3"], "keep_after_run": []}}
    (work / "config.yaml").write_text(yaml.safe_dump(bare, sort_keys=False), encoding="utf-8")
    assert run(work, FakeRunner(), "E4") == 0
    train = run_dir(work, "fsq-c10-E4-laya-s11-head") / "train"
    assert (train / "best" / "model.safetensors").is_file() and not (train / "ckpt").exists()


def test_an_export_check_without_a_temperature_is_redone(work):
    rd = run_dir(work, "fsq-c10-E4-laya-s11-head")
    assert run(work, FakeRunner(fail={"evaluate": 1}), "E4") == 1
    (rd / "export_check.json").write_text(json.dumps({"T": None, "passed": False}), encoding="utf-8")
    again = FakeRunner()
    assert run(work, again, "E4") == 0
    assert again.modules() == ["export_check", "evaluate"]


def test_a_finished_training_without_best_is_an_error_not_a_retrain(work, capsys):
    rd = run_dir(work, "fsq-c10-E4-laya-s11-head")
    assert run(work, FakeRunner(fail={"export_check": 1}), "E4") == 1
    import shutil
    shutil.rmtree(rd / "train" / "best")
    again = FakeRunner()
    assert run(work, again, "E4") == 1
    err = capsys.readouterr().err
    assert again.calls == [] and "finished earlier" in err and "model.safetensors" in err


def test_a_failing_step_names_its_log_and_the_matrix_continues(work, capsys):
    fake = FakeRunner(fail={"export_check": 2})
    assert run(work, fake, "E4", "B2-laya") == 1
    err = capsys.readouterr().err
    rd = run_dir(work, "fsq-c10-E4-laya-s11-head")
    line = next(x for x in err.splitlines() if "fsq-c10-E4-laya-s11-head" in x and "export_check" in x)
    assert "exit 2" in line and str(rd / "logs" / "export_check.log") in line
    assert fake.modules() == ["evaluate", "train_single", "export_check"]  # B2 (first in plan order) ran
    assert (run_dir(work, "fsq-c10-B2-laya-zs") / "done.json").is_file()
    assert not (rd / "done.json").exists()
    assert "1 failed" in err


def test_fail_fast_stops_at_the_first_failure(work):
    fake = FakeRunner(fail={"evaluate": 1})
    assert run(work, fake, "B2", extra=("--fail-fast",)) == 1
    assert fake.modules() == ["evaluate"]


def test_missing_outputs_after_a_zero_exit_fail_the_step(work, capsys):
    fake = FakeRunner(drop_split="ood_brand")
    assert run(work, fake, "B2-laya") == 1
    err = capsys.readouterr().err
    assert "ood_brand" in err and "evaluate.log" in err


def test_a_missing_variant_fails_before_any_subprocess(work, capsys):
    import shutil
    shutil.rmtree(work / "data_c7")
    fake = FakeRunner()
    assert run(work, fake, "E5") == 1
    err = capsys.readouterr().err
    assert fake.calls == []
    assert "python -m laya_poc.variants c7" in err and "data_c7" in err


def test_unknown_only_is_an_error(work, capsys):
    assert run(work, FakeRunner(), "E9") == 1
    assert "matches no run" in capsys.readouterr().err


def test_runs_csv_lists_every_done_run_sorted(work):
    assert run(work, FakeRunner(), "E4", "B2") == 0
    rows = list(csv.DictReader((work / "results" / "runs.csv").open(encoding="utf-8")))
    assert [r["run_name"] for r in rows] == sorted(["fsq-c10-B2-laya-zs", "fsq-c10-B2-laya_ml-zs",
                                                    "fsq-c10-E4-laya-s11-head"])


# ---------------------------------------------------------------- the commands parse with the real CLIs

def _real_parse(module: str, args: list[str]):
    if module == "train_single":
        from laya_poc.schedule import build_parser
        return build_parser().parse_args(args)
    mod = __import__(f"laya_poc.{module}", fromlist=["parse_args"])
    return mod.parse_args(args)


@pytest.mark.parametrize("only", ["E4", "fsq-c7-E5-laya-s11", "fsq-c10-E6-laya-s11-n1000", "B2-laya"])
def test_rendered_commands_parse_with_the_real_parsers(work, only):
    """Every command the matrix renders is accepted by the CLI it calls (train_single, export_check, evaluate)."""
    fake = FakeRunner()
    assert run(work, fake, only) == 0
    for module, o, cmd, _ in fake.calls:
        args = cmd[cmd.index("-m") + 2:]
        ns = _real_parse(module, args)
        if module == "train_single":
            assert ns.save_best and ns.final_eval and ns.freeze_encoder == ("-head" in o["--run-dir"][0])
        if module == "evaluate":
            assert ns.splits == list(SPLITS) and ns.out_dir and ns.preds_dir and ns.order_invariance_out
            assert ns.extra and ns.order_seed == 20260925
