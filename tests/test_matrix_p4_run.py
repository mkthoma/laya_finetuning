"""Phase 4 baselines through the matrix (spec P4 §5), run layer: `matrix run` dispatch to baseline_runs /
small_encoder / llm_baseline with the subprocess layer mocked (a fake runner writes the outputs the real CLIs
would): runs root per phase, --skip-optional, --init per model kind, --baseline-extra, done.json with kind, B4
cleanup, runs.csv with `kind`, and the rendered commands parsed by the real CLIs when they exist."""
from __future__ import annotations

import csv
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from laya_poc import matrix as M
from laya_poc import matrix_results as S
from laya_poc.io_utils import write_jsonl

from test_matrix_p4 import P4_NAMES, SPLITS
from test_matrix_run import FakeRunner, _data_dir, _json, _metrics, _rows


# ---------------------------------------------------------------- matrix run (mocked subprocesses)

def _split_outputs(run: Path, splits, *, subset: bool = False) -> None:
    for split in splits:
        res = {"split": split, "temperature": 1.1, "pre": _metrics(0.4, 0.07), "post": _metrics(0.4, 0.04),
               **({"subset": True} if subset else {})}
        _json(run / "eval" / f"{split}.json", res)
        write_jsonl(run / "preds" / f"{split}.jsonl", [{"id": "x", "y": 0}])


class FakeP4Runner(FakeRunner):
    """FakeRunner plus the Phase 4 CLIs: each writes the spec P4 §1 layout into --run-dir."""

    def __init__(self, fail=None, drop: str | None = None):
        super().__init__(fail)
        self.drop = drop

    def _layout(self, o, *, order: bool, subset: bool = False) -> Path:
        run = Path(o["--run-dir"][0])
        names = list(o["--splits"]) + [e.split("=", 1)[0] for e in o.get("--extra", [])]
        _split_outputs(run, dict.fromkeys(names), subset=subset)
        if self.drop != "calibration.json":
            _json(run / "calibration.json", {"T": 0.9, "clamped": False, "fitted_on": "val", "n": 8, "extra": {}})
        if order:
            _json(run / "order_invariance.json", {"split": "test_id", "n": 8, "perms": 5, "mean_agreement": 0.97,
                                                  "passed_99": False})
        return run

    def baseline_runs(self, o) -> None:
        self._layout(o, order="--order-invariance" in o)

    def small_encoder(self, o) -> None:
        run = self._layout(o, order=True)
        (run / "train" / "ckpt").mkdir(parents=True, exist_ok=True)
        (run / "train" / "best").mkdir(parents=True, exist_ok=True)
        (run / "train" / "best" / "model.safetensors").write_bytes(b"w")
        _json(run / "train" / "summary.json", {"epochs": 3, "epochs_run": 2.5, "best_epoch": 2, "best_opt_step": 48,
                                               "stop_reason": "epochs", "seconds": 240.0,
                                               "n_train_rows_per_epoch": [8, 8, 8]})

    def llm_baseline(self, o) -> None:
        self._layout(o, order=False, subset=True)


def _positional(cmd: list[str]) -> str:
    return cmd[cmd.index("-m") + 2]


@pytest.fixture()
def work(tmp_path, cfg) -> Path:
    """A work dir with data (train.jsonl, 3 epoch files, every eval split), data_c7, data_eval and config.yaml."""
    w = tmp_path / "work"
    for d in ("data", "data_c7"):
        _data_dir(w / d)
        _rows(w / d / "train.jsonl", 8, "train")
    _rows(w / "data_eval" / "trap_candidates.jsonl", 4, "trap_candidates")
    _rows(w / "data_eval" / "trap_candidates_c7.jsonl", 4, "trap_candidates")
    (w / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return w


def run(work: Path, runner: FakeRunner, *only: str, extra: tuple[str, ...] = ()) -> int:
    argv = ["run", "--only", *only, "--config", str(work / "config.yaml"), "--work", str(work),
            "--device", "cpu", "--card", "CPU", *extra]
    return M.main(argv, runner=runner)


def p4_dir(work: Path, name: str) -> Path:
    return work / "runs" / "p4" / name


def _csv(work: Path) -> tuple[list[str], list[dict]]:
    with (work / "results" / "runs.csv").open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def test_b1_runs_majority_and_prior_per_scheme_through_baseline_runs(work):
    fake = FakeP4Runner()
    assert run(work, fake, "B1") == 0
    assert fake.modules() == ["baseline_runs"] * 4
    assert [_positional(c[2]) for c in fake.calls] == ["majority", "prior", "majority", "prior"]
    for (_, o, _, log), name, scheme in zip(fake.calls, P4_NAMES[:4], ("c10", "c10", "c7", "c7")):
        rd = p4_dir(work, name)
        data = work / ("data" if scheme == "c10" else "data_c7")
        traps = work / "data_eval" / ("trap_candidates.jsonl" if scheme == "c10" else "trap_candidates_c7.jsonl")
        assert o["--scheme"] == [scheme] and o["--data-dir"] == [str(data)] and o["--run-dir"] == [str(rd)]
        assert o["--splits"] == list(SPLITS) and o["--extra"] == [f"trap_candidates={traps}"]
        assert o["--config"] == [str(rd / "run_config.yaml")] and log == rd / "logs" / "baseline.log"
        assert not {"--order-invariance", "--device", "--init"} & set(o)
        kind = name.rsplit("-", 1)[1]
        done = json.loads((rd / "done.json").read_text(encoding="utf-8"))
        assert (done["kind"], done["arm"], done["model"], done["seed"], done["card"]) == (kind, "B1", kind, None, "CPU")
    header, rows = _csv(work)
    assert header == [*S.CSV_COLUMNS, "kind"]  # the Phase 3 columns in order, `kind` at the end
    assert [r["run_name"] for r in rows] == sorted(P4_NAMES[:4]) and {r["T"] for r in rows} == {"0.9"}
    assert {r["kind"] for r in rows} == {"majority", "prior"} and all(r["epochs_run"] == "" for r in rows)


def test_b3_asks_for_order_invariance(work):
    fake = FakeP4Runner()
    assert run(work, fake, "fsq-c10-B3-tfidf_lr") == 0
    o = fake.calls[0][1]
    assert _positional(fake.calls[0][2]) == "tfidf_lr" and "--order-invariance" in o
    rows = _csv(work)[1]
    assert rows[0]["order_invariance"] == "0.97" and rows[0]["kind"] == "tfidf_lr"


def test_b4_trains_a_small_encoder_and_cleanup_keeps_best(work):
    fake = FakeP4Runner()
    assert run(work, fake, "fsq-c10-B4-modernbert_base-s22") == 0
    module, o, _, log = fake.calls[0]
    rd = p4_dir(work, "fsq-c10-B4-modernbert_base-s22")
    assert module == "small_encoder" and log == rd / "logs" / "small_encoder.log"
    assert (o["--model"], o["--seed"], o["--scheme"], o["--device"]) == (["modernbert_base"], ["22"], ["c10"],
                                                                         ["cpu"])
    assert o["--data-dir"] == [str(work / "data")] and o["--run-dir"] == [str(rd)] and "--init" not in o
    assert not (rd / "train" / "ckpt").exists() and (rd / "train" / "best" / "model.safetensors").is_file()
    done = json.loads((rd / "done.json").read_text(encoding="utf-8"))
    assert (done["kind"], done["seed"]) == ("small_encoder", 22)
    row = _csv(work)[1][0]
    assert (row["epochs_run"], row["train_seconds"], row["best_opt_step"], row["stop_reason"], row["T"]) == (
        "2.5", "240.0", "48", "epochs", "0.9")


def test_b5_is_skipped_with_skip_optional_and_runs_otherwise(work, capsys):
    fake = FakeP4Runner()
    assert run(work, fake, "B5", extra=("--skip-optional",)) == 0
    out = capsys.readouterr().out
    assert fake.calls == [] and "optional, disabled (--skip-optional)" in out
    assert not p4_dir(work, "fsq-c10-B5-qwen3_4b").exists()
    assert run(work, fake, "B5") == 0
    module, o, _, _ = fake.calls[0]
    assert module == "llm_baseline" and o["--model"] == ["qwen3_4b"] and o["--device"] == ["cpu"]
    assert o["--splits"] == list(SPLITS) and "--order-invariance" not in o
    assert (p4_dir(work, "fsq-c10-B5-qwen3_4b") / "done.json").is_file()


def test_skip_optional_leaves_the_other_runs_alone(work):
    fake = FakeP4Runner()
    assert run(work, fake, "fsq-c10-B1-prior", "B5", extra=("--skip-optional",)) == 0
    assert fake.modules() == ["baseline_runs"]


def test_a_local_init_goes_to_b4_and_b5_only(work, tmp_path, capsys):
    model = tmp_path / "tiny_encoder"
    model.mkdir()
    fake = FakeP4Runner()
    assert run(work, fake, "fsq-c10-B1-prior", "fsq-c10-B4-mmbert_small-s11", extra=("--init", str(model))) == 0
    b1, b4 = (c[1] for c in fake.calls)
    assert "--init" not in b1 and b4["--init"] == [str(model)]
    again = FakeP4Runner()
    assert run(work, again, "B4", "B5", extra=("--init", str(model))) == 1
    assert again.calls == [] and "one kind per call" in capsys.readouterr().err
    assert run(work, again, "B4", "B5", extra=("--init", str(model), "--skip-optional")) == 0  # B5 skipped


def test_baseline_extra_is_appended_last_and_its_epochs_count(work, capsys):
    (work / "data" / "train_e2.jsonl").unlink()  # 2 epoch files: enough for --epochs 2, not for the config's 3
    fake = FakeP4Runner()
    assert run(work, fake, "fsq-c10-B4-mmbert_small-s33",
               extra=("--baseline-extra=--max-steps 2", "--baseline-extra=--epochs 2")) == 0
    assert fake.calls[0][2][-4:] == ["--max-steps", "2", "--epochs", "2"]
    assert run(work, FakeP4Runner(), "fsq-c10-B4-mmbert_small-s11") == 1
    assert "train_e2.jsonl" in capsys.readouterr().err


def test_done_baselines_are_skipped_and_a_failed_one_reruns(work, capsys):
    assert run(work, FakeP4Runner(fail={"baseline_runs": 3}), "fsq-c7-B3-tfidf_lr") == 1
    err = capsys.readouterr().err
    rd = p4_dir(work, "fsq-c7-B3-tfidf_lr")
    assert "exit 3" in err and str(rd / "logs" / "baseline.log") in err and not (rd / "done.json").exists()
    second = FakeP4Runner()
    assert run(work, second, "fsq-c7-B3-tfidf_lr") == 0
    assert second.modules() == ["baseline_runs"] and (rd / "logs" / "baseline_2.log").is_file()
    third = FakeP4Runner()
    assert run(work, third, "fsq-c7-B3-tfidf_lr") == 0
    assert third.calls == [] and "skip: done.json exists" in capsys.readouterr().out


def test_missing_outputs_after_a_zero_exit_fail_the_run(work, capsys):
    assert run(work, FakeP4Runner(drop="calibration.json"), "fsq-c10-B1-prior") == 1
    err = capsys.readouterr().err
    assert "calibration.json" in err and "baseline.log" in err


def test_a_done_baseline_missing_a_new_split_warns(work, cfg, capsys):
    assert run(work, FakeP4Runner(), "fsq-c10-B1-prior") == 0
    grown = {**cfg, "phase3": {**cfg["phase3"], "eval_splits": [*SPLITS, "ood_extra"]}}
    (work / "config.yaml").write_text(yaml.safe_dump(grown, sort_keys=False), encoding="utf-8")
    (work / "data" / "ood_extra.jsonl").write_text("", encoding="utf-8")
    quiet = FakeP4Runner()
    assert run(work, quiet, "fsq-c10-B1-prior") == 0
    out = capsys.readouterr().out
    assert quiet.calls == [] and "WARNING" in out and "ood_extra" in out and "baseline runs again" in out


def test_laya_and_baseline_runs_go_to_their_phase_roots_and_share_runs_csv(work):
    for n in (1000, 3000, 10000):
        _data_dir(work / f"data_lc{n}", eval_splits=False)
    fake = FakeP4Runner()
    assert run(work, fake, "E4", "fsq-c10-B1-majority") == 0
    laya = work / "runs" / "p3" / "fsq-c10-E4-laya-s11-head"
    assert (laya / "done.json").is_file() and (p4_dir(work, "fsq-c10-B1-majority") / "done.json").is_file()
    assert "kind" not in json.loads((laya / "done.json").read_text(encoding="utf-8"))  # Phase 3 done.json unchanged
    header, rows = _csv(work)
    assert header[-1] == "kind" and {r["run_name"]: r["kind"] for r in rows} == {
        "fsq-c10-B1-majority": "majority", "fsq-c10-E4-laya-s11-head": "laya"}


def test_runs_root_override_puts_every_selected_run_there(work):
    fake = FakeP4Runner()
    assert run(work, fake, "fsq-c10-B1-prior", extra=("--runs-root", "runs/custom")) == 0
    assert (work / "runs" / "custom" / "fsq-c10-B1-prior" / "done.json").is_file()
    assert not (work / "runs" / "p4").exists()


def test_a_missing_c7_variant_fails_before_any_subprocess(work, capsys):
    import shutil
    shutil.rmtree(work / "data_c7")
    fake = FakeP4Runner()
    assert run(work, fake, "fsq-c7-B1-prior") == 1
    assert fake.calls == [] and "python -m laya_poc.variants c7" in capsys.readouterr().err


# ---------------------------------------------------------------- the commands parse with the real CLIs

def _real_parser_args(module: str, args: list[str]):
    mod = importlib.import_module(f"laya_poc.{module}")
    if hasattr(mod, "parse_args"):
        return mod.parse_args(args)
    return mod.build_parser().parse_args(args)


@pytest.mark.parametrize("only,module", [("fsq-c10-B1-majority", "baseline_runs"),
                                         ("fsq-c7-B1-prior", "baseline_runs"),
                                         ("fsq-c10-B3-tfidf_lr", "baseline_runs"),
                                         ("fsq-c10-B4-mmbert_small-s11", "small_encoder"),
                                         ("fsq-c10-B5-qwen3_4b", "llm_baseline")])
def test_rendered_baseline_commands_parse_with_the_real_parsers(work, tmp_path, only, module):
    """Every command the matrix renders for a baseline is accepted by the CLI it calls (spec P4 §2-§4)."""
    if importlib.util.find_spec(f"laya_poc.{module}") is None:
        pytest.skip(f"laya_poc.{module} does not exist yet (written by another Phase 4 agent)")
    init = tmp_path / "local_model"
    init.mkdir()
    extra = ("--init", str(init)) if module != "baseline_runs" else ()
    fake = FakeP4Runner()
    assert run(work, fake, only, extra=extra) == 0
    (called, o, cmd, _), = fake.calls
    assert called == module
    ns = vars(_real_parser_args(module, cmd[cmd.index("-m") + 2:]))
    assert ns["scheme"] == o["--scheme"][0] and Path(ns["run_dir"]) == Path(o["--run-dir"][0])
    assert Path(ns["data_dir"]) == Path(o["--data-dir"][0]) and Path(ns["config"]) == Path(o["--config"][0])
    assert list(ns["splits"]) == list(SPLITS) and [str(e) for e in ns["extra"]] == o["--extra"]
    if module != "baseline_runs":
        assert ns["model"] == o["--model"][0] and ns["device"] == "cpu" and str(ns["init"]) == str(init)
    if module == "small_encoder":
        assert int(ns["seed"]) == int(o["--seed"][0])
    if only == "fsq-c10-B3-tfidf_lr":
        assert ns["order_invariance"] is True
