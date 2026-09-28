"""Exit checks when a results root holds only one phase, and for downloaded archives (no weights)."""
import json
from pathlib import Path

from laya_poc import matrix_decision as D
from laya_poc import matrix_report as R
from laya_poc.matrix_plan import RunSpec


E2 = RunSpec(arm="E2", model="laya", scheme="c10", seed=11)            # fsq-c10-E2-laya-s11
B3 = RunSpec(arm="B3", model="tfidf_lr", scheme="c10", kind="tfidf_lr")  # fsq-c10-B3-tfidf_lr


def _csv(path: Path, names: list[str]) -> Path:
    path.write_text("run_name,T,val_macro_f1\n" + "".join(f"{n},1.05,0.57\n" for n in names), encoding="utf-8")
    return path


def _trained_run(root: Path, name: str, *, weights: bool) -> None:
    d = root / name
    (d / "train").mkdir(parents=True)
    (d / "done.json").write_text("{}", encoding="utf-8")
    (d / "train" / "summary.json").write_text(json.dumps({"best_opt_step": 3000}), encoding="utf-8")
    if weights:
        (d / "train" / "best").mkdir()
        (d / "train" / "best" / R.BEST_WEIGHTS).write_bytes(b"w")


def test_phase3_exit_is_not_run_when_no_phase3_run_is_in_the_roots(tmp_path):
    ex = R.exit_check([E2], [tmp_path], _csv(tmp_path / "runs.csv", []))
    assert ex["verdict"] == "NOT RUN" and ex["passed"] is None and ex["total"] == 1


def test_phase4_exit_is_not_run_when_no_baseline_run_is_in_the_roots(tmp_path):
    ex = D.phase4_exit_check([B3], [tmp_path], ["val"])
    assert ex["verdict"] == "NOT RUN" and ex["passed"] is None


def test_archive_without_weights_fails_strictly_but_passes_in_archived_mode(tmp_path):
    _trained_run(tmp_path, "fsq-c10-E2-laya-s11", weights=False)
    csv = _csv(tmp_path / "runs.csv", ["fsq-c10-E2-laya-s11"])
    strict = R.exit_check([E2], [tmp_path], csv)
    archived = R.exit_check([E2], [tmp_path], csv, archived=True)
    assert strict["verdict"] == "FAIL" and "best/" in strict["runs"][0]["missing"]
    assert archived["verdict"] == "PASS"


def test_archived_mode_still_needs_a_recorded_best(tmp_path):
    _trained_run(tmp_path, "fsq-c10-E2-laya-s11", weights=False)
    (tmp_path / "fsq-c10-E2-laya-s11" / "train" / "summary.json").write_text("{}", encoding="utf-8")
    ex = R.exit_check([E2], [tmp_path], _csv(tmp_path / "runs.csv", ["fsq-c10-E2-laya-s11"]),
                      archived=True)
    assert ex["verdict"] == "FAIL"


def test_strict_mode_with_weights_is_unchanged(tmp_path):
    _trained_run(tmp_path, "fsq-c10-E2-laya-s11", weights=True)
    ex = R.exit_check([E2], [tmp_path], _csv(tmp_path / "runs.csv", ["fsq-c10-E2-laya-s11"]))
    assert ex["verdict"] == "PASS" and ex["passed"] is True
