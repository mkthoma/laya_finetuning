"""smoke_render: the §7.13 rows, the E1 wall-clock extrapolation and the markdown (inputs from test_smoke_report)."""
import pytest

from laya_poc import smoke_render as R
from laya_poc import smoke_report as S
from test_smoke_report import (CKPT_SECONDS, EVAL_N, EVAL_SECONDS, EXIT, PLAN, _mean_loss, control_log, good_inputs,
                               resumed_log, summary)

E1 = {"train_size": 25000, "stripped_rate": 0.07, "epochs": 4, "val_size": 3000, "eval_every_opt_steps": 250,
      "ckpt_every_min": 15}
LOSS_ROW = "Loss at step 0 / 120 / 250; NaN?"
E1_ROW = "Extrapolated E1 wall-clock (h) = s/step × steps"


def test_e1_micro_steps():
    assert R.e1_micro_steps(25000, 0.07, 4, 8) == 3344 * 4


def _info(inputs):
    return R.collect_info(inputs, S.evaluate_exit(inputs, EXIT, **PLAN), e1=E1, kill_at=122)


def test_collect_info_extrapolates_e1_hours_from_the_mean_plus_eval_and_checkpoints():
    info = _info(good_inputs())
    train_s = 0.9 * 3344 * 4                          # mean s/micro x E1 micro-steps
    n_evals = 3344 // 250 + 1                         # 3344 opt steps (ACC 4): 13 periodic + the final eval
    eval_s = n_evals * 3000 * EVAL_SECONDS / EVAL_N   # the final eval's s/row x E1 val rows
    ckpt_s = int((train_s + eval_s) // (15 * 60)) * CKPT_SECONDS
    assert info["sec_per_micro"] == 0.9
    assert info["e1_train_hours"] == pytest.approx(train_s / 3600)
    assert info["e1_hours"] == pytest.approx((train_s + eval_s + ckpt_s) / 3600)
    text = info["rows"][E1_ROW]
    assert "mean" in text and f"{n_evals} evals" in text and "14 checkpoints" in text
    assert "7/20" in info["rows"]["Zero-shot on 20 records: correct / 20 (each checkpoint)"]


def test_e1_estimate_falls_back_to_the_median_and_says_what_it_leaves_out():
    inputs = {**good_inputs(), "control_summary": summary(sec_per_micro_mean=None),
              "control_log": control_log(val_ce=(2.2, None)),
              "resumed_log": [e for e in resumed_log() if e["event"] != "ckpt"]}
    info = _info(inputs)
    assert info["sec_per_micro"] == 0.8 and info["e1_hours"] == pytest.approx(0.8 * 3344 * 4 / 3600)
    text = info["rows"][E1_ROW]
    assert "median" in text and "eval time excluded" in text and "checkpoint time excluded" in text


def test_template_loss_row_reports_window_means_and_val_ce():
    row = _info(good_inputs())["rows"][LOSS_ROW]
    assert row.startswith("mean loss_ce")
    for lo, hi in ((1, 10), (111, 130), (241, 250)):
        assert f"{lo}..{hi}: {_mean_loss(lo, hi):.4f}" in row
    assert "val_ce 2.2000 (initial) -> 1.6000 (final)" in row and "NaN/inf: none" in row
    no_evals = _info({**good_inputs(), "control_log": control_log(val_ce=(None, None))})["rows"][LOSS_ROW]
    assert "val_ce n/a" in no_evals and "1..10" in no_evals


@pytest.mark.parametrize("control_summary", [
    [1, 2],
    summary(sec_per_micro_mean="0.8"),
    summary(micro_batch="8"),
    summary(micro_batch=0),
    summary(micro_batch=True),
    summary(sec_per_micro_mean=float("nan")),
    summary(sec_per_micro_mean=None, sec_per_micro_median=float("nan")),
])
def test_collect_info_survives_malformed_control_summary(control_summary):
    inputs = {**good_inputs(), "control_summary": control_summary}
    rows = S.evaluate_exit(inputs, EXIT, **PLAN)
    info = R.collect_info(inputs, rows, e1=E1, kill_at=122)
    assert info["e1_hours"] is None
    assert "n/a" in info["rows"][E1_ROW]
    R.render_markdown(rows, info, S.verdict(rows), notes=[])


@pytest.mark.parametrize("key", ["control_log", "resumed_log"])
def test_collect_info_survives_malformed_logs(key):
    inputs = {**good_inputs(), key: [1, {"event": "eval", "opt_step": "x"}, {"event": "ckpt", "seconds": "5"}]}
    rows = S.evaluate_exit(inputs, EXIT, **PLAN)
    R.render_markdown(rows, R.collect_info(inputs, rows, e1=E1, kill_at=122), S.verdict(rows), notes=[])


def _export_row(**over):
    inputs = {**good_inputs(), "export_check": {**good_inputs()["export_check"], **over}}
    info = R.collect_info(inputs, S.evaluate_exit(inputs, EXIT, **PLAN), e1=E1, kill_at=122)
    return info["extra"]["Export / calibration round trip"]


def test_export_row_passes_only_when_fitted_t_is_applied():
    assert _export_row().startswith("PASS")
    clamped = _export_row(T=0.05, T_applied=0.5, clamped=True, calibration_ok=False)
    assert clamped.startswith("WARN") and "[0.5, 5]" in clamped
    assert _export_row(T=0.05, T_applied=0.5, clamped=True).startswith("WARN")  # JSON without calibration_ok


def test_export_row_shows_failed_cross_checks():
    row = _export_row(passed=False, train_eval={"ok": False, "note": "acc 0.6100 vs training 0.3000"},
                      budgets={"ok": True, "note": "max_len 512, head_max_len 192"})
    assert row.startswith("FAIL") and "training 0.3000" in row and "budgets ok" in row


def test_render_markdown_contains_template_checklist_and_verdict():
    inputs = good_inputs()
    rows = S.evaluate_exit(inputs, EXIT, **PLAN)
    info = R.collect_info(inputs, rows, e1={"train_size": 25000, "stripped_rate": 0.07, "epochs": 4}, kill_at=122)
    md = R.render_markdown(rows, info, S.verdict(rows), notes=[])
    assert "| GPU / CC / driver | Tesla T4" in md
    assert "Verdict: **PASS**" in md
    for name in S.CRITERIA:
        assert name in md
    rows[0] = {**rows[0], "passed": False}
    assert "Verdict: **FAIL**" in R.render_markdown(rows, info, S.verdict(rows), notes=["missing x"])
