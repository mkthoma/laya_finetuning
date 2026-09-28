import json
import math

import pytest

from laya_poc import smoke_report as S

EXIT = {"parity_max_dp": 0.02, "parity_min_agree": 0.995, "resume_max_rel_dev": 0.02, "min_loss_drop": 0.20,
        "max_vram_gb": 14.0}
PLAN = {"kill_at": 122, "ckpt_every": 40, "compare_steps": 20}
ACC = 4
VAL_CE = (2.2, 1.6)      # control val_ce: --initial-eval (opt 0) -> --final-eval
EVAL_SECONDS, EVAL_N, CKPT_SECONDS = 4.0, 200, 25.0


def _loss(k, n=250, start=2.3, end=1.5):
    return start + (end - start) * (k - 1) / (n - 1)


def _mean_loss(lo, hi):
    return sum(_loss(k) for k in range(lo, hi + 1)) / (hi - lo + 1)


def _micro(k, loss_ce, finite=True):
    return {"t": float(k), "event": "micro", "micro_step": k, "epoch": 0, "loss": loss_ce + 0.3,
            "loss_ce": loss_ce, "loss_rl": 0.3, "reward": 0.1, "sigma": 0.4, "finite": finite}


def _lr_factor(opt_step):
    return 1.0 - opt_step / 100


def _opt(k, scale=65536.0, lr_factor=1.0):
    step = math.ceil(k / ACC)
    f = _lr_factor(step) * lr_factor
    return {"t": float(k), "event": "opt", "opt_step": step, "micro_step": k, "lr_enc": 2e-5 * f,
            "lr_head": 1e-4 * f, "scale": scale, "vram_reserved_gb": 9.0, "sec_per_micro": 0.8}


def _eval(opt_step, val_ce, initial=False):
    ev = {"t": 0.0, "event": "eval", "opt_step": opt_step, "val_ce": val_ce, "val_acc": 0.5, "val_macro_f1": 0.4,
          "n": EVAL_N, "seconds": EVAL_SECONDS}
    return {**ev, "initial": True} if initial else ev


def control_log(n=250, scale=65536.0, loss=_loss, val_ce=VAL_CE):
    ev = [{"t": 0.0, "event": "start", "micro_batch": 8, "grad_accum": ACC}]
    if val_ce[0] is not None:
        ev.append(_eval(0, val_ce[0], initial=True))
    for k in range(1, n + 1):
        ev.append(_micro(k, loss(k)))
        if k % ACC == 0 or k == n:
            ev.append(_opt(k, scale))
    if val_ce[1] is not None:
        ev.append(_eval(math.ceil(n / ACC), val_ce[1]))
    return ev + [{"t": 999.0, "event": "done", "micro_steps": n, "opt_steps": math.ceil(n / ACC)}]


def resumed_log(n=250, kill_at=122, from_step=120, deviation=0.0, loss=_loss, opt_over=lambda opt_step: {}):
    ev = [{"t": 0.0, "event": "start"}]
    for k in range(1, kill_at + 1):
        # pre-crash values after the checkpoint are replayed after the resume: make them differ so a
        # report that used them would be caught
        ev.append(_micro(k, loss(k) + (5.0 if k > from_step else 0.0)))
        if k % ACC == 0:
            ev.append(_opt(k, lr_factor=3.0 if k > from_step else 1.0))
        if k % PLAN["ckpt_every"] == 0:
            ev.append({"t": float(k), "event": "ckpt", "opt_step": k // ACC, "micro_step": k, "seconds": CKPT_SECONDS})
    ev.append({"t": 500.0, "event": "crash_injected", "micro_step": kill_at})
    ev += [{"t": 600.0, "event": "start"},
           {"t": 601.0, "event": "resumed", "path": "/x/ckpt/step0000030.pt", "from_micro_step": from_step,
            "from_opt_step": from_step // ACC, "epoch": 0}]
    for k in range(from_step + 1, n + 1):
        ev.append(_micro(k, loss(k) * (1 + deviation)))
        if k % ACC == 0 or k == n:
            opt = _opt(k)
            ev.append({**opt, **opt_over(opt["opt_step"])})
    return ev + [{"t": 999.0, "event": "done", "micro_steps": n, "opt_steps": math.ceil(n / ACC)}]


def summary(**over):
    base = {"micro_steps": 250, "opt_steps": 63, "resumed_from": None, "peak_vram_alloc_gb": 7.5,
            "peak_vram_reserved_gb": 9.1, "sec_per_micro_median": 0.8, "sec_per_micro_mean": 0.9, "nonfinite": 0,
            "min_scale": 32768.0, "device": "cuda", "card": "T4", "micro_batch": 8, "grad_accum": ACC,
            "model": "laya", "seed": 11}
    return {**base, **over}


def parity(model, **over):
    base = {"model": model, "n": 200, "max_dp": 0.004, "argmax_agree": 1.0, "padded_max_dp": 0.001,
            "padded_nan": False, "nan": False, "passed": True, "device_ref": "cpu", "device_test": "cuda",
            "dtype_test": "float16"}
    return {**base, **over}


def good_inputs():
    return {
        "env": {"gpu_name": "Tesla T4", "capability": [7, 5], "driver": "580.1", "laya_version": "0.3.20",
                "laya_commit": "4066d5d5fbf08b66c6757ddeedbd797bd7655bc0", "transformers": "5.17.0",
                "torch": "2.11.0+cu130", "torch_cuda": "13.0", "warnings": []},
        "data_report": {"split_sizes": {"train": 2000, "val": 200}},
        "zeroshot": {"laya": {"correct": 7, "n": 20}, "laya_ml": {"correct": 6, "n": 20}},
        "parity_laya": parity("laya"), "parity_laya_ml": parity("laya_ml"),
        "control_log": control_log(), "control_summary": summary(),
        "resumed_log": resumed_log(), "resumed_summary": summary(resumed_from=120),
        "crash_exit": {"returncode": -9},
        "export_check": {"T": 1.3, "T_applied": 1.3, "clamped": False, "passed": True, "roundtrip_max_dp": 4e-5,
                         "pre": {"ece": 0.08, "nll": 1.1}, "post": {"ece": 0.03, "nll": 1.0}},
    }


def _by_name(rows):
    return {r["name"]: r for r in rows}


def _evaluate(inputs):
    return _by_name(S.evaluate_exit(inputs, EXIT, **PLAN))


# ---- evaluate_exit ------------------------------------------------------------------------------

def test_all_criteria_pass_on_good_inputs():
    rows = S.evaluate_exit(good_inputs(), EXIT, **PLAN)
    assert [r["name"] for r in rows] == list(S.CRITERIA)
    failed = [r for r in rows if not r["passed"]]
    assert not failed, failed
    for r in rows:
        assert set(r) == {"name", "value", "threshold", "passed", "note"}
    assert S.verdict(rows) == "PASS"


def test_resume_uses_post_resume_events_only():
    r = _evaluate(good_inputs())[S.RESUME_LOSS]
    assert r["passed"] and r["value"] == pytest.approx(0.0, abs=1e-12)
    assert "121..140" in r["note"]
    assert "33 opt steps" in r["note"]   # opt 31..63: lr_enc, lr_head and scale compared at each


def test_resume_detects_lost_scheduler_state():
    # a scheduler restarted at the resume: the loss_ce window alone cannot see it
    log = resumed_log(opt_over=lambda s: {"lr_enc": 2e-5 * _lr_factor(s - 30)} if s > 30 else {})
    r = _evaluate({**good_inputs(), "resumed_log": log})[S.RESUME_LOSS]
    assert r["value"] == pytest.approx(0.0, abs=1e-12)
    assert not r["passed"] and "opt 31 lr_enc" in r["note"]


def test_resume_detects_reset_grad_scaler():
    log = resumed_log(opt_over=lambda s: {"scale": 131072.0} if s >= 40 else {})
    r = _evaluate({**good_inputs(), "resumed_log": log})[S.RESUME_LOSS]
    assert not r["passed"] and "opt 40 scale" in r["note"]


@pytest.mark.parametrize("rel,passed", [(1e-14, True), (1e-9, False)])
def test_resume_lr_tolerance_is_relative_1e_12(rel, passed):
    log = resumed_log(opt_over=lambda s: {"lr_head": 1e-4 * _lr_factor(s) * (1 + rel)})
    r = _evaluate({**good_inputs(), "resumed_log": log})[S.RESUME_LOSS]
    assert r["passed"] is passed
    assert passed or "opt 31 lr_head" in r["note"]


def test_resume_needs_opt_events_after_the_resume_point():
    log = [e for e in resumed_log() if not (e["event"] == "opt" and e["opt_step"] > 30)]
    r = _evaluate({**good_inputs(), "resumed_log": log})[S.RESUME_LOSS]
    assert not r["passed"] and "no opt step" in r["note"]


def test_expected_resume_point():
    assert S.expected_resume_step(122, 40) == 120
    assert S.expected_resume_step(120, 40) == 80   # the crash comes before step 120's checkpoint
    assert S.expected_resume_step(10, 8) == 8


@pytest.mark.parametrize("model,over", [
    ("laya", {"max_dp": 0.05}),
    ("laya", {"argmax_agree": 0.99}),
    ("laya_ml", {"nan": True}),
    ("laya_ml", {"padded_max_dp": 0.03}),
    ("laya", {"device_test": "cpu"}),
    ("laya", {"max_dp": None}),
    ("laya", {"padded_max_dp": None}),  # padded check absent or not computable: parity.passed says FAIL too
    ("laya_ml", {"cpu_fallback": True, "passed": False}),  # Laya answered a batch on CPU after a CUDA OOM
])
def test_parity_failures(model, over):
    inputs = {**good_inputs(), f"parity_{model}": parity(model, **over)}
    rows = _evaluate(inputs)
    assert not rows[f"parity {model}"]["passed"]
    other = "laya_ml" if model == "laya" else "laya"
    assert rows[f"parity {other}"]["passed"]


def test_crash_run_must_exit_non_zero():
    rows = _evaluate({**good_inputs(), "crash_exit": {"returncode": 0}})
    assert not rows[S.CRASH]["passed"]


def test_crash_without_injection_event_is_not_a_drill():
    # a Python exception (rc 1) or the RAM OOM-killer (-9) between checkpoint and kill_at is a real failure
    log = [e for e in resumed_log() if e["event"] != "crash_injected"]
    for rc in (1, -9):
        row = _evaluate({**good_inputs(), "resumed_log": log, "crash_exit": {"returncode": rc}})[S.CRASH]
        assert not row["passed"] and "crash_injected" in row["note"]


def test_crash_injected_at_another_step_fails():
    log = [{**e, "micro_step": 118} if e["event"] == "crash_injected" else e for e in resumed_log()]
    row = _evaluate({**good_inputs(), "resumed_log": log})[S.CRASH]
    assert not row["passed"] and "118" in row["note"]


def test_crash_needs_the_resumed_log():
    row = _evaluate({**good_inputs(), "resumed_log": None})[S.CRASH]
    assert not row["passed"] and "missing resumed/log.jsonl" in row["note"]


def test_resume_missing_event_fails():
    log = [e for e in resumed_log() if e["event"] != "resumed"]
    rows = _evaluate({**good_inputs(), "resumed_log": log})
    assert not rows[S.RESUME_POINT]["passed"] and "resumed" in rows[S.RESUME_POINT]["note"]
    assert not rows[S.RESUME_LOSS]["passed"]


def test_resume_from_wrong_checkpoint_fails():
    rows = _evaluate({**good_inputs(), "resumed_log": resumed_log(from_step=80)})
    assert not rows[S.RESUME_POINT]["passed"]
    assert rows[S.RESUME_POINT]["value"] == 80 and "120" in str(rows[S.RESUME_POINT]["threshold"])
    # the pre-crash opt events 21..30 carry other learning rates; only the replayed ones are compared
    assert rows[S.RESUME_LOSS]["passed"], rows[S.RESUME_LOSS]


def test_resume_loss_deviation_fails():
    rows = _evaluate({**good_inputs(), "resumed_log": resumed_log(deviation=0.05)})
    assert not rows[S.RESUME_LOSS]["passed"]
    assert rows[S.RESUME_LOSS]["value"] == pytest.approx(0.05, rel=1e-6)


def test_resume_loss_needs_every_step_in_window():
    log = [e for e in resumed_log() if not (e["event"] == "micro" and e["micro_step"] == 130
                                            and e["loss_ce"] < 4)]
    rows = _evaluate({**good_inputs(), "resumed_log": log})
    assert not rows[S.RESUME_LOSS]["passed"] and "missing" in rows[S.RESUME_LOSS]["note"]


def test_flat_val_ce_fails_loss_drop():
    rows = _evaluate({**good_inputs(), "control_log": control_log(val_ce=(2.0, 2.0))})
    assert not rows[S.LOSS_DROP]["passed"] and rows[S.LOSS_DROP]["value"] == pytest.approx(0.0)


def test_loss_drop_grades_val_ce_initial_vs_final_eval():
    r = _evaluate(good_inputs())[S.LOSS_DROP]
    assert r["passed"] and r["value"] == pytest.approx((VAL_CE[0] - VAL_CE[1]) / VAL_CE[0])
    assert "val_ce 2.2000 (opt 0" in r["note"] and "1.6000 (opt 63" in r["note"]
    # the 30-micro-step training windows are information only
    assert f"1..30 {_mean_loss(1, 30):.4f}" in r["note"] and f"221..250 {_mean_loss(221, 250):.4f}" in r["note"]


def test_loss_drop_ignores_the_training_loss_windows():
    flat_train = _evaluate({**good_inputs(), "control_log": control_log(loss=lambda k: 2.0)})[S.LOSS_DROP]
    assert flat_train["passed"]
    small = _evaluate({**good_inputs(), "control_log": control_log(val_ce=(2.0, 1.9))})[S.LOSS_DROP]
    assert not small["passed"] and small["value"] == pytest.approx(0.05)


@pytest.mark.parametrize("log", [control_log(val_ce=(None, 1.6)), control_log(val_ce=(2.2, None)),
                                 control_log(val_ce=(2.2, None)) + [_eval(30, 1.5)],    # not at the last opt step
                                 [_eval(63, 1.6)] + control_log(val_ce=(2.2, None)),    # before the initial eval
                                 [{**e, "initial": False} if e["event"] == "eval" else e for e in control_log()]],
                         ids=["no-initial", "no-final", "mid-run-only", "stale-final", "initial-not-flagged"])
def test_loss_drop_needs_the_initial_and_final_eval(log):
    r = _evaluate({**good_inputs(), "control_log": log})[S.LOSS_DROP]
    assert not r["passed"] and r["value"] is None
    assert "missing initial/final eval (control must run with --initial-eval --final-eval)" in r["note"]


def test_nonfinite_loss_fails_numerics():
    log = control_log()
    idx = next(i for i, e in enumerate(log) if e["event"] == "micro" and e["micro_step"] == 50)
    log[idx] = {**_micro(50, float("nan")), "finite": False}
    rows = _evaluate({**good_inputs(), "control_log": log})
    assert not rows[S.NUMERICS]["passed"]


def test_scaler_collapse_fails_numerics():
    rows = _evaluate({**good_inputs(), "control_log": control_log(scale=0.5)})
    assert not rows[S.NUMERICS]["passed"] and "0.5" in rows[S.NUMERICS]["note"]


def test_summary_nonfinite_count_fails_numerics():
    rows = _evaluate({**good_inputs(), "resumed_summary": summary(nonfinite=3)})
    assert not rows[S.NUMERICS]["passed"]


def test_vram_over_budget_fails():
    rows = _evaluate({**good_inputs(), "resumed_summary": summary(peak_vram_reserved_gb=14.5)})
    assert not rows[S.VRAM]["passed"] and rows[S.VRAM]["value"] == 14.5


def test_vram_cap_is_labelled_in_decimal_gigabytes():
    assert "GB (10^9 bytes)" in _evaluate(good_inputs())[S.VRAM]["threshold"]


def test_vram_unknown_fails():
    inputs = {**good_inputs(), "control_summary": summary(peak_vram_reserved_gb=None),
              "resumed_summary": summary(peak_vram_reserved_gb=None)}
    assert not _evaluate(inputs)[S.VRAM]["passed"]


def test_missing_inputs_fail_with_note_and_never_crash():
    rows = S.evaluate_exit({}, EXIT, **PLAN)
    assert len(rows) == len(S.CRITERIA) and not any(r["passed"] for r in rows)
    notes = " ".join(r["note"] for r in rows)
    assert "missing parity_laya.json" in notes and "missing crash_exit.json" in notes
    assert "missing control/log.jsonl" in notes and "missing resumed/log.jsonl" in notes


def test_garbage_inputs_fail_instead_of_crashing():
    inputs = {**good_inputs(), "parity_laya": {"max_dp": "oops"}, "control_log": [{"event": "micro"}]}
    rows = _evaluate(inputs)
    assert not rows["parity laya"]["passed"] and not rows[S.LOSS_DROP]["passed"]


# ---- CLI -----------------------------------------------------------------------------------------

def _write(run_root, rel, data):
    p = run_root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, list):
        p.write_text("".join(json.dumps(e) + "\n" for e in data), encoding="utf-8")
    else:
        p.write_text(json.dumps(data), encoding="utf-8")


def _write_run_root(tmp_path, inputs):
    run_root, data_dir = tmp_path / "runs" / "smoke", tmp_path / "data"
    for key, rel in S.INPUT_FILES.items():
        if inputs.get(key) is not None:
            _write(run_root, rel, inputs[key])
    _write(data_dir, "data_report.json", inputs["data_report"])
    return run_root, data_dir


def test_cli_pass(tmp_path, capsys):
    run_root, data_dir = _write_run_root(tmp_path, good_inputs())
    out = run_root / "smoke_report.md"
    rc = S.main(["--run-root", str(run_root), "--data-dir", str(data_dir), "--out", str(out)])
    assert rc == 0
    assert "Verdict: **PASS**" in out.read_text(encoding="utf-8")
    js = json.loads((run_root / "smoke_report.json").read_text(encoding="utf-8"))
    assert js["verdict"] == "PASS" and len(js["criteria"]) == len(S.CRITERIA)
    assert "PASS" in capsys.readouterr().out


def test_cli_default_data_dir_and_truncated_log_line(tmp_path):
    run_root, _ = _write_run_root(tmp_path, good_inputs())
    with (run_root / "resumed" / "log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"t": 1.0, "event": "mic')  # a SIGKILL can leave half a line
    out = tmp_path / "report.md"
    assert S.main(["--run-root", str(run_root), "--out", str(out)]) == 0
    js = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert js["verdict"] == "PASS"
    assert any("skipped 1 unreadable line" in n for n in js["notes"])


def test_cli_empty_run_root_reports_fail_without_crashing(tmp_path, capsys):
    out = tmp_path / "r.md"
    rc = S.main(["--run-root", str(tmp_path / "nothing"), "--out", str(out)])
    assert rc == 0
    md = out.read_text(encoding="utf-8")
    assert "Verdict: **FAIL**" in md and "missing env.json" in md
    assert "FAIL" in capsys.readouterr().out


def test_cli_malformed_summary_still_writes_the_report(tmp_path):
    run_root, data_dir = _write_run_root(tmp_path, {**good_inputs(), "control_summary": [1, 2]})
    out = run_root / "smoke_report.md"
    assert S.main(["--run-root", str(run_root), "--data-dir", str(data_dir), "--out", str(out)]) == 0
    assert "Verdict: **FAIL**" in out.read_text(encoding="utf-8")
    assert json.loads((run_root / "smoke_report.json").read_text(encoding="utf-8"))["verdict"] == "FAIL"


def test_cli_plan_overrides_for_tiny_dry_run(tmp_path):
    n, kill, every = 24, 10, 8
    inputs = {**good_inputs(), "control_log": control_log(n=n, loss=lambda k: _loss(k, n=n)),
              "resumed_log": resumed_log(n=n, kill_at=kill, from_step=every, loss=lambda k: _loss(k, n=n))}
    run_root, data_dir = _write_run_root(tmp_path, inputs)
    out = run_root / "smoke_report.md"
    assert S.main(["--run-root", str(run_root), "--data-dir", str(data_dir), "--out", str(out),
                   "--kill-at", str(kill), "--ckpt-every", str(every), "--compare-steps", "8"]) == 0
    crit = _by_name(json.loads((run_root / "smoke_report.json").read_text(encoding="utf-8"))["criteria"])
    assert crit[S.RESUME_POINT]["passed"] and crit[S.RESUME_LOSS]["passed"]
    assert "9..16" in crit[S.RESUME_LOSS]["note"]


@pytest.mark.parametrize("val_ce,passed,doc_note", [((2.0, 1.5), True, "met"), ((2.0, 1.7), True, "not met"),
                                                   ((2.0, 1.9), False, "not met")])
def test_loss_drop_grades_the_configured_threshold_and_reports_the_doc_target(val_ce, passed, doc_note):
    # 10% is graded (decided 2026-09-26); the doc's 20% is only reported.
    exit_cfg = {**EXIT, "min_loss_drop": 0.10, "doc_loss_drop": 0.20}
    inputs = {**good_inputs(), "control_log": control_log(val_ce=val_ce)}
    r = _by_name(S.evaluate_exit(inputs, exit_cfg, **PLAN))[S.LOSS_DROP]
    assert r["passed"] is passed
    assert f"doc §6.2 target 20% (info): {doc_note}" in r["note"]
