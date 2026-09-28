import json
import subprocess
from types import SimpleNamespace

import pytest

from laya_poc import bench_cpu as B
from laya_poc import labels as L
from laya_poc.io_utils import write_jsonl


def _rows_file(tmp_path, n=6):
    rows = [{"id": f"test_id-{i:06d}", "state": json.dumps({"country": "GB", "name": f"Rosa Pizza {i}"},
                                                            separators=(",", ":")), "label": "dining"}
            for i in range(n)]
    path = tmp_path / "test_id.jsonl"
    write_jsonl(path, rows)
    return path


def _ok(threads, **extra):
    res = {"threads": threads, "cold_s": 1.5, "p50_ms": 40.0, "p95_ms": 60.0, "mean_ms": 45.0, "batch_rps": 30.0,
           "peak_rss_gb": 1.2, **extra}
    return SimpleNamespace(returncode=0, stdout=f"some laya warning\n{B.MARKER}{json.dumps(res)}\n", stderr="")


# ---------------------------------------------------------------- pure helpers

def test_latency_stats_percentiles():
    s = B.latency_stats([float(x) for x in range(1, 101)])
    assert s["p50_ms"] == pytest.approx(50.5) and s["p95_ms"] == pytest.approx(95.05)
    assert s["mean_ms"] == pytest.approx(50.5) and s["max_ms"] == 100.0


def test_budget_flags_are_information_against_cpu_budget(cfg):
    budget = cfg["cpu_budget"]
    assert B.budget_flags({"p95_ms": 400.0, "batch_rps": 9.0}, budget) == {"p95_ok": True, "rps_ok": True}
    assert B.budget_flags({"p95_ms": 600.0, "batch_rps": 7.0}, budget) == {"p95_ok": False, "rps_ok": False}


def test_parse_worker_output_finds_the_marker_line_among_noise():
    out = f"Warning: something\n{B.MARKER}{json.dumps({'threads': 2})}\ntrailing log line\n"
    assert B.parse_worker_output(out) == {"threads": 2}
    assert B.parse_worker_output("no result here\n") is None


def test_worker_env_pins_threads_hides_the_gpu_and_forces_fp32():
    base = {"PATH": "x", "LAYA_CPU_AMP": "bf16", "CUDA_VISIBLE_DEVICES": "0"}
    env = B.worker_env(2, base)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LAYA_THREADS"):
        assert env[k] == "2"
    assert env["CUDA_VISIBLE_DEVICES"] == "" and "LAYA_CPU_AMP" not in env and env["PATH"] == "x"
    assert base["LAYA_CPU_AMP"] == "bf16" and base["CUDA_VISIBLE_DEVICES"] == "0"  # input untouched


def test_worker_cmd_parses_back_into_a_one_setting_worker(tmp_path):
    args = B.parse_args(["--ckpt", "hub", "--rows", str(tmp_path / "r.jsonl"), "--out", str(tmp_path / "o.json"),
                         "--threads", "1", "2", "--n", "7", "--warmup", "3", "--batch-size", "4"])
    cmd = B.worker_cmd(args, 2)
    assert cmd[1:3] == ["-m", "laya_poc.bench_cpu"]
    w = B.parse_args(cmd[3:])
    assert w.worker_threads == 2 and w.n == 7 and w.warmup == 3 and w.batch_size == 4 and w.ckpt == "hub"


def test_peak_rss_is_reported_with_its_source():
    gb, source = B.peak_rss_gb()
    assert gb is not None and gb > 0 and isinstance(source, str) and source


# ---------------------------------------------------------------- parent with a fake runner

def _args(tmp_path, *extra):
    return ["--ckpt", "hub", "--rows", str(_rows_file(tmp_path)), "--out", str(tmp_path / "bench.json"),
            "--n", "4", "--warmup", "1", *extra]


def test_run_collects_one_json_line_per_thread_setting(tmp_path, monkeypatch, capsys):
    calls = []

    def runner(cmd, env, timeout):
        threads = int(cmd[cmd.index("--worker-threads") + 1])
        calls.append((threads, env["OMP_NUM_THREADS"]))
        if threads == 2:
            return SimpleNamespace(returncode=1, stdout="", stderr="Traceback ...\nRuntimeError: boom\n")
        return _ok(threads)

    monkeypatch.setattr(B, "run_worker", runner)
    monkeypatch.setattr(B, "prefetch", lambda source: None)
    assert B.main(_args(tmp_path, "--threads", "1", "2")) == 0
    res = json.loads((tmp_path / "bench.json").read_text(encoding="utf-8"))
    assert calls == [(1, "1"), (2, "2")]
    assert [r["threads"] for r in res["results"]] == [1] and res["results"][0]["p95_ok"] is True
    assert res["errors"] == [{"threads": 2, "returncode": 1, "error": "RuntimeError: boom"}]
    assert res["backend"] == "torch" and res["budget"]["p95_ms"] == 500 and "Phase 5" in res["budget"]["note"]
    for k in ("cpu_count", "platform", "n", "warmup", "batch_size", "model", "ckpt", "seconds"):
        assert k in res
    assert len(capsys.readouterr().out.strip().splitlines()) <= 6


def test_run_fails_when_no_setting_succeeds_but_keeps_the_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(B, "run_worker", lambda cmd, env, timeout: SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(B, "prefetch", lambda source: None)
    assert B.main(_args(tmp_path, "--threads", "1")) == 1
    res = json.loads((tmp_path / "bench.json").read_text(encoding="utf-8"))
    assert res["results"] == [] and res["errors"][0]["error"].startswith("no result line")


def test_run_records_a_worker_timeout(tmp_path, monkeypatch):
    def runner(cmd, env, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(B, "run_worker", runner)
    monkeypatch.setattr(B, "prefetch", lambda source: None)
    assert B.main(_args(tmp_path, "--threads", "1", "--timeout-min", "1")) == 1
    res = json.loads((tmp_path / "bench.json").read_text(encoding="utf-8"))
    assert "timed out" in res["errors"][0]["error"]


def test_onnx_backend_is_a_clear_phase_5_not_implemented(tmp_path, capsys):
    assert B.main(_args(tmp_path, "--backend", "onnx")) == 1
    err = capsys.readouterr().err
    assert "NotImplementedError" in err and "Phase 5" in err


def test_relative_ckpt_is_rejected(tmp_path, capsys):
    assert B.main(["--ckpt", "rel/best", "--rows", str(_rows_file(tmp_path)), "--out", str(tmp_path / "b.json")]) == 1
    assert "absolute" in capsys.readouterr().err


# ---------------------------------------------------------------- real tiny checkpoint

@pytest.fixture
def restore_torch_threads():
    """measure() is a worker body: it sets torch's process-wide thread count. Restore it so later tests in the
    same pytest process are not silently pinned to 1 thread (that slowed and flaked test_train_single)."""
    import torch

    before = torch.get_num_threads()
    yield
    torch.set_num_threads(before)


@pytest.mark.torch
def test_measure_in_process_on_the_tiny_checkpoint(tiny_ckpt_dir, restore_torch_threads):
    states = [json.dumps({"country": "GB", "name": f"Rosa Pizza {i}"}, separators=(",", ":")) for i in range(5)]
    res = B.measure(tiny_ckpt_dir, states, L.question("c10"), threads=1, warmup=1, batch_size=2, max_len=512,
                    head_max_len=192)
    assert res["threads"] == 1 and res["torch_threads"] == 1 and res["n"] == 5 and res["device"] == "cpu"
    assert res["cold_s"] > 0 and 0 < res["p50_ms"] <= res["p95_ms"] <= res["max_ms"] and res["batch_rps"] > 0
    assert res["amp"] is False and res["peak_rss_gb"] > 0


@pytest.mark.torch
def test_cli_runs_a_fresh_worker_process(tiny_ckpt_dir, tmp_path):
    out = tmp_path / "bench.json"
    rc = B.main(["--ckpt", str(tiny_ckpt_dir), "--rows", str(_rows_file(tmp_path)), "--out", str(out),
                 "--threads", "1", "--n", "3", "--warmup", "1", "--batch-size", "2"])
    assert rc == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["errors"] == [] and len(res["results"]) == 1
    r = res["results"][0]
    assert r["threads"] == 1 and r["n"] == 3 and r["p95_ms"] > 0 and r["batch_rps"] > 0 and r["cold_s"] > 0
