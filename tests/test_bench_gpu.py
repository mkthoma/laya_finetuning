"""bench_gpu: the plan and fail-fast device check, the parent with a fake runner (one fresh process per model, a failed
model or batch size does not stop the rest, the JSON's plan / machine / rows), one REAL end-to-end run on tiny models
with --device cpu, and a real CUDA run that only a GPU host executes. The worker: test_bench_gpu_worker.py."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from laya_poc import bench_gpu as G
from laya_poc import bench_gpu_worker as W
from laya_poc.bench_sources import Source, model_kind
from laya_poc.io_utils import write_jsonl

ALL4 = ["laya", "laya_ml", "modernbert_base", "mmbert_small"]
GIB = 2 ** 30
FAKE_GPU = {"gpu": "Tesla T4", "capability": "7.5", "vram_total_gb": 14.56, "cuda": "12.8", "cudnn": 91000,
            "multiprocessors": 40, "driver": "550.54.15"}
FAKE_MACHINE = {"cpu_count": 2, "physical_cores": 1, "physical_cores_source": "test", "cpu_model": "Test CPU",
                "ram_gb": 12.7, "platform": "p", "python": "3", "versions": {"torch": "2.9", "laya": "0.3"}}


def _rows_file(tmp_path, n=12):
    rows = [{"id": f"test_id-{i:06d}", "label": "dining",
             "state": json.dumps({"country": "GB", "name": f"Rosa Pizza {i} " + "x " * (i % 5)},
                                 separators=(",", ":"))} for i in range(n)]
    path = tmp_path / "test_id.jsonl"
    write_jsonl(path, rows)
    return path


def _args(tmp_path, *extra):
    return ["--rows", str(_rows_file(tmp_path)), "--out", str(tmp_path / "bench_gpu.json"), *extra]


# ---------------------------------------------------------------- plan, device check, worker command

def test_defaults_are_the_colab_gpu_benchmark_and_models_come_from_config(cfg, tmp_path):
    plan = G.make_plan(G.parse_args(_args(tmp_path)), cfg)
    assert plan.models == tuple(cfg["phase5"]["bench"]["models"]) == tuple(ALL4)
    assert (plan.latency_n, plan.batch_n, plan.batch_sizes, plan.warmup) == (500, 2000, (32, 64), 20)
    assert plan.device == "cuda" and plan.ckpt == {m: "hub" for m in ALL4}
    assert plan.backends == {"laya": "torch", "laya_ml": "torch", "modernbert_base": "hf", "mmbert_small": "hf"}


def test_plan_takes_explicit_flags_and_rejects_bad_values(cfg, tmp_path):
    plan = G.make_plan(G.parse_args(_args(tmp_path, "--models", "mmbert_small", "laya", "mmbert_small",
                                          "--batch-sizes", "64", "16", "64", "--latency-n", "7", "--batch-n", "9",
                                          "--warmup", "0", "--device", "cpu")), cfg)
    assert plan.models == ("mmbert_small", "laya") and plan.batch_sizes == (64, 16)  # de-duplicated, order kept
    assert (plan.latency_n, plan.batch_n, plan.warmup, plan.device) == (7, 9, 0, "cpu")
    for bad in (["--batch-sizes", "0"], ["--latency-n", "0"], ["--batch-n", "-1"], ["--warmup", "-1"]):
        with pytest.raises(ValueError, match="invalid settings"):
            G.make_plan(G.parse_args(_args(tmp_path, *bad)), cfg)
    with pytest.raises(ValueError, match="unknown model 'bert_tiny'"):
        G.make_plan(G.parse_args(_args(tmp_path, "--models", "bert_tiny")), cfg)


def test_cuda_without_a_gpu_fails_fast_before_any_download_or_worker(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(G, "cuda_available", lambda: False)
    monkeypatch.setattr(G, "resolve_sources", lambda *a: pytest.fail("must not resolve (download) anything"))
    monkeypatch.setattr(G, "run_worker", lambda *a: pytest.fail("must not start a worker"))
    assert G.main(_args(tmp_path)) == 1
    err = capsys.readouterr().err
    assert "no CUDA device" in err and "GPU runtime" in err and "--device cpu" in err
    with pytest.raises(RuntimeError, match="no CUDA device"):
        G.check_device("cuda")
    G.check_device("cpu")  # never needs a GPU


def test_worker_env_keeps_the_gpu_for_cuda_hides_it_for_cpu_and_goes_offline():
    base = {"PATH": "x", "CUDA_VISIBLE_DEVICES": "0", "LAYA_CUDA_AMP": "fp16"}
    cuda, cpu = G.worker_env("cuda", base), G.worker_env("cpu", base)
    assert cuda["CUDA_VISIBLE_DEVICES"] == "0" and cuda["LAYA_CUDA_AMP"] == "fp16" and cuda["PATH"] == "x"
    assert cpu["CUDA_VISIBLE_DEVICES"] == ""
    for env in (cuda, cpu):
        assert env["HF_HUB_OFFLINE"] == "1" and env["PYTHONIOENCODING"] == "utf-8"
        assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert base["CUDA_VISIBLE_DEVICES"] == "0" and "HF_HUB_OFFLINE" not in base  # input untouched


def test_worker_cmd_parses_back_into_a_one_model_worker(cfg, tmp_path):
    args = G.parse_args(_args(tmp_path, "--models", "laya", "--batch-sizes", "8", "16", "--latency-n", "5",
                              "--batch-n", "6", "--warmup", "2", "--device", "cpu"))
    plan = G.make_plan(args, cfg)
    src = Source(model="laya", kind="laya", path=tmp_path / "ckpt", label="x")
    cmd = G.worker_cmd(args, plan, src)
    assert cmd[1:3] == ["-m", "laya_poc.bench_gpu"]
    w = G.parse_args(cmd[3:])
    assert w.worker and w.models == ["laya"] and w.ckpt == [str(tmp_path / "ckpt")]
    assert (w.latency_n, w.batch_n, w.batch_sizes, w.warmup, w.device) == (5, 6, [8, 16], 2, "cpu")


# ---------------------------------------------------------------- the parent with a fake runner

class FakeRunner:
    """Answers worker commands like the worker would: one MARKER line with its rows; records what ran."""

    def __init__(self, fail_model=None, oom_model=None, timeout_model=None):
        self.calls, self.fail, self.oom, self.timeout = [], fail_model, oom_model, timeout_model

    def __call__(self, cmd, env, timeout):
        model = cmd[cmd.index("--models") + 1]
        sizes = [int(b) for b in cmd[cmd.index("--batch-sizes") + 1:cmd.index("--warmup")]]
        self.calls.append((model, sizes, env["HF_HUB_OFFLINE"], timeout))
        if model == self.timeout:
            raise subprocess.TimeoutExpired(cmd, timeout)
        if model == self.fail:
            return SimpleNamespace(returncode=1, stdout="", stderr="loading...\nbench_gpu worker: error: boom\n")
        rows = [{"batch_size": bs, "device": "cuda", "gpu": "Tesla T4", "amp": True, "dtype": "float16",
                 "cold_s": 9.5, "first_predict_ms": 800.0, "p50_ms": 21.0, "p95_ms": 30.0, "mean_ms": 22.0,
                 "max_ms": 40.0, "batch_rps": 100.0 + bs, "batch_seconds": 20.0, "peak_vram_gb": 2.5,
                 "peak_vram_reserved_gb": 3.0, "pid": 1000 + len(self.calls)} for bs in sizes]
        errors = []
        if model == self.oom:
            errors = [{"batch_size": sizes[-1], "error": "OutOfMemoryError: CUDA out of memory"}]
            rows = rows[:-1]
        payload = {"results": rows, "errors": errors}
        return SimpleNamespace(returncode=0, stdout=f"laya warning\n{W.MARKER}{json.dumps(payload)}\n", stderr="")


def _run(tmp_path, monkeypatch, runner, *extra):
    def resolve(cfg, plan):
        return {m: Source(model=m, kind=model_kind(cfg, m), path=tmp_path / m, label=f"fake:{m}")
                for m in plan.models}

    monkeypatch.setattr(G, "cuda_available", lambda: True)
    monkeypatch.setattr(G, "run_worker", runner)
    monkeypatch.setattr(G, "resolve_sources", resolve)
    monkeypatch.setattr(G, "machine_info", lambda: dict(FAKE_MACHINE))
    monkeypatch.setattr(G, "gpu_facts", lambda: dict(FAKE_GPU))
    out = tmp_path / "bench_gpu.json"
    rc = G.main(_args(tmp_path, *extra))
    return rc, json.loads(out.read_text(encoding="utf-8"))


def test_one_fresh_process_per_model_and_one_row_per_model_and_batch_size(tmp_path, monkeypatch, capsys):
    runner = FakeRunner()
    rc, res = _run(tmp_path, monkeypatch, runner, "--timeout-min", "2")
    assert rc == 0 and res["errors"] == []
    assert [c[0] for c in runner.calls] == ALL4 and all(c[1] == [32, 64] for c in runner.calls)
    assert all(c[2] == "1" and c[3] == 120 for c in runner.calls)  # offline workers, per-model timeout
    assert [(r["model"], r["batch_size"]) for r in res["results"]] == [(m, b) for m in ALL4 for b in (32, 64)]
    backends = {r["model"]: r["backend"] for r in res["results"]}
    assert backends == {"laya": "torch", "laya_ml": "torch", "modernbert_base": "hf", "mmbert_small": "hf"}
    for r in res["results"]:
        assert {"model", "backend", "device", "gpu", "amp", "dtype", "batch_size", "cold_s", "first_predict_ms",
                "p50_ms", "p95_ms", "mean_ms", "max_ms", "batch_rps", "batch_seconds", "peak_vram_gb",
                "source"} <= set(r)
        assert r["source"] == f"fake:{r['model']}" and r["hardware"] == res["hardware"]
    plan = res["plan"]
    assert plan["models"] == ALL4 and plan["batch_sizes"] == [32, 64] and plan["latency_n"] == 500
    assert plan["batch_n"] == 2000 and plan["warmup"] == 20 and plan["device"] == "cuda"
    assert plan["rows"].endswith("test_id.jsonl") and plan["batch_warmup_batches"] == W.BATCH_WARMUP_BATCHES
    m = res["machine"]
    assert m["gpu"] == "Tesla T4" and m["capability"] == "7.5" and m["vram_total_gb"] == 14.56
    assert m["cuda"] == "12.8" and m["driver"] == "550.54.15" and m["cpu_count"] == 2 and m["versions"]["laya"]
    assert res["hardware"] == "Tesla T4 (cc 7.5, 14.56 GiB), CUDA 12.8, driver 550.54.15; host Test CPU (2 vCPUs)"
    assert res["schema_version"] == G.SCHEMA_VERSION and res["seconds"] >= 0
    assert res["sources"]["laya"] == {"kind": "laya", "path": str(tmp_path / "laya"), "label": "fake:laya"}
    out = capsys.readouterr().out
    assert "bench_gpu: laya/torch bs 32 on Tesla T4: cold 9.5 s, first 800 ms, p50 21.0 ms, p95 30.0 ms, batch " \
           "132.0 rec/s, peak VRAM 2.50 GiB (float16 autocast)" in out
    assert len(out.strip().splitlines()) <= 16


def test_a_failed_model_or_batch_size_is_an_error_and_the_rest_still_runs(tmp_path, monkeypatch, capsys):
    rc, res = _run(tmp_path, monkeypatch, FakeRunner(fail_model="laya_ml", oom_model="laya",
                                                     timeout_model="mmbert_small"))
    assert rc == 0
    assert [(r["model"], r["batch_size"]) for r in res["results"]] == [("laya", 32), ("modernbert_base", 32),
                                                                        ("modernbert_base", 64)]
    errs = {(e["model"], e["batch_size"]): e for e in res["errors"]}
    assert set(errs) == {("laya", 64), ("laya_ml", None), ("mmbert_small", None)}
    assert errs[("laya", 64)]["error"].startswith("OutOfMemoryError") and errs[("laya", 64)]["backend"] == "torch"
    assert errs[("laya_ml", None)] == {"model": "laya_ml", "backend": "torch", "batch_size": None, "returncode": 1,
                                       "error": "bench_gpu worker: error: boom"}
    assert "timed out" in errs[("mmbert_small", None)]["error"]
    out = capsys.readouterr().out
    assert "bench_gpu: laya/torch bs 64: FAILED: OutOfMemoryError" in out and "laya_ml/torch: FAILED" in out


def test_no_result_at_all_exits_1_but_keeps_the_json(tmp_path, monkeypatch):
    runner = lambda cmd, env, timeout: SimpleNamespace(returncode=0, stdout="no marker\n", stderr="")  # noqa: E731
    rc, res = _run(tmp_path, monkeypatch, runner, "--models", "laya")
    assert rc == 1 and res["results"] == [] and res["errors"][0]["error"].startswith("no result line")


# ---------------------------------------------------------------- real end-to-end on tiny models

@pytest.fixture(scope="module")
def tiny_encoder(cfg, tmp_path_factory):
    pytest.importorskip("transformers")
    from test_small_encoder import build_tiny_encoder, hub_config_dir

    try:
        src = hub_config_dir(cfg, "modernbert_base")
    except Exception as exc:  # offline, rate-limited, ...
        pytest.skip(f"Hub unavailable: {exc}")
    return build_tiny_encoder(src, tmp_path_factory.mktemp("gpu_tiny_encoder") / "enc")


def _tiny_run(tmp_path, tiny_ckpt_dir, tiny_encoder, device):
    out = tmp_path / "bench_gpu.json"
    rc = G.main(["--models", "laya", "modernbert_base", "--ckpt", f"laya={tiny_ckpt_dir}",
                 f"modernbert_base={tiny_encoder}", "--rows", str(_rows_file(tmp_path)), "--out", str(out),
                 "--device", device, "--latency-n", "3", "--batch-n", "6", "--batch-sizes", "2", "4",
                 "--warmup", "1"])
    res = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0 and res["errors"] == [], res["errors"]
    rows = {(r["model"], r["batch_size"]): r for r in res["results"]}
    assert set(rows) == {("laya", 2), ("laya", 4), ("modernbert_base", 2), ("modernbert_base", 4)}
    for m in ("laya", "modernbert_base"):
        a, b = rows[(m, 2)], rows[(m, 4)]
        assert a["pid"] == b["pid"] and a["p95_ms"] == b["p95_ms"] and a["cold_s"] == b["cold_s"] > 0
        assert a["n_latency"] == 3 and a["n_batch"] == 6 and a["batch_rps"] > 0 and a["device"] == device
    assert rows[("laya", 2)]["pid"] != rows[("modernbert_base", 2)]["pid"]  # a fresh process per model
    return res, rows


@pytest.mark.torch
def test_real_run_on_tiny_models_with_device_cpu(tiny_ckpt_dir, tiny_encoder, tmp_path):
    res, rows = _tiny_run(tmp_path, tiny_ckpt_dir, tiny_encoder, "cpu")
    assert rows[("laya", 2)]["amp"] is False and rows[("laya", 2)]["dtype"] == "float32"
    assert rows[("modernbert_base", 4)]["amp"] is False and rows[("modernbert_base", 4)]["max_length"] == 256
    assert rows[("laya", 2)]["peak_vram_gb"] is None and rows[("laya", 2)]["gpu"] is None
    assert res["plan"]["device"] == "cpu" and Path(res["sources"]["laya"]["path"]) == Path(tiny_ckpt_dir)


def _has_cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.torch
@pytest.mark.skipif(not _has_cuda(), reason="needs a CUDA GPU (runs on the Colab GPU host, not the laptop)")
def test_real_run_on_tiny_models_with_device_cuda(tiny_ckpt_dir, tiny_encoder, tmp_path):
    res, rows = _tiny_run(tmp_path, tiny_ckpt_dir, tiny_encoder, "cuda")
    assert rows[("laya", 2)]["amp"] is True and rows[("modernbert_base", 2)]["dtype"] == "float16"
    assert all(r["peak_vram_gb"] > 0 and r["gpu"] for r in rows.values())
    assert res["machine"]["gpu"] and res["machine"]["vram_total_gb"] > 0
