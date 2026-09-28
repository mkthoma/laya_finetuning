"""bench_cpu Phase 5 sweep form: plan defaults, thread cap, sources, backends, the parent with a fake runner, machine
facts, and one REAL end-to-end quick sweep on tiny models (tiny Laya: torch + onnx; tiny ModernBERT: hf)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from laya_poc import bench_cpu as B
from laya_poc import bench_machine as M
from laya_poc import bench_plan as P
from laya_poc import bench_sources as S
from laya_poc.io_utils import write_jsonl

ALL4 = ["laya", "laya_ml", "modernbert_base", "mmbert_small"]
MACHINE_4C = {"cpu_count": 8, "physical_cores": 4, "physical_cores_source": "test", "cpu_model": "Test CPU",
              "ram_gb": 16.0, "platform": "p", "python": "3", "versions": {}}


def _rows_file(tmp_path, n=40):
    rows = [{"id": f"test_id-{i:06d}", "label": "dining",
             "state": json.dumps({"country": "GB", "locality": "Leeds", "name": f"Rosa Pizza {i} " + "x " * (i % 9)},
                                 separators=(",", ":"))} for i in range(n)]
    path = tmp_path / "test_id.jsonl"
    write_jsonl(path, rows)
    return path


def _plan(cfg, *argv):
    return P.make_plan(B.parse_args(["--rows", "r.jsonl", "--out", "o.json", *argv]), cfg)


# ---------------------------------------------------------------- plan, backends, ckpt, thread cap

def test_sweep_plan_defaults_come_from_config_phase5_bench(cfg):
    plan = _plan(cfg, "--models", *ALL4)
    b = cfg["phase5"]["bench"]
    assert plan.form == "sweep" and plan.models == tuple(ALL4) and plan.thread_cap == "physical"
    assert plan.backends == {"laya": ["torch", "onnx"], "laya_ml": ["torch", "onnx"], "modernbert_base": ["hf"],
                             "mmbert_small": ["hf"]}  # config lists [torch] for the encoders: an alias of hf
    assert list(plan.threads) == b["threads"] and plan.latency_n == b["latency_n"] and plan.batch_n == b["batch_n"]
    assert plan.warmup == b["warmup"] and plan.batch_size == b["batch_size"]
    assert plan.check_n == b["onnx"]["check_n"] and plan.opset == b["onnx"]["opset"] == 18
    assert plan.ckpt == {m: "hub" for m in ALL4} and plan.needs_onnx() == ["laya", "laya_ml"]
    assert plan.onnx_dir.parts[-2:] == ("runs", "onnx")


def test_quick_shrinks_the_row_counts_and_explicit_flags_win(cfg):
    q = _plan(cfg, "--models", "laya", "--quick")
    assert (q.latency_n, q.batch_n, q.warmup, q.check_n) == (P.QUICK["latency_n"], P.QUICK["batch_n"],
                                                              P.QUICK["warmup"], P.QUICK["check_n"]) and q.quick
    e = _plan(cfg, "--models", "laya", "--quick", "--latency-n", "3", "--check-n", "0")
    assert e.latency_n == 3 and e.check_n == 0 and e.batch_n == P.QUICK["batch_n"]
    n = _plan(cfg, "--models", "laya", "--n", "7")
    assert n.latency_n == n.batch_n == 7


def test_backends_filter_per_model_and_skip_models_with_none(cfg):
    plan = _plan(cfg, "--models", *ALL4, "--backends", "onnx")
    assert plan.models == ("laya", "laya_ml") and plan.backends == {"laya": ["onnx"], "laya_ml": ["onnx"]}
    assert [s["model"] for s in plan.skipped_models] == ["modernbert_base", "mmbert_small"]
    torch_only = _plan(cfg, "--models", "laya", "mmbert_small", "--backends", "torch")
    assert torch_only.backends == {"laya": ["torch"], "mmbert_small": ["hf"]}
    with pytest.raises(ValueError, match="unknown model"):
        _plan(cfg, "--models", "laya", "bert_tiny")
    with pytest.raises(ValueError, match="no requested backend"):
        _plan(cfg, "--models", "modernbert_base", "--backends", "onnx")
    with pytest.raises(ValueError, match="positive"):
        _plan(cfg, "--models", "laya", "--threads", "0")


def test_ckpt_accepts_hub_one_path_or_model_path_pairs():
    assert S.parse_ckpt(None, ["laya", "laya_ml"]) == {"laya": "hub", "laya_ml": "hub"}
    assert S.parse_ckpt(["/abs/ck"], ["laya"]) == {"laya": "/abs/ck"}
    assert S.parse_ckpt(["laya_ml=/abs/ml"], ["laya", "laya_ml"]) == {"laya": "hub", "laya_ml": "/abs/ml"}
    with pytest.raises(ValueError, match="exactly one model"):
        S.parse_ckpt(["/abs/ck"], ["laya", "laya_ml"])
    with pytest.raises(ValueError, match="not among"):
        S.parse_ckpt(["mmbert_small=/abs/x"], ["laya"])
    with pytest.raises(ValueError, match="not a mix"):
        S.parse_ckpt(["hub", "laya=/abs/x"], ["laya"])


def test_local_sources_must_be_absolute_existing_model_dirs(cfg, tmp_path):
    laya_dir, hf_dir = tmp_path / "laya", tmp_path / "enc"
    laya_dir.mkdir()
    hf_dir.mkdir()
    (laya_dir / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    src = S.resolve_source(cfg, "laya_ml", str(laya_dir))
    assert (src.kind, src.path, src.label) == ("laya", laya_dir, "local:laya")  # no user path in the label
    assert S.resolve_source(cfg, "mmbert_small", str(hf_dir)).kind == "hf"
    with pytest.raises(ValueError, match="absolute"):
        S.resolve_source(cfg, "laya", "rel/dir")
    with pytest.raises(FileNotFoundError, match="rl_agent_config.json"):
        S.resolve_source(cfg, "laya", str(hf_dir))


def test_local_label_is_repo_relative_inside_the_project_else_the_folder_name(tmp_path):
    repo = tmp_path / "repo"
    assert S.local_label(repo / "runs" / "p3" / "best", root=repo) == "local:runs/p3/best"
    assert S.local_label(tmp_path / "elsewhere" / "best", root=repo) == "local:best"


def _fake_snapshot(root, repo_files, calls):
    """A stand-in for huggingface_hub.snapshot_download: 'downloads' the repo files matching allow_patterns."""
    from fnmatch import fnmatch

    def snapshot_download(repo_id, revision=None, allow_patterns=None, token=None):
        calls.append(list(allow_patterns))
        root.mkdir(exist_ok=True)
        for name in repo_files:
            if any(fnmatch(name, p) for p in allow_patterns):
                (root / name).write_text("x", encoding="utf-8")
        return str(root)
    return snapshot_download


@pytest.mark.parametrize("weights, fallback", [("model.safetensors", False), ("pytorch_model.bin", True)])
def test_hub_encoder_falls_back_to_bin_weights_only_without_safetensors(cfg, tmp_path, monkeypatch, weights,
                                                                        fallback):
    import huggingface_hub

    calls = []
    repo = ["config.json", "tokenizer.json", weights, "onnx/model.onnx"]
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot(tmp_path / "snap", repo, calls))
    src = S.resolve_source(cfg, "mmbert_small", "hub")
    assert (src.path / weights).exists() and not (src.path / "onnx").exists()
    assert len(calls) == (2 if fallback else 1) and "*.bin" not in calls[0]  # mmBERT-small ships only a .bin


def test_thread_cap_skips_settings_above_the_physical_cores():
    run, skipped, cap = P.cap_threads([1, 2, 4, 8], "physical", MACHINE_4C)
    assert run == [1, 2, 4] and [s["threads"] for s in skipped] == [8] and "4 physical cores" in skipped[0]["reason"]
    assert cap == {"basis": "physical", "limit": 4, "note": None}
    assert P.cap_threads([1, 2, 4, 8], "logical", MACHINE_4C)[0] == [1, 2, 4, 8]
    assert P.cap_threads([1, 16], "none", MACHINE_4C)[0] == [1, 16]
    run, _, cap = P.cap_threads([1, 8, 16], "physical", {**MACHINE_4C, "physical_cores": None})
    assert run == [1, 8] and "unknown" in cap["note"]


# ---------------------------------------------------------------- machine facts

def test_linux_cpuinfo_counts_distinct_physical_cores():
    block = "processor\t: {p}\nphysical id\t: {s}\ncore id\t\t: {c}\nmodel name\t: X\n\n"
    text = "".join(block.format(p=i, s=i // 4, c=(i % 4) // 2) for i in range(8))  # 2 sockets x 2 cores x 2 HT
    assert M._linux_physical_cores(text) == 4
    assert M._linux_physical_cores("processor\t: 0\n") is None


def test_machine_info_has_every_hardware_field():
    m = M.machine_info()
    assert m["cpu_count"] >= 1 and m["physical_cores"] and m["physical_cores"] <= m["cpu_count"]
    assert m["cpu_model"] and m["platform"] and m["versions"]["torch"]
    label = M.hardware_label(m)
    assert f"{m['physical_cores']}C/{m['cpu_count']}T" in label and m["cpu_model"] in label


# ---------------------------------------------------------------- the sweep parent with a fake runner

class FakeRunner:
    """Answers onnx_export (writes <out>.json like the CLI) and worker commands; records what ran."""

    def __init__(self, accepted=True, export_fails=()):
        self.calls, self.accepted, self.export_fails = [], accepted, set(export_fails)

    def __call__(self, cmd, env, timeout):
        module = cmd[2]
        if module == "laya_poc.onnx_export":
            return self.export(cmd)
        model, backend = cmd[cmd.index("--model") + 1], cmd[cmd.index("--worker-backend") + 1]
        threads = int(cmd[cmd.index("--worker-threads") + 1])
        self.calls.append(("worker", model, backend, threads, env["OMP_NUM_THREADS"]))
        row = {"model": model, "backend": backend, "threads": threads, "cold_s": 2.0, "p50_ms": 300.0,
               "p95_ms": 450.0, "mean_ms": 320.0, "batch_rps": 9.0, "peak_rss_gb": 2.0, "n_latency": 10, "n_batch": 32}
        return SimpleNamespace(returncode=0, stdout=f"{B.MARKER}{json.dumps(row)}\n", stderr="")

    def export(self, cmd):
        model, out = cmd[cmd.index("--model") + 1], Path(cmd[cmd.index("--out") + 1])
        self.calls.append(("export", model, "--reuse" in cmd, cmd[cmd.index("--check-n") + 1]))
        if model in self.export_fails:
            return SimpleNamespace(returncode=1, stdout="", stderr="onnx_export: error: RuntimeError: no graph\n")
        out.parent.mkdir(parents=True, exist_ok=True)
        acc = {"n": 32, "argmax_agree": 1.0 if self.accepted else 0.9, "max_dp": 1e-5, "passed": self.accepted}
        meta = {"ckpt": f"fake:{model}", "export": {"exporter": "dynamo", "dummy_batch": 2}, "acceptance": acc}
        Path(f"{out}.json").write_text(json.dumps(meta), encoding="utf-8")
        return SimpleNamespace(returncode=0 if self.accepted else 1, stdout="onnx_export: acceptance ...\n", stderr="")


def _sweep(tmp_path, monkeypatch, runner, *extra):
    def resolve(cfg, plan):
        return {m: S.Source(model=m, kind=S.model_kind(cfg, m), path=tmp_path / m, label=f"fake:{m}")
                for m in plan.models}

    monkeypatch.setattr(B, "run_worker", runner)
    monkeypatch.setattr(B, "resolve_sources", resolve)
    monkeypatch.setattr(B, "machine_info", lambda: dict(MACHINE_4C))
    out = tmp_path / "bench.json"
    rc = B.main(["--models", *ALL4, "--rows", str(_rows_file(tmp_path)), "--out", str(out), "--quick",
                 "--onnx-dir", str(tmp_path / "onnx"), *extra])
    return rc, json.loads(out.read_text(encoding="utf-8"))


def test_sweep_runs_one_fresh_process_per_model_backend_and_thread(tmp_path, monkeypatch, capsys):
    runner = FakeRunner()
    rc, res = _sweep(tmp_path, monkeypatch, runner, "--threads", "1", "4", "8")
    assert rc == 0 and res["form"] == "sweep" and res["errors"] == []
    exports = [c for c in runner.calls if c[0] == "export"]
    assert exports == [("export", "laya", True, "32"), ("export", "laya_ml", True, "32")]  # before any worker
    assert runner.calls[:2] == exports
    workers = [c[1:4] for c in runner.calls if c[0] == "worker"]
    assert workers == [(m, b, t) for m, bs in (("laya", ("torch", "onnx")), ("laya_ml", ("torch", "onnx")),
                                              ("modernbert_base", ("hf",)), ("mmbert_small", ("hf",)))
                       for b in bs for t in (1, 4)]
    assert all(c[4] == str(c[3]) for c in runner.calls if c[0] == "worker")  # thread pools pinned per setting
    assert res["threads_run"] == [1, 4] and res["threads_skipped"][0]["threads"] == 8
    onnx_rows = [r for r in res["results"] if r["backend"] == "onnx"]
    assert len(onnx_rows) == 4 and all(r["onnx_accepted"] is True for r in onnx_rows)
    assert all("onnx_accepted" not in r for r in res["results"] if r["backend"] != "onnx")
    assert res["onnx"]["laya"]["accepted"] is True and res["onnx"]["laya"]["export"]["dummy_batch"] == 2
    assert res["hardware"] == "Test CPU (4C/8T, 16 GB RAM)" and res["machine"]["physical_cores"] == 4
    for r in res["results"]:
        assert {"model", "backend", "threads", "cold_s", "p50_ms", "p95_ms", "mean_ms", "batch_rps", "peak_rss_gb",
                "n_latency", "n_batch", "p95_ok", "rps_ok", "source", "hardware"} <= set(r)
    assert res["quick"] is True and res["latency_n"] == P.QUICK["latency_n"] and "model" not in res
    assert len(capsys.readouterr().out.strip().splitlines()) <= 20


def test_a_failed_onnx_export_is_an_error_and_the_other_backends_still_run(tmp_path, monkeypatch):
    stale = tmp_path / "onnx" / "laya.onnx.json"  # an earlier run's passing record must not be mistaken for this one
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"acceptance": {"passed": True}, "export": {}}), encoding="utf-8")
    rc, res = _sweep(tmp_path, monkeypatch, FakeRunner(export_fails={"laya"}), "--threads", "1", "--reexport")
    assert rc == 0
    assert [e["error"] for e in res["errors"]] == ["ONNX export failed: onnx_export: error: RuntimeError: no graph"]
    assert res["onnx"]["laya"]["accepted"] is None and "error" in res["onnx"]["laya"]
    backends = {(r["model"], r["backend"]) for r in res["results"]}
    assert ("laya", "torch") in backends and ("laya", "onnx") not in backends and ("laya_ml", "onnx") in backends


def test_a_failed_acceptance_still_benchmarks_but_marks_the_onnx_rows(tmp_path, monkeypatch):
    rc, res = _sweep(tmp_path, monkeypatch, FakeRunner(accepted=False), "--threads", "1")
    assert rc == 0 and res["onnx"]["laya"]["accepted"] is False
    assert {r["onnx_accepted"] for r in res["results"] if r["backend"] == "onnx"} == {False}


# ---------------------------------------------------------------- real end-to-end quick sweep on tiny models

@pytest.fixture(scope="module")
def tiny_encoder(cfg, tmp_path_factory):
    pytest.importorskip("transformers")
    from test_small_encoder import build_tiny_encoder, hub_config_dir

    try:
        src = hub_config_dir(cfg, "modernbert_base")
    except Exception as exc:  # offline, rate-limited, ...
        pytest.skip(f"Hub unavailable: {exc}")
    return build_tiny_encoder(src, tmp_path_factory.mktemp("bench_tiny_encoder") / "enc")


@pytest.fixture(scope="module")
def distinct_ckpt(en_snapshot, tmp_path_factory):
    """A tiny Laya checkpoint whose option logits differ (the shared tiny_ckpt_dir's are near-ties at init 0.05, so
    an argmax-agreement check on it would be a coin flip)."""
    from conftest import build_tiny_checkpoint

    return build_tiny_checkpoint(en_snapshot, tmp_path_factory.mktemp("bench_distinct") / "ckpt", init_scale=0.5)


@pytest.mark.torch
def test_real_quick_sweep_on_tiny_models(distinct_ckpt, tiny_encoder, tmp_path):
    pytest.importorskip("onnxruntime")
    out = tmp_path / "bench.json"
    rc = B.main(["--models", "laya", "modernbert_base", "--ckpt", f"laya={distinct_ckpt}",
                 f"modernbert_base={tiny_encoder}", "--rows", str(_rows_file(tmp_path)), "--out", str(out),
                 "--quick", "--threads", "1", "--onnx-dir", str(tmp_path / "onnx"), "--latency-n", "4",
                 "--batch-n", "8", "--check-n", "12", "--batch-size", "4"])
    res = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0 and res["errors"] == [], res["errors"]
    rows = {(r["model"], r["backend"]): r for r in res["results"]}
    assert set(rows) == {("laya", "torch"), ("laya", "onnx"), ("modernbert_base", "hf")}
    assert len({r["pid"] for r in res["results"]}) == 3  # a fresh process per setting
    assert rows[("laya", "onnx")]["framework"] == "onnxruntime" and rows[("laya", "onnx")]["ort_intra_op_threads"] == 1
    assert rows[("laya", "onnx")]["onnx_accepted"] is True
    hf = rows[("modernbert_base", "hf")]
    assert hf["max_length"] == 256 and hf["dtype"] == "float32" and hf["framework"] == "pytorch"
    for r in rows.values():
        assert r["n_latency"] == 4 and r["n_batch"] == 8 and r["p95_ms"] > 0 and r["batch_rps"] > 0
    acc = res["onnx"]["laya"]["acceptance"]
    assert acc["passed"] and acc["n"] == 12 and acc["max_dp"] <= 1e-3 and acc["argmax_agree"] >= 0.999
    assert (tmp_path / "onnx" / "laya.onnx").is_file()
