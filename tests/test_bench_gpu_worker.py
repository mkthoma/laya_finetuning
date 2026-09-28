"""bench_gpu_worker: the timing core it shares with the CPU benchmark (bench_worker.time_latency / time_batch), the
sync wrapper, the measurement on fake runtimes (shared batch-1 latency, per-batch-size warm-up, throughput and peak
VRAM on a faked torch.cuda, OOM and CPU-fallback rows), the GPU facts, the Laya and encoder runtimes with fake models
(loaded as evaluation / Phase 4 load them), and the worker entry's MARKER line. The parent: test_bench_gpu.py."""
import json
from types import SimpleNamespace

import pytest

from laya_poc import bench_gpu as G
from laya_poc import bench_gpu_worker as W
from laya_poc import bench_worker as BW
from laya_poc import labels as L
from test_bench_gpu import GIB, _args

# ---------------------------------------------------------------- timing core (bench_worker) and sync

def test_time_latency_and_time_batch_split_time_runtime_without_changing_it():
    calls = []
    rt = BW.Runtime(predict_one=lambda s: calls.append(("one", s)), predict_many=lambda ss: calls.append(ss))
    got, lat = BW.time_latency(lambda: rt, ["a", "b", "c"], warmup=2)
    assert got is rt and lat["n_latency"] == 3 and lat["warmup"] == 2 and lat["cold_s"] >= 0
    assert [c[1] for c in calls] == ["a", "a", "b", "a", "b", "c"]  # first call, warm-up (cycled), latency
    assert {"p50_ms", "p95_ms", "mean_ms", "max_ms", "first_predict_ms"} <= set(lat)
    b = BW.time_batch(rt.predict_many, ("x", "y"))
    assert calls[-1] == ["x", "y"] and b["n_batch"] == 2 and b["batch_rps"] > 0 and b["batch_seconds"] >= 0
    with pytest.raises(TypeError, match="compact JSON strings"):
        BW.time_batch(rt.predict_many, [{"a": 1}])
    full = BW.time_runtime(lambda: rt, ["a"], ["x"], 0)
    assert list(full)[:6] == ["n", "n_latency", "n_batch", "warmup", "cold_s", "first_predict_ms"]


def test_synced_calls_the_sync_around_the_timed_call():
    events = []
    fn = W.synced(lambda x, y=0: events.append(("call", x, y)) or "out", lambda: events.append("sync"))
    assert fn(1, 2) == "out" and events == ["sync", ("call", 1, 2), "sync"]
    assert W.cuda_sync("cpu")() is None  # no-op off CUDA (never imports a CUDA context)


# ---------------------------------------------------------------- the worker's measurement on fake runtimes

STATES = [json.dumps({"name": f"n{i}"}, separators=(",", ":")) for i in range(10)]


class FakeRuntime:
    def __init__(self, oom_at=None, fallback_at=None, fallback_one=False):
        self.calls, self.oom_at, self.fallback_at, self.fallback_one = [], oom_at, fallback_at, fallback_one

    def load(self):
        return W.GpuRuntime(predict_one=self.one, predict_batch=self.many, info={"framework": "fake", "amp": False})

    def one(self, s):
        self.calls.append(("one", s))
        if self.fallback_one:
            print("Warning: GPU memory exceeded during inference. Retrying this request on CPU...")

    def many(self, ss, bs):
        self.calls.append(("many", len(ss), bs))
        if bs == self.oom_at:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        if bs == self.fallback_at:
            print("Warning: GPU memory exceeded during inference. Retrying this request on CPU...")


def test_measure_model_shares_batch1_latency_and_times_each_batch_size_after_its_warmup():
    fake = FakeRuntime(oom_at=8)
    out = W.measure_model(fake.load, STATES[:4], STATES, batch_sizes=[2, 8, 3], warmup=1, device="cpu")
    many = [c for c in fake.calls if c[0] == "many"]
    k = W.BATCH_WARMUP_BATCHES
    assert many == [("many", min(10, 2 * k), 2), ("many", 10, 2), ("many", min(10, 8 * k), 8),
                    ("many", min(10, 3 * k), 3), ("many", 10, 3)]  # warm-up slice, then the timed call
    assert len([c for c in fake.calls if c[0] == "one"]) == 1 + 1 + 4  # first call, warm-up, latency
    assert [r["batch_size"] for r in out["results"]] == [2, 3]
    r2, r3 = out["results"]
    for key in ("cold_s", "first_predict_ms", "p50_ms", "p95_ms", "mean_ms", "max_ms", "n_latency"):
        assert r2[key] == r3[key]  # batch-1 latency is measured once per model and repeated on its rows
    assert r2["n_batch"] == 10 and r2["batch_warmup_rows"] == min(10, 2 * k) and r2["batch_rps"] > 0
    assert r2["peak_vram_gb"] is None and r2["b1_peak_vram_gb"] is None and r2["cpu_fallback"] is False
    assert r2["framework"] == "fake" and r2["peak_rss_gb"] > 0
    (err,) = out["errors"]
    assert err["batch_size"] == 8 and err["error"].startswith("RuntimeError: CUDA out of memory")


def test_a_cpu_fallback_makes_that_batch_size_an_error_and_in_the_latency_phase_the_whole_model():
    out = W.measure_model(FakeRuntime(fallback_at=4).load, STATES[:3], STATES, batch_sizes=[2, 4], warmup=0,
                          device="cpu")
    assert [r["batch_size"] for r in out["results"]] == [2]
    assert "fell back to the CPU at batch size 4" in out["errors"][0]["error"]
    with pytest.raises(RuntimeError, match="fell back to the CPU during the batch-1 latency"):
        W.measure_model(FakeRuntime(fallback_one=True).load, STATES[:3], STATES, batch_sizes=[2], warmup=0,
                        device="cpu")


@pytest.fixture
def fake_cuda(monkeypatch):
    """torch.cuda's memory API faked on a CPU-only torch: records resets, reports growing peaks."""
    torch = pytest.importorskip("torch")
    log = {"reset": 0, "empty": 0, "sync": 0, "peak": 0}

    def reset():
        log["reset"] += 1

    def empty():
        log["empty"] += 1

    def peak():
        log["peak"] += 1
        return (1 + log["reset"]) * GIB

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: log.__setitem__("sync", log["sync"] + 1))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", peak)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2 * (1 + log["reset"]) * GIB)
    monkeypatch.setattr(W, "gpu_facts", lambda: {"gpu": "Tesla T4"})
    return log


def test_on_cuda_peak_vram_is_reset_before_each_batch_size_and_read_after_it(fake_cuda):
    out = W.measure_model(FakeRuntime().load, STATES[:2], STATES, batch_sizes=[2, 4], warmup=0, device="cuda")
    r2, r4 = out["results"]
    assert fake_cuda["reset"] == 2 and fake_cuda["empty"] == 2  # never before the load (cold start stays honest)
    assert r2["b1_peak_vram_gb"] == 1.0 and r2["b1_peak_vram_reserved_gb"] == 2.0  # a fresh process: from zero
    assert (r2["peak_vram_gb"], r2["peak_vram_reserved_gb"]) == (2.0, 4.0)
    assert (r4["peak_vram_gb"], r4["peak_vram_reserved_gb"]) == (3.0, 6.0)
    assert r2["gpu"] == r4["gpu"] == "Tesla T4" and r2["device"] == "cuda"
    assert W.peak_vram("cpu") == {"peak_vram_gb": None, "peak_vram_reserved_gb": None}


def test_gpu_facts_record_name_capability_vram_cuda_and_driver(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "NVIDIA RTX PRO 6000 Blackwell Server Edition")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i=0: SimpleNamespace(total_memory=95 * GIB, multi_processor_count=188))
    monkeypatch.setattr(W, "nvidia_driver", lambda: "580.65")
    f = W.gpu_facts()
    assert f["gpu"].startswith("NVIDIA RTX PRO 6000") and f["capability"] == "12.0" and f["vram_total_gb"] == 95.0
    assert f["multiprocessors"] == 188 and f["driver"] == "580.65" and "cuda" in f
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert W.gpu_facts()["gpu"] is None


# ---------------------------------------------------------------- runtimes with fake models

class FakeAgent:
    def __init__(self):
        self.calls = []
        self.device, self.amp_enabled, self.dtype = SimpleNamespace(type="cuda"), True, "torch.float16"
        self.model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype="torch.float32")]))

    def predict(self, s, q, **kw):
        self.calls.append(("predict", s, kw))

    def predict_batch(self, ss, q, **kw):
        self.calls.append(("predict_batch", list(ss), kw))


def test_laya_runtime_loads_like_evaluate_and_batches_length_sorted(monkeypatch, tmp_path):
    import laya_poc.evaluate as E

    agent, loaded, syncs = FakeAgent(), [], []
    monkeypatch.setattr(E, "load_checked", lambda src, device: loaded.append((src, device)) or agent)
    monkeypatch.setattr(W, "cuda_sync", lambda device: lambda: syncs.append(device))
    monkeypatch.setenv("LAYA_CUDA_AMP", "fp16")
    rt = W.laya_gpu_loader(tmp_path, L.question("c10"), device="cuda", max_len=512, head_max_len=192)()
    rt.predict_one("s")
    rt.predict_batch(["a", "b"], 16)
    assert loaded == [(tmp_path, "cuda")] and syncs == ["cuda"] * 4
    budget = {"max_len": 512, "head_max_len": 192}
    assert agent.calls == [("predict", "s", budget),
                           ("predict_batch", ["a", "b"], {"batch_size": 16, "sort_by_length": True, **budget})]
    assert rt.info["backend"] == "torch" and rt.info["amp"] is True and rt.info["dtype"] == "float16"
    assert rt.info["param_dtype"] == "float32" and rt.info["laya_cuda_amp"] == "fp16" and rt.info["device"] == "cuda"


def test_encoder_runtime_scores_like_phase_4_on_cuda(cfg, monkeypatch, tmp_path):
    import laya_poc.bench_hf as H
    import laya_poc.small_encoder_data as SD
    import laya_poc.small_encoder_train  # noqa: F401  imported BEFORE the patch: hf_gpu_loader imports it, and a
    # first import under the patch would bind the fake score_states into it for every later test

    moved, scored = [], []
    param = SimpleNamespace(dtype="torch.float32", device=SimpleNamespace(type="cuda"))
    model = SimpleNamespace(to=lambda d: moved.append(d), parameters=lambda: iter([param]),
                            config=SimpleNamespace(num_labels=10, _attn_implementation="sdpa"))
    monkeypatch.setattr(H, "load_model", lambda c, key, path: (model, "tok"))
    monkeypatch.setattr(SD, "score_states", lambda m, t, ss, **kw: scored.append((list(ss), kw)))
    monkeypatch.setattr(W, "cuda_sync", lambda device: lambda: None)
    rt = W.hf_gpu_loader(cfg, "modernbert_base", tmp_path, device="cuda")()
    rt.predict_one("s")
    rt.predict_batch(["a", "b"], 32)
    st = cfg["phase4"]["small_encoder_train"]
    kw = {"max_length": st["max_length"], "device": "cuda", "fp16": bool(st.get("fp16", True))}
    assert moved == ["cuda"] and scored == [(["s"], {"batch_size": 1, **kw}), (["a", "b"], {"batch_size": 32, **kw})]
    assert rt.info["backend"] == "hf" and rt.info["amp"] is True and rt.info["dtype"] == "float16"
    assert rt.info["param_dtype"] == "float32" and rt.info["max_length"] == 256
    cpu = W.hf_gpu_loader(cfg, "modernbert_base", tmp_path, device="cpu")
    param.device.type = "cpu"
    assert cpu().info["amp"] is False and cpu().info["dtype"] == "float32"


def test_worker_main_prints_one_marker_line_with_its_rows(cfg, monkeypatch, tmp_path, capsys):
    ckpt = tmp_path / "laya_ckpt"
    ckpt.mkdir()
    (ckpt / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    fake, seen = FakeRuntime(), {}

    def loader(path, question, **kw):
        seen.update(path=path, question=question, **kw)
        return fake.load

    monkeypatch.setattr(W, "laya_gpu_loader", loader)
    rc = G.main(_args(tmp_path, "--worker", "--models", "laya", "--ckpt", str(ckpt), "--device", "cpu",
                      "--latency-n", "3", "--batch-n", "5", "--batch-sizes", "2", "--warmup", "1"))
    assert rc == 0
    assert seen["path"] == ckpt and seen["device"] == "cpu" and seen["max_len"] == cfg["model"]["laya"]["max_len"]
    assert seen["question"] == L.question(cfg["labels"]["scheme"])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith(W.MARKER)]
    (payload,) = [json.loads(ln[len(W.MARKER):]) for ln in lines]
    (row,) = payload["results"]
    assert row["batch_size"] == 2 and row["n_latency"] == 3 and row["n_batch"] == 5 and row["pid"] > 0
    assert payload["errors"] == []
