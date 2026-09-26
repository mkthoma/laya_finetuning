import json

import pytest

from laya_poc import env_check as E

EXPECTED = "4066d5d5fbf08b66c6757ddeedbd797bd7655bc0"

FIELDS = {"python", "platform", "torch", "torch_cuda", "cuda_available", "gpu_name", "capability", "arch_list",
          "has_sm75", "driver", "vram_total_gb", "transformers", "huggingface_hub", "duckdb", "laya_version",
          "laya_commit", "laya_commit_expected", "on_colab", "ram_gb", "disk_free_gb", "card"}


def _env(**over):
    base = {"cuda_available": True, "gpu_name": "Tesla T4", "capability": [7, 5], "arch_list": ["sm_75", "sm_80"],
            "has_sm75": True, "card": "T4", "laya_version": "0.3.20", "laya_commit": EXPECTED,
            "laya_commit_expected": EXPECTED}
    return {**base, **over}


def test_collect_env_has_every_field(cfg):
    env = E.collect_env(cfg)
    assert FIELDS <= set(env)
    assert env["laya_commit_expected"] == cfg["laya"]["commit"]
    assert isinstance(env["cuda_available"], bool) and isinstance(env["on_colab"], bool)
    assert env["disk_free_gb"] is None or env["disk_free_gb"] > 0
    json.dumps(env)  # JSON-serialisable


def test_collect_env_on_cpu_only_machine(cfg, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    env = E.collect_env(cfg)
    assert env["cuda_available"] is False
    assert env["gpu_name"] is None and env["card"] is None and env["vram_total_gb"] is None
    assert env["capability"] is None


@pytest.mark.parametrize("text,expected", [
    (json.dumps({"url": "https://github.com/x/laya.git", "vcs_info": {"vcs": "git", "commit_id": EXPECTED}}), EXPECTED),
    (json.dumps({"url": "file:///tmp/laya", "dir_info": {}}), None),
    ("not json", None),
    (None, None),
])
def test_commit_from_direct_url(text, expected):
    assert E.commit_from_direct_url(text) == expected


def test_driver_is_none_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(E.shutil, "which", lambda name: None)
    assert E.nvidia_driver() is None


def test_card_or_none():
    assert E.card_or_none("Tesla T4") == "T4"
    assert E.card_or_none("NVIDIA A100-SXM4-40GB") is None
    assert E.card_or_none(None) is None


def test_evaluate_env_passes_on_expected_t4():
    errors, warnings = E.evaluate_env(_env(), require_gpu=True, expect_card="T4")
    assert errors == [] and warnings == []


def test_evaluate_env_fails_without_gpu_when_required():
    env = _env(cuda_available=False, gpu_name=None, capability=None, card=None, has_sm75=False, arch_list=[])
    errors, _ = E.evaluate_env(env, require_gpu=True, expect_card=None)
    assert any("CUDA" in e for e in errors)
    assert E.evaluate_env(env, require_gpu=False, expect_card=None)[0] == []


def test_evaluate_env_warns_on_other_card():
    errors, warnings = E.evaluate_env(_env(gpu_name="NVIDIA L4", card="L4", capability=[8, 9]),
                                      require_gpu=True, expect_card="T4")
    assert errors == []
    assert any("L4" in w and "T4" in w for w in warnings)


def test_evaluate_env_fails_on_wrong_commit_only_when_known():
    errors, _ = E.evaluate_env(_env(laya_commit="deadbeef"), require_gpu=False, expect_card=None)
    assert any("deadbeef" in e for e in errors)
    errors, warnings = E.evaluate_env(_env(laya_commit=None), require_gpu=False, expect_card=None)
    assert errors == [] and any("commit" in w for w in warnings)


def test_evaluate_env_fails_without_laya():
    errors, _ = E.evaluate_env(_env(laya_version=None, laya_commit=None), require_gpu=False, expect_card=None)
    assert any("laya" in e for e in errors)


def test_evaluate_env_fails_when_torch_has_no_kernels_for_the_gpu():
    env = _env(arch_list=["sm_80", "sm_86"], has_sm75=False)
    errors, _ = E.evaluate_env(env, require_gpu=True, expect_card="T4")
    assert any("sm_75" in e for e in errors)  # critique §D.4: a T4 run needs sm_75 kernels
    errors, warnings = E.evaluate_env(env, require_gpu=False, expect_card="T4")
    assert errors == [] and any("sm_75" in w for w in warnings)


def test_evaluate_env_accepts_l4_on_a_wheel_without_sm89():
    # official cu126/cu128 wheels ship sm_86 SASS, which runs on sm_89
    env = _env(gpu_name="NVIDIA L4", card="L4", capability=[8, 9], arch_list=["sm_75", "sm_80", "sm_86", "sm_90"])
    assert E.evaluate_env(env, require_gpu=True, expect_card="L4") == ([], [])


def test_evaluate_env_checks_free_disk():
    # crash run: up to 3 x 5 GB resumable checkpoints, plus final 0.84 GB and ~3 GB of HF cache
    (err,), _ = E.evaluate_env(_env(disk_free_gb=12.5), require_gpu=True, expect_card="T4")
    assert "12.5" in err and "20" in err
    errors, warnings = E.evaluate_env(_env(disk_free_gb=12.5), require_gpu=False, expect_card=None)
    assert errors == [] and any("12.5" in w for w in warnings)  # local dry runs write tiny checkpoints
    errors, (warn,) = E.evaluate_env(_env(disk_free_gb=25.0), require_gpu=True, expect_card="T4")
    assert errors == [] and "25.0" in warn and "30" in warn
    assert E.evaluate_env(_env(disk_free_gb=80.0), require_gpu=True, expect_card="T4") == ([], [])
    assert E.evaluate_env(_env(disk_free_gb=None), require_gpu=True, expect_card="T4") == ([], [])


def test_evaluate_env_errors_carry_their_own_fix():
    no_gpu = _env(cuda_available=False, gpu_name=None, capability=None, card=None, has_sm75=False, arch_list=[])
    (err,), _ = E.evaluate_env(no_gpu, require_gpu=True, expect_card="T4")
    assert "New Colab Server > GPU > T4" in err and "not Auto Connect" in err
    (err,), _ = E.evaluate_env(_env(arch_list=["sm_80", "sm_86"], has_sm75=False), require_gpu=True,
                               expect_card="T4")
    assert "runtime version" in err and "2026.07" in err
    (err,), _ = E.evaluate_env(_env(laya_commit="deadbeef"), require_gpu=True, expect_card="T4")
    assert "deadbeef" in err and "re-run Step 2" in err
    (err,), _ = E.evaluate_env(_env(laya_version=None, laya_commit=None), require_gpu=True, expect_card="T4")
    assert "re-run Step 2" in err
    (err,), _ = E.evaluate_env(_env(disk_free_gb=5.0), require_gpu=True, expect_card="T4")
    assert "Step 13b" in err and "New Colab Server" in err


@pytest.mark.parametrize("capability,arch_list,ok", [
    ([7, 5], ["sm_75"], True),
    ([7, 5], ["sm_70"], True),            # SASS runs on the same major with a higher minor
    ([7, 5], ["sm_80", "sm_86"], False),  # never on a lower major
    ([8, 9], ["sm_86"], True),
    ([8, 6], ["sm_89"], False),           # nor on a lower minor
    ([8, 9], ["sm_90"], False),
    ([12, 0], ["sm_90", "compute_90"], True),  # PTX JIT-compiles for any newer GPU
    ([7, 5], ["compute_80"], False),
    ([9, 0], ["sm_90a"], True),           # arch-specific SASS: exact match only
    ([10, 3], ["sm_100a"], False),
    ([7, 5], [], False),
    ([7, 5], ["sm_x", "", "gfx90a"], False),
])
def test_arch_supported(capability, arch_list, ok):
    assert E.arch_supported(capability, arch_list) is ok


def test_main_writes_json_and_never_prints_token(tmp_path, monkeypatch, capsys):
    secret = "hf_" + "x" * 30
    monkeypatch.setenv("HF_TOKEN", secret)
    out = tmp_path / "sub" / "env.json"
    rc = E.main(["--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert FIELDS <= set(data) and data["passed"] is True
    printed = capsys.readouterr()
    assert secret not in printed.out + printed.err
    assert secret not in out.read_text(encoding="utf-8")


def test_main_require_gpu_fails_on_cpu(tmp_path, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = tmp_path / "env.json"
    rc = E.main(["--out", str(out), "--require-gpu", "--expect-card", "T4"])
    assert rc == 1
    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is False
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "CUDA" in err[0]


def test_main_prints_disk_and_one_line_per_error(tmp_path, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(E, "disk_free_gb", lambda path=None: 7.5)
    rc = E.main(["--out", str(tmp_path / "env.json"), "--require-gpu", "--expect-card", "T4"])
    assert rc == 1
    printed = capsys.readouterr()
    assert "disk free 7.5GiB" in printed.out
    err = printed.err.strip().splitlines()
    assert len(err) == 2 and all(line.startswith("env_check: FAIL: ") for line in err)
    assert any("CUDA" in line for line in err) and any("7.5" in line for line in err)


def test_run_cli_reports_one_line_and_keeps_traceback(tmp_path, capsys):
    def boom() -> int:
        raise RuntimeError("bad thing")

    detail = tmp_path / "x.json"
    rc = E.run_cli("demo", boom, detail)
    assert rc == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert err == ["demo: error: RuntimeError: bad thing (traceback: " + str(detail) + ".traceback.txt)"]
    assert "RuntimeError" in (tmp_path / "x.json.traceback.txt").read_text(encoding="utf-8")
