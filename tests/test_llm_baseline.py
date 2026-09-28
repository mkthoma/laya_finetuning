"""B5 reference LLM (llm_baseline.py): subsets, prompt, packed key scoring vs brute force, CLI outputs.

The model tests use a tiny RANDOM Qwen3 causal LM built from the real Qwen3-4B config.json (tiny dims, a large
init range so key scores differ by nats, not micro-nats) plus the real tokenizer, both fetched at the pinned
revision (config + tokenizer files only); they skip when the Hub is unreachable.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from laya_poc import labels as L
from laya_poc import llm_baseline as lb
from laya_poc.io_utils import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
TINY = dict(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            num_hidden_layers=2, max_window_layers=2, initializer_range=0.5)
HUB_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
EVAL_KEYS = {"model", "ckpt", "split", "n", "n_unlabelled", "n_abstained_no_evidence", "temperature", "pre", "post",
             "abstention"}
STRIPPED_KEYS = {"abstain_rate_at_tau", "false_confident_rate", "false_confident_threshold", "gate_abstain_rate",
                 "no_gate"}


def _rows(n: int, split: str = "test_id", labelled_every: int = 1) -> list[dict]:
    return [{"id": f"{split}-{i:06d}", "state": "{}", "label": "dining" if i % labelled_every == 0 else None}
            for i in range(n)]


# ---------------------------------------------------------------- pure

@pytest.mark.parametrize("scheme", ["c10", "c7"])
def test_instruction_lists_every_key_with_its_description_and_the_state(scheme):
    state = '{"country":"GB","name":"Rosa\'s Trattoria"}'
    text = lb.instruction(scheme, state)
    assert text.startswith(L.INSTRUCTIONS) and state in text
    assert all(f"{k}: {d}" in text for k, d in L.criteria(scheme).items())
    assert lb.render(None, text, "plain").endswith(f"\n{lb.PLAIN_CUE}")


def test_val_subset_is_the_first_n_labelled_rows_in_file_order():
    rows = _rows(20, "val", labelled_every=3)
    keep, note = lb.select_rows("val", rows, {"val_for_temperature": 4})
    assert [r["id"] for r in keep] == ["val-000000", "val-000003", "val-000006", "val-000009"]
    assert note["subset"] is True and note["n_source"] == 20 and "first 4 labelled" in note["subset_rule"]


@pytest.mark.parametrize("split", ["stripped_test", "trap_candidates"])
def test_stripped_and_trap_splits_are_scored_in_full(split):
    rows = _rows(30, split)
    keep, note = lb.select_rows(split, rows, {"subset_per_pool": 5, "subset_seed": 1})
    assert keep == rows and note["n_source"] == 30


def test_pool_sample_is_deterministic_seeded_by_id_and_keeps_file_order():
    cfg = {"subset_per_pool": 10, "subset_seed": 20260925}
    rows = _rows(50)
    a, note = lb.select_rows("test_id", rows, cfg)
    b, _ = lb.select_rows("test_id", rows, cfg)
    ids = [r["id"] for r in a]
    assert a == b and len(a) == 10 and ids == sorted(ids) and note["n_source"] == 50
    shuffled = random.Random(0).sample(rows, len(rows))
    assert sorted(r["id"] for r in lb.select_rows("test_id", shuffled, cfg)[0]) == ids   # ids, not positions
    assert [r["id"] for r in lb.select_rows("test_id", rows, {**cfg, "subset_seed": 1})[0]] != ids
    assert {r["id"][-6:] for r in lb.select_rows("ood_country", _rows(50, "ood_country"), cfg)[0]} != \
        {i[-6:] for i in ids}                                                                  # split-specific
    assert lb.select_rows("ood_brand", rows[:7], cfg)[0] == rows[:7]                           # all when fewer


def test_pair_index_maps_every_key_token_to_the_row_that_predicts_it():
    src, tgt, key_of = lb.pair_index([[1], [2, 3], [4, 5, 6]])
    # packed continuation = [2, 4, 5] at rows 1..3; row 0 = the last prompt token
    assert (src, tgt, key_of) == ([0, 0, 1, 0, 2, 3], [1, 2, 3, 4, 5, 6], [0, 1, 1, 2, 2, 2])


# ---------------------------------------------------------------- torch (no Hub)

@pytest.mark.torch
def test_packed_batch_and_mask_isolate_each_key():
    torch = pytest.importorskip("torch")
    b = lb.packed_batch([[10, 11, 12], [20, 21]], [[1], [2, 3], [4, 5, 6]], pad_id=0)
    assert b["input_ids"].tolist() == [[10, 11, 12, 2, 4, 5], [20, 21, 2, 4, 5, 0]]
    assert b["position_ids"].tolist() == [[0, 1, 2, 3, 3, 4], [0, 1, 2, 2, 3, 0]]
    allowed = lb.packed_allowed(b["seg"])
    a0, a1 = allowed[0].int().tolist(), allowed[1].int().tolist()
    assert a0[2] == [1, 1, 1, 0, 0, 0]                      # prompt: causal, never a key token
    assert a0[3] == [1, 1, 1, 1, 0, 0]                      # key 1 token: prompt + itself
    assert a0[4] == [1, 1, 1, 0, 1, 0] and a0[5] == [1, 1, 1, 0, 1, 1]   # key 2: prompt + its own tokens only
    assert a1[5] == [0, 0, 0, 0, 0, 1]                      # pad: itself only (no NaN rows)
    mask = lb.additive_mask(allowed, torch.float16)
    assert mask.shape == (2, 1, 6, 6) and mask[0, 0, 2, 3].item() == torch.finfo(torch.float16).min
    assert mask[0, 0, 5, 4].item() == 0


@pytest.mark.torch
def test_pick_dtype_bf16_from_ampere_fp16_below_fp32_on_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    assert lb.pick_dtype("cpu") == torch.float32
    for cc, want in [((7, 5), torch.float16), ((8, 0), torch.bfloat16), ((12, 0), torch.bfloat16)]:
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, _cc=cc: _cc)
        assert lb.pick_dtype("cuda") == want


# ---------------------------------------------------------------- tiny Qwen3 from the real config + tokenizer

@pytest.fixture(scope="session")
def qwen_snapshot(cfg):
    pytest.importorskip("torch")
    spec = cfg["phase4"]["llms"]["qwen3_4b"]
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(spec["id"], revision=spec["revision"], allow_patterns=HUB_FILES))
    except Exception as exc:  # offline, rate-limited, ...
        pytest.skip(f"Hub unavailable: {exc}")


def build_tiny_llm(src: Path, dst: Path, seed: int = 0) -> Path:
    """A random tiny causal LM with the real config's architecture (+ TINY dims) and the real tokenizer, saved to
    dst: a local dir `llm_baseline --init` accepts. src: a snapshot with config.json and the tokenizer files."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    real = json.loads((Path(src) / "config.json").read_text(encoding="utf-8"))
    kept = {k: v for k, v in real.items() if k not in ("architectures", "transformers_version", "torch_dtype")}
    torch.manual_seed(seed)
    AutoModelForCausalLM.from_config(AutoConfig.for_model(**{**kept, **TINY})).save_pretrained(dst)
    AutoTokenizer.from_pretrained(src).save_pretrained(dst)
    return Path(dst)


@pytest.fixture(scope="session")
def tiny_llm_dir(qwen_snapshot, tmp_path_factory):
    return build_tiny_llm(qwen_snapshot, tmp_path_factory.mktemp("tiny_qwen3"))


@pytest.fixture(scope="session")
def tiny_llm(tiny_llm_dir):
    return lb.load_llm(str(tiny_llm_dir), None, "cpu")


def _prompts(tok, scheme: str, style: str, n: int = 3) -> list[list[int]]:
    states = ['{"country":"GB","name":"Rosa Pizza 1"}', '{"country":"JP","locality":"Osaka","name":"Sun Gym 2"}',
              '{"country":"DE","name":"Kings Hotel 3","tel":"0123 456789","website":"example3.com"}',
              '{"country":"FR","name":"Blue Park"}', '{"country":"US","locality":"Austin","name":"Blue Clinic 5"}']
    return lb.encode_rows(tok, [{"state": s} for s in states[:n]], scheme, style)


def _brute(model, prompt: list[int], key: list[int]) -> float:
    """One unpadded forward per (item, key) through the model's own causal-LM head."""
    import torch

    with torch.inference_mode():
        logp = torch.log_softmax(model(input_ids=torch.tensor([prompt + key])).logits[0].float(), dim=-1)
    return sum(float(logp[len(prompt) - 1 + t, tok_id]) for t, tok_id in enumerate(key))


def _keys(tok, scheme: str, style: str) -> list[list[int]]:
    prompt = lb.render(tok, lb.instruction(scheme, "{}"), style)
    return lb.key_token_ids(tok, prompt, L.option_keys(scheme), style)


@pytest.mark.torch
def test_chat_prompt_turns_qwen3_thinking_off(tiny_llm):
    tok, _ = tiny_llm
    text = lb.render(tok, lb.instruction("c10", "{}"), "chat")
    assert text.startswith("<|im_start|>user\n" + L.INSTRUCTIONS)
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


@pytest.mark.torch
@pytest.mark.parametrize("style", ["chat", "plain"])
def test_key_tokens_match_the_joint_tokenisation(tiny_llm, style):
    tok, _ = tiny_llm
    c10, c7 = _keys(tok, "c10", style), _keys(tok, "c7", style)
    assert len(c10) == 10 and len(c7) == 7
    if style == "plain":
        assert all(len(k) == 1 for k in c10)               # " arts" etc. are single tokens
    assert max(len(k) for k in c7) >= (3 if style == "chat" else 2)    # civic_services: multi-token keys


@pytest.mark.torch
@pytest.mark.parametrize("style", ["chat", "plain"])
@pytest.mark.parametrize("scheme", ["c10", "c7"])
def test_packed_and_per_key_scores_equal_brute_force(tiny_llm, scheme, style):
    tok, model = tiny_llm
    prompts, keys = _prompts(tok, scheme, style, n=2), _keys(tok, scheme, style)
    want = np.array([[_brute(model, p, k) for k in keys] for p in prompts])
    import torch
    with torch.inference_mode():
        packed = lb.packed_scores(model, prompts, keys, tok.pad_token_id)
        per_key = lb.per_key_scores(model, prompts, keys, tok.pad_token_id)
    assert np.abs(packed - want).max() < 1e-4 and np.abs(per_key - want).max() < 1e-4
    assert np.ptp(want, axis=1).min() > 0.1                 # the scores really differ between keys


@pytest.mark.torch
def test_a_mask_letting_keys_see_each_other_is_detected(tiny_llm, monkeypatch):
    """Sensitivity: the 1e-4 equality above would catch a mask bug (here: a plain causal mask on the packed row)."""
    import torch
    tok, model = tiny_llm
    prompts, keys = _prompts(tok, "c7", "chat", n=2), _keys(tok, "c7", "chat")
    want = np.array([[_brute(model, p, k) for k in keys] for p in prompts])
    monkeypatch.setattr(lb, "packed_allowed", lambda seg: torch.tril(torch.ones(seg.shape[0], seg.shape[1],
                                                                                seg.shape[1], dtype=torch.bool)))
    with torch.inference_mode():
        broken = lb.packed_scores(model, prompts, keys, tok.pad_token_id)
    assert np.abs(broken - want).max() > 1e-2


@pytest.mark.torch
@pytest.mark.parametrize("method", ["packed", "per_key"])
def test_batching_does_not_change_scores(tiny_llm, method):
    tok, model = tiny_llm
    prompts, keys = _prompts(tok, "c7", "chat", n=5), _keys(tok, "c7", "chat")
    runs = [lb.Scorer(model, keys, tok.pad_token_id, bs, method).raw(prompts) for bs in (1, 2, 7 * 5)]
    assert all(np.abs(r - runs[0]).max() < 1e-4 for r in runs[1:])
    check = lb.verify(lb.Scorer(model, keys, tok.pad_token_id, 2), prompts, half=False)
    assert check["passed"] and check["max_abs_diff"] < 1e-4


@pytest.mark.torch
def test_softmax_over_keys_is_a_distribution_with_the_raw_argmax(tiny_llm):
    tok, model = tiny_llm
    raw = lb.Scorer(model, _keys(tok, "c10", "chat"), tok.pad_token_id, 4).raw(_prompts(tok, "c10", "chat"))
    logp = lb.log_softmax(raw)
    assert np.allclose(np.exp(logp).sum(1), 1.0) and (logp.argmax(1) == raw.argmax(1)).all()
    assert np.allclose(lb.log_softmax(raw - 7.0), logp)     # a per-row shift changes nothing


# ---------------------------------------------------------------- CLI end to end (tiny model, CPU)

def _data_dir(root: Path) -> tuple[Path, Path]:
    from synth import rows_from_records, synthetic_records, write_synthetic_data_dir

    data = write_synthetic_data_dir(root / "data", n_train=8, n_val=24)
    write_jsonl(data / "ood_country.jsonl", rows_from_records(synthetic_records(15, 7), "ood_country"))
    stripped = [{**r, "state": json.dumps({"country": json.loads(r["state"])["country"]}), "has_evidence": False}
                for r in rows_from_records(synthetic_records(6, 9), "stripped_test")]
    write_jsonl(data / "stripped_test.jsonl", stripped)
    trap = root / "eval" / "trap_candidates.jsonl"
    write_jsonl(trap, rows_from_records(synthetic_records(5, 11), "trap_candidates"))
    return data, trap


def _config(tmp: Path) -> Path:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["phase4"]["llm_eval"] = {**cfg["phase4"]["llm_eval"], "subset_per_pool": 10, "val_for_temperature": 12}
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _run(tiny_llm_dir: Path, data: Path, trap: Path, config: Path, run_dir: Path) -> int:
    return lb.main(["--model", "qwen3_4b", "--scheme", "c10", "--data-dir", str(data), "--run-dir", str(run_dir),
                    "--splits", "val", "test_id", "ood_country", "stripped_test", "trap_candidates",
                    "--extra", f"trap_candidates={trap}", "--device", "cpu", "--init", str(tiny_llm_dir),
                    "--batch-size", "3", "--config", str(config)])


@pytest.mark.torch
def test_cli_writes_the_run_layout_through_baseline_eval_deterministically(tiny_llm_dir, tmp_path):
    from laya_poc import baseline_eval as be

    data, trap = _data_dir(tmp_path)
    config = _config(tmp_path)
    runs = [tmp_path / "runs" / name for name in ("a", "b")]
    assert all(_run(tiny_llm_dir, data, trap, config, r) == 0 for r in runs)
    ev = {s: json.loads((runs[0] / "eval" / f"{s}.json").read_text(encoding="utf-8"))
          for s in ("val", "test_id", "ood_country", "stripped_test", "trap_candidates")}
    assert {s: (p["n"], p["n_source"]) for s, p in ev.items()} == {
        "val": (12, 24), "test_id": (10, 24), "ood_country": (10, 15), "stripped_test": (6, 6),
        "trap_candidates": (5, 5)}
    assert all(p["subset"] is True and EVAL_KEYS <= set(p) for p in ev.values())
    assert "tau" in ev["val"] and STRIPPED_KEYS <= set(ev["stripped_test"])
    assert ev["stripped_test"]["n_abstained_no_evidence"] == 6 and ev["test_id"]["n_abstained_no_evidence"] == 0
    rows = read_jsonl(data / "test_id.jsonl")
    keep, note = lb.select_rows("test_id", rows, {"subset_per_pool": 10, "subset_seed": 20260925})
    direct, _ = be.split_outputs(rows=keep, keys=L.option_keys("c10"), scores=np.zeros((10, 10)), T=1.0,
                                 split="test_id", model="qwen3_4b", ckpt="x", evidence_fields=["name"],
                                 abstain_cfg={}, tau=ev["val"]["tau"], meta={**note, "rows": "r", "device": "cpu",
                                                                             "batch_size": 3})
    assert list(ev["test_id"]) == list(direct)                            # the exact baseline_eval schema
    preds = read_jsonl(runs[0] / "preds" / "test_id.jsonl")
    assert [p["id"] for p in preds] == [r["id"] for r in keep]
    assert all(abs(sum(p["p"]) - 1) < 1e-9 and abs(p["answer_confidence"] - max(p["p"])) < 1e-9 for p in preds)
    assert len(read_jsonl(runs[0] / "preds" / "no_gate" / "stripped_test.jsonl")) == 6
    cal = json.loads((runs[0] / "calibration.json").read_text(encoding="utf-8"))
    assert cal["fitted_on"] == "val" and cal["n"] == 12 and cal["T"] == ev["val"]["temperature"] > 0
    assert cal["extra"]["scoring"] == "packed" and cal["extra"]["verify"]["passed"]
    assert cal["extra"]["dtype"] == "float32" and cal["extra"]["subsets"]["test_id"]["n"] == 10
    assert "{state}" in cal["extra"]["prompt_template"] and not (runs[0] / "order_invariance.json").exists()
    for sub in ("preds/val.jsonl", "preds/test_id.jsonl", "preds/no_gate/stripped_test.jsonl"):
        assert (runs[0] / sub).read_bytes() == (runs[1] / sub).read_bytes()


def test_cli_fails_cleanly_on_an_unknown_model(tmp_path, capsys):
    from synth import write_synthetic_data_dir

    data = write_synthetic_data_dir(tmp_path / "data")
    rc = lb.main(["--model", "gpt_nope", "--data-dir", str(data), "--run-dir", str(tmp_path / "run"),
                  "--splits", "val", "--device", "cpu"])
    assert rc == 1 and "not in config phase4.llms" in capsys.readouterr().err
