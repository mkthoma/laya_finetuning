"""Phase 7 sample inferences on SYNTHETIC rows and predictions (never FSQ data): record rendering and escaping, the
id / label joins and their failures, accuracy and the focus-vs-versus 2x2, deterministic sampling, the no-evidence
(stripped) section, the NOTICE + LOCAL ONLY header, the metrics-only stats JSON and the git-ignore guard."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from laya_poc import phase7_samples as S
from laya_poc import phase7_samples_md as M
from laya_poc.io_utils import write_jsonl
from laya_poc.labels import option_keys

KEYS = option_keys("c10")
NOTICE = "Copyright 2024 Example Labs.\n\n    http://example.invalid/LICENSE\n\nDerived from X; modified: sampled.\n"
LOCAL_ONLY = ("**LOCAL ONLY: contains FSQ-derived rows. Do not commit or share outside the team "
              "(design doc Appendix D).**")
N = 40


# ---------------------------------------------------------------- synthetic fixtures

def probs(top: int, second: int) -> list[float]:
    p = [0.15 / 8] * len(KEYS)
    p[top], p[second] = 0.6, 0.25
    return p


def _truth(i: int) -> int:
    return i % len(KEYS)


def _wrong(i: int, shift: int = 1) -> int:
    return (_truth(i) + shift) % len(KEYS)


def focus_pred(i: int) -> int:
    """alpha: right when i % 4 in (0, 1)."""
    return _truth(i) if i % 4 in (0, 1) else _wrong(i)


def versus_pred(i: int) -> int:
    """beta: right when i % 4 in (0, 2); in the both-wrong cell (i % 4 == 3) the same wrong label when i % 8 == 3."""
    if i % 4 in (0, 2):
        return _truth(i)
    return _wrong(i) if i % 8 in (1, 3) else _wrong(i, 2)


def data_rows(split: str, n: int = N, evidence: bool = True) -> list[dict]:
    rows = []
    for i in range(n):
        state = ({"name": f"Synthetic Place {i}", "locality": "Townville", "country": "GB",
                  "website": f"http://site{i}.invalid"} if evidence else {"country": "GB"})
        rows.append({"id": f"{split}-{i:06d}", "split": split, "state": json.dumps(state), "label": KEYS[_truth(i)],
                     "country": "GB", "has_evidence": evidence})
    return rows


def pred_rows(split: str, pick, n: int = N, abstain: bool = False) -> list[dict]:
    out = []
    for i in range(n):
        top = pick(i)
        p = None if abstain else probs(top, _wrong(i, 3) if top != _wrong(i, 3) else _wrong(i, 4))
        out.append({"id": f"{split}-{i:06d}", "y": _truth(i), "p": p,
                    "answer_confidence": None if abstain else max(p), "abstained": abstain})
    return out


def make_run(root: Path, name: str, model: str, pick, splits=("test_id",), labels=KEYS) -> Path:
    d = root / f"fsq-c10-{name}"
    for split in splits:
        stripped = split == "stripped_test"
        write_jsonl(d / "preds" / f"{split}.jsonl", pred_rows(split, pick, abstain=stripped))
        if stripped:
            write_jsonl(d / "preds" / "no_gate" / f"{split}.jsonl", pred_rows(split, pick))
        (d / "eval").mkdir(parents=True, exist_ok=True)
        (d / "eval" / f"{split}.json").write_text(json.dumps({"labels": list(labels), "model": model}),
                                                  encoding="utf-8")
    return d


@pytest.fixture()
def world(tmp_path):
    data = tmp_path / "data"
    splits = ("test_id", "ood_script", "stripped_test")
    for split in splits:
        write_jsonl(data / f"{split}.jsonl", data_rows(split, evidence=split != "stripped_test"))
    (data / "NOTICE_FSQ.txt").write_text(NOTICE, encoding="utf-8")
    runs = {"alpha": make_run(tmp_path / "runs", "alpha", "laya_ml", focus_pred, splits),
            "beta": make_run(tmp_path / "runs", "beta", "mmbert_small", versus_pred, splits),
            "eng": make_run(tmp_path / "runs", "eng", "laya", focus_pred, splits)}
    return {"root": tmp_path, "data": data, "runs": runs, "splits": splits}


def cli(world, *extra: str, out: str = "out/samples.md", stats: str = "out/stats.json") -> list[str]:
    root = world["root"]
    args = ["--data-dir", str(world["data"]), "--focus", "alpha", "--versus", "beta",
            "--splits", *world["splits"], "--per-cell", "2", "--seed", "7",
            "--out", str(root / out), "--stats-out", str(root / stats)]
    for name, d in world["runs"].items():
        args += ["--run", f"{name}={d}"]
    return args + list(extra)


# ---------------------------------------------------------------- record rendering

def test_render_record_orders_name_locality_country_then_others():
    state = {"country": "GB", "website": "http://x.invalid", "name": "Corner Cafe", "locality": "Leeds"}
    assert M.render_record(state) == "Corner Cafe · Leeds · GB · website=http://x.invalid"


def test_render_record_truncates_long_values_and_escapes_markdown():
    state = {"name": "A" * 200, "address": "1 Pipe | Street\nLine two", "country": "TH"}
    out = M.render_record(state)
    name = out.split(" · ")[0]
    assert len(name) <= M.MAX_VALUE and name.endswith("…")
    assert "\n" not in out and "\\|" in out and "| " not in out.replace("\\|", "")
    assert "address=1 Pipe \\| Street Line two" in out


def test_render_record_without_evidence_is_just_the_country():
    assert M.render_record({"country": "TH"}) == "TH"


# ---------------------------------------------------------------- loading and joins

def test_load_split_joins_preds_by_id(world):
    sd = S.load_split(world["data"], world["runs"], "test_id")
    assert sd.labels == tuple(KEYS) and len(sd.rows) == N
    assert sd.models == {"alpha": "laya_ml", "beta": "mmbert_small", "eng": "laya"}
    first = sd.rows[0]["id"]
    assert sd.preds["alpha"][first].top == focus_pred(0)


def test_id_mismatch_fails_with_a_clear_message(world):
    path = world["runs"]["beta"] / "preds" / "test_id.jsonl"
    rows = pred_rows("test_id", versus_pred)[:-1]
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match=r"beta test_id: prediction ids differ from the data ids \(1 missing, 0 extra"):
        S.load_split(world["data"], world["runs"], "test_id")


def test_label_order_mismatch_between_runs_fails(world):
    make_run(world["root"] / "runs", "beta", "mmbert_small", versus_pred, labels=list(reversed(KEYS)))
    with pytest.raises(ValueError, match="label order"):
        S.load_split(world["data"], world["runs"], "test_id")


def test_pred_y_disagreeing_with_the_data_label_fails(world):
    path = world["runs"]["alpha"] / "preds" / "test_id.jsonl"
    rows = pred_rows("test_id", focus_pred)
    rows[3] = {**rows[3], "y": _wrong(3)}
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="does not match the data label"):
        S.load_split(world["data"], world["runs"], "test_id")


def test_probability_width_must_match_the_labels(world):
    path = world["runs"]["alpha"] / "preds" / "test_id.jsonl"
    rows = pred_rows("test_id", focus_pred)
    rows[0] = {**rows[0], "p": rows[0]["p"][:5]}
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="5 probabilities for 10 labels"):
        S.load_split(world["data"], world["runs"], "test_id")


# ---------------------------------------------------------------- metrics

def test_run_summary_accuracy_over_answered_rows(world):
    sd = S.load_split(world["data"], world["runs"], "test_id")
    assert S.run_summary(sd, "alpha") == {"n": N, "answered": N, "abstained": 0, "correct": N // 2,
                                          "accuracy": 0.5, "judged": True}
    stripped = S.load_split(world["data"], world["runs"], "stripped_test")
    summ = S.run_summary(stripped, "alpha")
    assert summ["abstained"] == N and summ["answered"] == 0 and summ["accuracy"] is None


def test_english_model_is_not_judged_on_the_thai_split(world):
    sd = S.load_split(world["data"], world["runs"], "ood_script")
    assert S.run_summary(sd, "eng")["judged"] is False
    assert S.run_summary(sd, "alpha")["judged"] is True


def test_agreement_two_by_two_and_same_wrong_label(world):
    sd = S.load_split(world["data"], world["runs"], "test_id")
    ag = S.agreement(sd, "alpha", "beta")
    assert ag.counts() == {"n": N, "both_right": 10, "focus_only_right": 10, "versus_only_right": 10,
                           "both_wrong": 10, "both_wrong_same_label": 5, "either_abstained": 0}
    assert all(sd.preds["alpha"][i].top == sd.y[i] == sd.preds["beta"][i].top for i in ag.cells["both_right"])


def test_sampling_is_deterministic_sorted_and_bounded():
    ids = [f"x-{i:04d}" for i in range(100)]
    a = S.sample_ids(ids, 5, np.random.default_rng(1))
    b = S.sample_ids(ids, 5, np.random.default_rng(1))
    assert a == b == sorted(a) and len(set(a)) == 5 and set(a) <= set(ids)
    assert S.sample_ids(ids[:2], 5, np.random.default_rng(1)) == ids[:2]
    assert S.sample_ids([], 5, np.random.default_rng(1)) == []


def test_same_seed_same_markdown_different_seed_different_sample(world):
    root = world["root"]
    assert S.main(cli(world)) == 0
    first = (root / "out/samples.md").read_text(encoding="utf-8")
    assert S.main(cli(world)) == 0
    assert _body(first) == _body((root / "out/samples.md").read_text(encoding="utf-8"))
    args = cli(world)
    args[args.index("--seed") + 1] = "8"
    assert S.main(args) == 0
    assert _body(first) != _body((root / "out/samples.md").read_text(encoding="utf-8"))


def _body(md: str) -> str:
    return "\n".join(line for line in md.splitlines() if not line.startswith("Generated "))


# ---------------------------------------------------------------- markdown

@pytest.fixture()
def rendered(world):
    assert S.main(cli(world)) == 0
    return (world["root"] / "out/samples.md").read_text(encoding="utf-8")


def test_markdown_starts_with_the_notice_verbatim_then_local_only(rendered):
    assert rendered.startswith(NOTICE.rstrip("\n"))
    rest = rendered[len(NOTICE.rstrip("\n")):].lstrip("\n")
    assert rest.startswith(LOCAL_ONLY)
    assert "## How to read this" in rendered


def test_notice_without_the_modified_statement_is_refused(world):
    (world["data"] / "NOTICE_FSQ.txt").write_text("Copyright only.\n", encoding="utf-8")
    assert S.main(cli(world)) == 1
    assert not (world["root"] / "out/samples.md").exists()


def test_markdown_split_sections_tables_and_examples(rendered):
    assert "## test_id: in-distribution test" in rendered
    assert "## ood_script: non-Latin script (Thai)" in rendered
    assert "alpha (laya_ml): accuracy 0.5000 (20/40 right, 0 abstained)" in rendered
    assert "| **alpha right** | 10 | 10 |" in rendered
    assert "| **alpha wrong** | 10 | 10 (same wrong label: 5) |" in rendered
    for cell in ("Both right", "Only alpha right", "Only beta right", "Both wrong"):
        assert f"### {cell} (10 rows; showing 2)" in rendered
    assert "| Record | True label | alpha | beta | eng | alpha runner-up |" in rendered
    assert "Synthetic Place" in rendered and " · Townville · GB · website=" in rendered
    assert "(0.60) ✓" in rendered and "(0.60) ✗" in rendered and "(0.25)" in rendered


def test_markdown_marks_the_english_model_on_the_thai_split(rendered):
    section = rendered[rendered.index("## ood_script"):rendered.index("## stripped_test")]
    assert "eng (not judged: English model)" in section
    assert "eng (laya): accuracy 0.5000 (20/40 right, 0 abstained; not judged: English model)" in section
    test_id = rendered[rendered.index("## test_id"):rendered.index("## ood_script")]
    assert "not judged" not in test_id


def test_markdown_stripped_section_shows_abstention_and_the_ungated_answer(rendered):
    section = rendered[rendered.index("## stripped_test"):]
    assert "no evidence" in section
    assert "alpha (laya_ml): abstained 40/40 (1.0000)" in section
    assert "| abstained; ungated: " in section and "(0.60) ✓ |" in section
    assert "Both right" not in section


# ---------------------------------------------------------------- stats JSON

def test_stats_json_is_metrics_only(world, rendered):
    text = (world["root"] / "out/stats.json").read_text(encoding="utf-8")
    stats = json.loads(text)
    assert stats["seed"] == 7 and stats["per_cell"] == 2 and stats["focus"] == "alpha" and stats["versus"] == "beta"
    assert stats["runs"]["alpha"] == {"dir": "fsq-c10-alpha", "model": "laya_ml"}
    sp = stats["splits"]["test_id"]
    assert sp["runs"]["beta"]["accuracy"] == 0.5 and sp["agreement"]["both_wrong_same_label"] == 5
    assert sp["sampled"] == {"both_right": 2, "focus_only_right": 2, "versus_only_right": 2, "both_wrong": 2}
    assert stats["splits"]["stripped_test"]["runs"]["alpha"]["abstained"] == N
    assert stats["splits"]["ood_script"]["runs"]["eng"]["judged"] is False
    for leak in ("Synthetic Place", "Townville", "site0", "test_id-0000", "stripped_test-0000", "GB"):
        assert leak not in text


def test_outputs_are_written_atomically(world, rendered):
    out = world["root"] / "out"
    assert sorted(p.name for p in out.iterdir()) == ["samples.md", "stats.json"]


# ---------------------------------------------------------------- CLI guards

def test_unknown_focus_run_fails(world):
    args = cli(world)
    args[args.index("--focus") + 1] = "gamma"
    assert S.main(args) == 1


def test_bad_run_spec_is_a_usage_error(world):
    with pytest.raises(SystemExit):
        S.main(cli(world, "--run", "no-equals-sign"))


@pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
def test_refuses_a_markdown_path_git_would_track(world):
    root = world["root"]
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    assert S.main(cli(world, out="tracked/samples.md", stats="tracked/stats.json")) == 1
    assert not (root / "tracked/samples.md").exists()
    assert S.main(cli(world, out="ignored/samples.md", stats="tracked/stats.json")) == 0
    assert (root / "ignored/samples.md").is_file()
