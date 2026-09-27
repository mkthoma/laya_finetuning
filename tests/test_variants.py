"""Phase 3 data variants (spec P3 §3): c7 relabelling, stratified train subsets (E6) and the trap-candidate
eval rows, built from a synthetic FULL-mode build_data run. CPU, no network beyond the tiny tokenizer."""
import json
import shutil

import pandas as pd
import pytest

from laya_poc import labels as L
from laya_poc.config import load_config
from laya_poc.io_utils import read_jsonl, sha256_file, write_jsonl

pytestmark = pytest.mark.torch
pytest.importorskip("laya")

from laya_poc import variants as V  # noqa: E402
from laya_poc.serialise import dumps, make_record  # noqa: E402
from test_build_data_full import _full  # noqa: E402

C10_Q = L.question("c10")
C7_Q = L.question("c7")
UNIFORM_C7 = json.dumps(L.gold(None, "c7", 0.0), ensure_ascii=False)


@pytest.fixture(scope="module")
def built(tmp_path_factory, cfg, tiny_ckpt_dir):
    """(scratch root, FULL data dir, config path, tiny checkpoint) shared by every test here."""
    root = tmp_path_factory.mktemp("variants")
    code, data = _full(root, cfg, tiny_ckpt_dir)
    assert code == 0
    return root, data, root / "config_data.yaml", tiny_ckpt_dir


def _run(built, command, out, *extra, data=None):
    _, src, conf, tiny = built
    return V.main([command, "--data-dir", str(data or src), "--out", str(out), "--config", str(conf),
                   "--models", "laya", "--init-tokenizer", str(tiny), *extra])


def _variant(out):
    return json.loads((out / "variant.json").read_text(encoding="utf-8"))


def _report(data):
    return json.loads((data / "data_report.json").read_text(encoding="utf-8"))


def _copy(src, dst, names):
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copyfile(src / name, dst / name)
    return dst


# ---- pure: stratified selection ------------------------------------------------------------------------

def test_stratified_quotas_use_largest_remainder_with_name_ties():
    assert V.stratified_quotas({"b": 3, "a": 5, "c": 2}, 5) == {"a": 3, "b": 1, "c": 1}
    assert V.stratified_quotas({"a": 5, "b": 3, "c": 2}, 10) == {"a": 5, "b": 3, "c": 2}
    assert V.stratified_quotas({"a": 1, "b": 1, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}
    with pytest.raises(ValueError):
        V.stratified_quotas({"a": 2}, 3)


REAL_LIKE = dict(zip(L.L1_TO_KEY, (2476, 2504, 2502, 2527, 2490, 2516, 2469, 2523, 2492, 2501)))  # real train


def _train_frame(counts):
    rows = [{"fsq_place_id": f"p{label[:3]}{i:05d}", "label": label} for label, n in counts.items() for i in range(n)]
    return pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)


def test_learning_curve_subsets_are_proportional_nested_and_in_split_order():
    train = _train_frame(REAL_LIKE)
    picked = {n: V.stratified_subset(train, n, seed=7) for n in (1000, 3000, 10000)}
    for n, sub in picked.items():
        assert len(sub) == n
        counts = sub["label"].value_counts().to_dict()
        assert counts == V.stratified_quotas(REAL_LIKE, n)
        assert all(abs(counts[k] - n * c / 25000) < 1 for k, c in REAL_LIKE.items())
        positions = train.index[train["fsq_place_id"].isin(set(sub["fsq_place_id"]))].tolist()
        assert sub["fsq_place_id"].tolist() == train.loc[positions, "fsq_place_id"].tolist()  # split order
    ids = {n: set(sub["fsq_place_id"]) for n, sub in picked.items()}
    assert ids[1000] < ids[3000] < ids[10000]  # E6 subsets nest
    assert V.stratified_subset(train, 1000, seed=7).equals(picked[1000])  # deterministic
    assert set(V.stratified_subset(train, 1000, seed=8)["fsq_place_id"]) != ids[1000]


@pytest.mark.parametrize("n", [0, 25000, 30000])
def test_subset_size_must_be_between_one_and_the_train_size(n):
    with pytest.raises(ValueError, match="--n"):
        V.stratified_subset(_train_frame(REAL_LIKE), n, seed=1)


# ---- c7 ------------------------------------------------------------------------------------------------

def _row(label, split="val", uniform=False):
    from synth import rows_from_records
    row = rows_from_records([({"country": "GB", "name": "Rosa Cafe"}, label)], split)[0]
    return {**row, "gold": json.dumps(L.gold(None, "c10", 0.0))} if uniform else row


def test_to_c7_rules_on_single_rows():
    assert V.to_c7(_row("arts"), 0.1)["gold"] == json.dumps(L.gold("culture", "c7", 0.1), ensure_ascii=False)
    stripped = V.to_c7(_row("sports", "stripped_test", uniform=True), 0.1)
    assert stripped["label"] == "leisure" and json.loads(stripped["gold"]) == json.loads(UNIFORM_C7)
    unlabelled = V.to_c7(_row(None, "train"), 0.1)
    assert unlabelled["label"] is None and json.loads(unlabelled["gold"]) == json.loads(UNIFORM_C7)
    source = _row("arts")
    V.to_c7(source, 0.1)
    assert source == _row("arts")  # input never mutated


@pytest.mark.parametrize("row, match", [
    ({**_row("arts"), "label": "culture"}, "not a c10 key"),
    (_row("arts", "stripped_test"), "is not uniform"),  # stripped rows carry the uniform no-evidence gold
    (_row("arts", uniform=True), "is uniform"),
    ({k: v for k, v in _row("arts").items() if k != "questions"}, "not a data row"),
])
def test_to_c7_rejects_inconsistent_rows(row, match):
    with pytest.raises(ValueError, match=match):
        V.to_c7(row, 0.1)


def test_c7_relabels_every_jsonl_and_keeps_everything_else(built, tmp_path, capsys):
    _, data, _, _ = built
    out = tmp_path / "data_c7"
    code = _run(built, "c7", out)
    assert code == 0, capsys.readouterr().err
    names = sorted(p.name for p in data.glob("*.jsonl"))
    assert sorted(p.name for p in out.glob("*.jsonl")) == names and "stripped_test.jsonl" in names
    for name in names:
        src, dst = read_jsonl(data / name), read_jsonl(out / name)
        assert len(src) == len(dst), name
        for a, b in zip(src, dst):
            assert list(a) == list(b)
            assert {k: v for k, v in a.items() if k not in ("questions", "gold", "label")} == \
                   {k: v for k, v in b.items() if k not in ("questions", "gold", "label")}
            assert json.loads(b["questions"]) == C7_Q
            assert b["label"] == (L.C7_MAP[a["label"]] if a["label"] is not None else None)
            if a["label"] is None or a["split"] == "stripped_test":
                assert json.loads(b["gold"]) == json.loads(UNIFORM_C7)
            else:
                assert json.loads(b["gold"]) == L.gold(b["label"], "c7", 0.1)
    stripped = read_jsonl(out / "stripped_test.jsonl")
    assert stripped and all(r["label"] in L.option_keys("c7") for r in stripped)  # TRUE label kept
    meta = _variant(out)
    assert meta["variant"] == "c7" and meta["scheme"] == "c7"
    assert meta["source"]["fingerprint_sha256"] == _report(data)["fingerprint_sha256"]
    assert meta["files"] == {n: sha256_file(out / n) for n in names}
    assert meta["rows"] == {n: len(read_jsonl(out / n)) for n in names}
    assert meta["budgets"]["laya"] >= _report(data)["budgets"]["laya"]  # 7 options leave more state room
    assert (out / "NOTICE_FSQ.txt").read_bytes() == (data / "NOTICE_FSQ.txt").read_bytes()
    assert len(capsys.readouterr().out.strip().splitlines()) <= 4


def test_c7_and_trap_rows_feed_build_items(built, tmp_path):
    from laya_poc.hub import load_tokenizer_dir
    from laya_poc.items import build_items
    _, _, _, tiny = built
    c7, traps = tmp_path / "c7", tmp_path / "traps"
    assert _run(built, "c7", c7) == 0 and _run(built, "traps", traps) == 0
    tok = load_tokenizer_dir(tiny)
    files = [(c7 / n, 7) for n in ("train_e0.jsonl", "val.jsonl", "stripped_test.jsonl")]
    for path, k in [*files, (traps / "trap_candidates.jsonl", 10), (traps / "trap_candidates_c7.jsonl", 7)]:
        for row in read_jsonl(path):
            items = build_items(tok, row, 512, 192)
            assert len(items[0]["markers"]) == k
            if row["label"] is not None and row["split"] != "stripped_test":
                assert items[0]["label"] == L.option_keys(f"c{k}").index(row["label"])


def test_c7_is_byte_deterministic(built, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert _run(built, "c7", a) == 0 and _run(built, "c7", b) == 0
    for path in sorted(a.iterdir()):
        assert path.read_bytes() == (b / path.name).read_bytes(), path.name


def test_c7_never_writes_into_the_source_or_another_build(built, tmp_path, capsys):
    """Neither --out == --data-dir nor swapped arguments (--out <a build>) may touch a build, even with --force."""
    _, data, _, _ = built
    before = sha256_file(data / "val.jsonl")
    assert _run(built, "c7", data) == 1
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1 and "--data-dir" in err
    assert sha256_file(data / "val.jsonl") == before and not (data / "variant.json").exists()
    other = _copy(data, tmp_path / "other_build", ["val.jsonl", "data_report.json"])
    assert _run(built, "c7", other, "--force") == 1 and "data_report.json" in capsys.readouterr().err
    assert sha256_file(other / "val.jsonl") == before


def test_existing_variant_output_needs_force(built, tmp_path, capsys):
    out = tmp_path / "c7"
    assert _run(built, "c7", out) == 0
    first = (out / "val.jsonl").read_bytes()
    (out / "val.jsonl").write_text("{}\n", encoding="utf-8")
    capsys.readouterr()
    assert _run(built, "c7", out) == 1 and "--force" in capsys.readouterr().err
    assert (out / "val.jsonl").read_text(encoding="utf-8") == "{}\n"  # untouched without --force
    (out / "train_e9.jsonl").write_text("{}\n", encoding="utf-8")  # stale output of another build
    (out / "keep.txt").write_text("user file", encoding="utf-8")
    assert _run(built, "c7", out, "--force") == 0
    assert (out / "val.jsonl").read_bytes() == first and not (out / "train_e9.jsonl").exists()
    assert (out / "keep.txt").exists()  # only variant files (*.jsonl, variant.json) are cleared


def test_a_failed_build_leaves_no_partial_variant(built, tmp_path, monkeypatch, capsys):
    out = tmp_path / "c7"
    calls = []

    def fail_on_second(rows, *args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("boom")

    monkeypatch.setattr(V, "assert_fits", fail_on_second)
    assert _run(built, "c7", out) == 1 and "boom" in capsys.readouterr().err
    assert out.is_dir() and not list(out.glob("*.jsonl")) and not (out / "variant.json").exists()
    monkeypatch.undo()
    assert _run(built, "c7", out) == 0  # a rerun needs no --force


def test_c7_asserts_every_state_fits_the_c7_budget(built, tmp_path, capsys):
    _, data, _, _ = built
    src = _copy(data, tmp_path / "src", ["val.jsonl", "data_report.json"])
    rows = read_jsonl(src / "val.jsonl")
    huge = {**rows[3], "state": dumps({"country": "GB", "name": " ".join(["Nnnn"] * 1500)})}
    write_jsonl(src / "val.jsonl", [*rows[:3], huge, *rows[4:]])
    out = tmp_path / "c7"
    assert _run(built, "c7", out, data=src) == 1
    err = capsys.readouterr().err
    assert rows[3]["id"] in err and "budget" in err and not (out / "variant.json").exists()


def test_c7_needs_a_c10_source(built, tmp_path, capsys):
    once = tmp_path / "c7"
    assert _run(built, "c7", once) == 0
    capsys.readouterr()
    assert _run(built, "c7", tmp_path / "c7c7", data=once) == 1
    assert "c10" in capsys.readouterr().err


def test_c7_without_a_report_fingerprint_recomputes_it(built, tmp_path):
    from laya_poc import freeze
    _, data, _, _ = built
    src = _copy(data, tmp_path / "src", ["val.jsonl", "test_id.jsonl"])
    out = tmp_path / "c7"
    assert _run(built, "c7", out, data=src) == 0
    expected = freeze.fingerprint_digest(freeze.fingerprint(src, ["test_id.jsonl", "val.jsonl"]))
    assert _variant(out)["source"] == {"fingerprint_sha256": expected, "from": "recomputed"}


# ---- subset --------------------------------------------------------------------------------------------

def test_subset_with_the_whole_split_reproduces_the_build_bytes(built, tmp_path):
    """The subset path IS build_data's augmentation + serialisation pipeline."""
    _, data, conf, tiny = built
    cfg = load_config(conf)
    ctx, _ = V.row_context(cfg, data, ["laya"], tiny, "c10")
    train = pd.read_parquet(data / "split_train.parquet")
    out = tmp_path / "whole"
    out.mkdir()
    V.write_subset(ctx, out, train, cfg)
    for name in ("train_e0.jsonl", "train_e1.jsonl", "train.jsonl"):
        assert (out / name).read_bytes() == (data / name).read_bytes(), name


def test_subset_cli_writes_train_epochs_clean_train_and_copies_val(built, tmp_path, capsys):
    _, data, conf, _ = built
    cfg = load_config(conf)
    out = tmp_path / "data_lc30"
    assert _run(built, "subset", out, "--n", "30") == 0, capsys.readouterr().err
    assert sorted(p.name for p in out.glob("*.jsonl")) == ["train.jsonl", "train_e0.jsonl", "train_e1.jsonl",
                                                           "val.jsonl"]
    assert (out / "val.jsonl").read_bytes() == (data / "val.jsonl").read_bytes()
    train = pd.read_parquet(data / "split_train.parquet")
    quotas = V.stratified_quotas(train["label"].value_counts().to_dict(), 30)
    clean = read_jsonl(out / "train.jsonl")
    assert len(clean) == 30 and [r["id"] for r in clean] == [f"train-{i:06d}" for i in range(30)]
    got = pd.Series([r["label"] for r in clean]).value_counts().to_dict()
    assert got == {L.l1_to_key(k): v for k, v in quotas.items() if v}
    assert {r["state"] for r in clean} <= {r["state"] for r in read_jsonl(data / "train.jsonl")}
    e0 = read_jsonl(out / "train_e0.jsonl")
    assert len(e0) == 30 + round(cfg["augment"]["stripped_rate"] * 30)
    assert all(r["label"] is None for r in e0 if r["aug"] in ("stripped", "emptied"))
    meta = _variant(out)
    assert meta["variant"] == "subset" and meta["scheme"] == "c10"
    assert meta["params"]["n"] == 30 and meta["params"]["seed"] == cfg["data"]["split_seed"]
    assert meta["params"]["epochs"] == 2 and meta["classes"] == {k: v for k, v in sorted(quotas.items())}
    assert meta["source"]["fingerprint_sha256"] == _report(data)["fingerprint_sha256"]
    assert meta["files"] == {p.name: sha256_file(p) for p in sorted(out.glob("*.jsonl"))}
    assert meta["rows"]["train_e0.jsonl"] == len(e0)


def test_subset_is_deterministic_and_seeded(built, tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    assert _run(built, "subset", a, "--n", "30") == 0 and _run(built, "subset", b, "--n", "30") == 0
    assert _run(built, "subset", c, "--n", "30", "--seed", "5") == 0
    for name in ("train.jsonl", "train_e0.jsonl", "train_e1.jsonl", "variant.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    assert (a / "train.jsonl").read_bytes() != (c / "train.jsonl").read_bytes()
    assert _variant(c)["classes"] == _variant(a)["classes"]


def test_subset_refuses_n_at_or_above_the_train_size(built, tmp_path, capsys):
    _, data, _, _ = built
    n = len(pd.read_parquet(data / "split_train.parquet"))
    assert _run(built, "subset", tmp_path / "o", "--n", str(n)) == 1
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1 and "--n" in err and str(n) in err
    assert not (tmp_path / "o" / "variant.json").exists()


def test_subset_refuses_models_the_source_was_not_built_for(built, tmp_path, capsys):
    code = _run(built, "subset", tmp_path / "o", "--n", "30", "--models", "laya", "laya_ml")
    assert code == 1 and "laya_ml" in capsys.readouterr().err


# ---- traps ---------------------------------------------------------------------------------------------

def _candidates(data):
    from laya_poc import traps as T
    return T.read_candidates(data / "trap_candidates.csv")


def test_traps_rows_are_serialised_like_eval_rows(built, tmp_path, capsys):
    _, data, conf, _ = built
    keep = load_config(conf)["serialise"]["keep_fields"]
    out = tmp_path / "data_eval"
    assert _run(built, "traps", out) == 0, capsys.readouterr().err
    cands = _candidates(data)
    rows = read_jsonl(out / "trap_candidates.jsonl")
    assert len(rows) == len(cands) > 0
    for i, (row, cand) in enumerate(zip(rows, cands.to_dict("records"))):
        assert row["id"] == f"trap_candidates-{i:06d}" and row["split"] == "trap_candidates"
        assert (row["fsq_place_id"], row["pattern"], row["multi"]) == (cand["fsq_place_id"], cand["pattern"],
                                                                       int(cand["multi"]))
        assert row["label"] == cand["label_key"] and json.loads(row["questions"]) == C10_Q
        assert json.loads(row["gold"]) == L.gold(cand["label_key"], "c10", 0.1)
        assert row["state"] == dumps(make_record(cand, keep)) and row["aug"] == "none" and row["has_evidence"]
    c7 = read_jsonl(out / "trap_candidates_c7.jsonl")
    assert [(r["id"], r["state"], r["fsq_place_id"]) for r in c7] == \
           [(r["id"], r["state"], r["fsq_place_id"]) for r in rows]
    assert [r["label"] for r in c7] == [L.C7_MAP[r["label"]] for r in rows]
    assert all(json.loads(r["questions"]) == C7_Q and json.loads(r["gold"]) == L.gold(r["label"], "c7", 0.1)
               for r in c7)
    meta = _variant(out)
    assert meta["variant"] == "traps" and meta["dropped"] == {"rejected": 0, "no_evidence": 0}
    assert meta["schemes"] == {"trap_candidates.jsonl": "c10", "trap_candidates_c7.jsonl": "c7"}
    assert meta["rows"] == {"trap_candidates.jsonl": len(rows), "trap_candidates_c7.jsonl": len(rows)}
    assert meta["field_values"] == "pool.parquet" and sum(meta["by_pattern"].values()) == len(rows)


def test_traps_from_the_csv_alone_match_the_pool_values(built, tmp_path):
    _, data, _, _ = built
    alone = _copy(data, tmp_path / "src", ["trap_candidates.csv", "data_report.json"])
    a, b = tmp_path / "a", tmp_path / "b"
    assert _run(built, "traps", a) == 0 and _run(built, "traps", b, data=alone) == 0
    assert _variant(b)["field_values"] == "trap_candidates.csv"
    for name in ("trap_candidates.jsonl", "trap_candidates_c7.jsonl"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_traps_drop_and_count_over_budget_and_no_evidence_rows(built, tmp_path):
    from laya_poc import traps as T
    _, data, conf, _ = built
    src = _copy(data, tmp_path / "src", ["data_report.json"])
    cands = _candidates(data)
    evidence = [f for f in load_config(conf)["serialise"]["evidence_fields"] if f in cands]
    cands.loc[0, "name"] = " ".join(["Nnnn"] * 1500)
    cands.loc[1, evidence] = ""
    T.write_candidates(cands, src / "trap_candidates.csv")
    out = tmp_path / "o"
    assert _run(built, "traps", out, data=src) == 0
    rows = read_jsonl(out / "trap_candidates.jsonl")
    assert len(rows) == len(cands) - 2 and _variant(out)["dropped"] == {"rejected": 1, "no_evidence": 1}
    assert [r["id"] for r in rows] == [f"trap_candidates-{i:06d}" for i in range(len(rows))]
    assert [r["fsq_place_id"] for r in rows] == cands["fsq_place_id"].tolist()[2:]


def test_traps_reject_an_unknown_pattern(built, tmp_path, capsys):
    from laya_poc import traps as T
    _, data, _, _ = built
    src = _copy(data, tmp_path / "src", ["data_report.json"])
    cands = _candidates(data)
    cands.loc[2, "pattern"] = "bank_wrod"
    T.write_candidates(cands, src / "trap_candidates.csv")
    out = tmp_path / "o"
    assert _run(built, "traps", out, data=src) == 1
    assert "bank_wrod" in capsys.readouterr().err and not list(out.glob("*.jsonl"))


def test_traps_without_a_candidate_csv_fail_with_one_line(built, tmp_path, capsys):
    _, data, _, _ = built
    src = _copy(data, tmp_path / "src", ["val.jsonl", "data_report.json"])
    assert _run(built, "traps", tmp_path / "o", data=src) == 1
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1 and "trap_candidates.csv" in err
