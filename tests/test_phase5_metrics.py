"""Phase 5 metrics (spec P5 §2): recomputation from preds vs the eval JSONs, grouping, eval-JSON aggregates, CLI.

Synthetic runs are written through baseline_eval.write_outputs, the exact eval JSON + preds writer of every run
(its scorer is evaluate's), so a match here is a match with the in-session numbers.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pytest

from laya_poc import phase5_metrics as M
from laya_poc.baseline_eval import write_outputs
from laya_poc.labels import option_keys
from laya_poc.phase5_classes import answered_at_tau, run_split
from laya_poc.phase5_load import SplitPreds, load_run, make_groups, stat5
from synth import rows_from_records, synthetic_records

EVIDENCE = ["name", "address", "locality", "tel", "website"]
SPLITS = ("val", "test_id", "ood_country", "ood_script", "ood_brand")


# ---------------------------------------------------------------- synthetic runs (shared with the other files)

def split_rows(split: str, n: int, scheme: str = "c10", seed: int = 0, no_evidence: int = 0) -> list[dict]:
    """Synthetic eval rows; the first `no_evidence` rows keep only `country` (the gate abstains on them)."""
    recs = synthetic_records(n, seed)
    recs = [({"country": r["country"]} if i < no_evidence else r, lab) for i, (r, lab) in enumerate(recs)]
    if scheme == "c7":
        from laya_poc.labels import C7_MAP
        recs = [(r, C7_MAP[lab]) for r, lab in recs]
    return rows_from_records(recs, split, scheme=scheme)


def scores_for(rows: list[dict], keys: list[str], skill: float, seed: int) -> np.ndarray:
    """Logits: noise plus `skill` on the true class (a skill of 0 is chance)."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(len(rows), len(keys)))
    for i, r in enumerate(rows):
        Z[i, keys.index(r["label"])] += skill
    return Z


def write_run(root: Path, cfg: dict, name: str, *, arm: str, model: str, seed: int | None, kind: str | None = None,
              scheme: str = "c10", skill: float = 2.0, n: int = 120, splits=SPLITS, T: float = 1.2,
              drop: tuple = (), no_evidence: int = 0, row_seed: int = 0) -> Path:
    """One finished run dir (eval/, preds/, done.json, order_invariance.json) in the Phase 3/4 layout."""
    rd, keys = root / name, option_keys(scheme)
    tau = None
    for j, split in enumerate(("val", *[s for s in splits if s != "val"])):
        rows = split_rows(split, n if split != "trap_candidates" else 40, scheme, seed=row_seed + 10 * j,
                          no_evidence=no_evidence if split != "val" else 0)
        rows = [r for i, r in enumerate(rows) if (split, i) not in drop]
        Z = scores_for(rows, keys, skill, zlib.crc32(f"{name}|{split}".encode()))
        pay = write_outputs(rd / "eval", rd / "preds", rows=rows, keys=keys, scores=Z, T=T, split=split,
                            model=model, ckpt="test", evidence_fields=EVIDENCE, abstain_cfg=cfg["abstain"], tau=tau)
        tau = pay.get("tau", tau)
    done = {"run_name": name, "arm": arm, "model": model, "scheme": scheme, "seed": seed, "subset": None,
            "head_only": False, "card": "G4", **({"kind": kind} if kind else {})}
    (rd / "done.json").write_text(json.dumps(done), encoding="utf-8")
    (rd / "order_invariance.json").write_text(json.dumps({"mean_agreement": 0.95, "passed_99": False}),
                                             encoding="utf-8")
    return rd


def write_matrix(root: Path, cfg: dict, n: int = 120) -> Path:
    """E2 laya x2, E3 laya_ml x3, B1 majority, B3, B4 mmbert_small x2, B4 modernbert_base x2."""
    for s in (11, 22):
        write_run(root, cfg, f"fsq-c10-E2-laya-s{s}", arm="E2", model="laya", seed=s, skill=2.5, n=n)
    for s in (11, 22, 33):
        write_run(root, cfg, f"fsq-c10-E3-laya_ml-s{s}", arm="E3", model="laya_ml", seed=s, skill=2.8, n=n)
    write_run(root, cfg, "fsq-c10-B1-majority", arm="B1", model="majority", seed=None, kind="majority", skill=0.0, n=n)
    write_run(root, cfg, "fsq-c10-B3-tfidf_lr", arm="B3", model="tfidf_lr", seed=None, kind="tfidf_lr", skill=1.5, n=n)
    for m in ("mmbert_small", "modernbert_base"):
        for s in (11, 22):
            write_run(root, cfg, f"fsq-c10-B4-{m}-s{s}", arm="B4", model=m, seed=s, kind="small_encoder", n=n)
    return root


@pytest.fixture(scope="module")
def mcfg():
    from laya_poc.config import load_config
    cfg = load_config()
    return {**cfg, "phase5": {**cfg["phase5"], "bootstrap": {"resamples": 200, "ci": 0.95, "seed": 7}}}


@pytest.fixture(scope="module")
def matrix(tmp_path_factory, mcfg):
    root = write_matrix(tmp_path_factory.mktemp("p5") / "runs", mcfg)
    return root, M.build(mcfg, [root], M.load_traps(M.parse_args(["--runs-root", str(root), "--out", "x"]), mcfg,
                                                     [root]), {"runs_roots": [str(root)]})


# ---------------------------------------------------------------- recomputation vs the eval JSON

def test_recomputed_macro_f1_equals_the_eval_json_for_every_run_and_split(matrix):
    root, res = matrix
    assert res["recomputed_check"]["passed"] and res["recomputed_check"]["max_abs_diff"] <= 1e-12
    assert res["recomputed_check"]["compared"] == 11 * len(SPLITS)
    for g in res["groups"].values():
        for split in SPLITS:
            b = g["splits"][split]
            assert b["recomputed"]["all"]["macro_f1"]["values"] == b["post"]["macro_f1"]["values"]


def test_per_class_confusion_and_tau_metrics_equal_the_eval_json_detail(matrix):
    root, _ = matrix
    run = load_run(root / "fsq-c10-E3-laya_ml-s22")
    for split in SPLITS:
        ev, keys = run.evals[split], run.labels(split)
        got = run_split(run.preds[split], keys, ev["abstention"]["tau"])
        assert got["cm"].tolist() == ev["post"]["detail"]["cm"]
        for k in keys:
            for m in ("p", "r", "f1"):
                assert got["per_class"][k][m] == pytest.approx(ev["post"]["detail"]["per_class"][k][m], abs=1e-15)
        assert got["macro_f1_9"] == ev["post"]["macro_f1_9"] and got["acc"] == ev["post"]["acc"]
        assert got["answered"]["coverage"] == pytest.approx(ev["abstention"]["coverage"], abs=1e-15)
        assert got["answered"]["acc"] == pytest.approx(ev["abstention"]["answered_acc"], abs=1e-15)


def test_group_cm_is_summed_over_seeds_and_lookalike_rates_use_true_support(matrix):
    root, res = matrix
    g = res["groups"]["E3 laya_ml c10"]["splits"]["test_id"]["recomputed"]
    cms = [np.array(load_run(root / f"fsq-c10-E3-laya_ml-s{s}").evals["test_id"]["post"]["detail"]["cm"])
           for s in (11, 22, 33)]
    assert g["cm"] == np.sum(cms, axis=0).tolist() and g["cm_runs"] == 3
    keys = g["labels"]
    cells = {(c["true"], c["pred"]): c for c in g["lookalike"]}
    assert len(cells) == 8 and ("arts", "dining") in cells and ("dining", "arts") in cells
    i, j = keys.index("retail"), keys.index("services")
    c = cells[("retail", "services")]
    assert c["count"] == sum(int(m[i, j]) for m in cms) and c["support"] == sum(int(m[i].sum()) for m in cms)
    assert c["rate"] == pytest.approx(c["count"] / c["support"])
    assert c["rate_runs"]["values"] == [m[i, j] / m[i].sum() for m in cms]


def test_gate_abstentions_count_as_abstentions_in_coverage(tmp_path, mcfg):
    rd = write_run(tmp_path, mcfg, "fsq-c10-E3-laya_ml-s11", arm="E3", model="laya_ml", seed=11, n=60,
                   splits=("val", "test_id"), no_evidence=15)
    run = load_run(rd)
    sp, ev = run.preds["test_id"], run.evals["test_id"]
    got = run_split(sp, run.labels("test_id"), ev["abstention"]["tau"])
    assert got["n_abstained"] == 15 and got["n_scored"] == 45 and got["macro_f1"] == ev["post"]["macro_f1"]
    at = answered_at_tau(sp, run.labels("test_id"), ev["abstention"]["tau"])
    assert at["n_answered"] == ev["abstention"]["n_answered"]
    assert at["coverage"] == pytest.approx(at["n_answered"] / 60)      # the 15 gated rows are abstentions too
    assert answered_at_tau(sp, run.labels("test_id"), None)["coverage"] is None


def test_a_split_where_the_gate_abstains_on_every_row_has_no_recomputed_block(tmp_path, mcfg):
    rd = write_run(tmp_path, mcfg, "r", arm="E3", model="laya_ml", seed=11, n=30, splits=("val", "test_id"),
                   no_evidence=30)
    run = load_run(rd)
    assert run_split(run.preds["test_id"], run.labels("test_id")) is None


# ---------------------------------------------------------------- eval-JSON aggregates and groups

def test_eval_aggregates_gaps_and_ood_averages(matrix):
    _, res = matrix
    g = res["groups"]["E2 laya c10"]
    tid, ctry = g["splits"]["test_id"]["post"]["macro_f1"], g["splits"]["ood_country"]["post"]["macro_f1"]
    assert g["splits"]["ood_country"]["gap"]["values"] == [a - b for a, b in zip(tid["values"], ctry["values"])]
    assert "gap" not in g["splits"]["test_id"]
    avg2 = g["ood_average"]["ood_country+ood_brand"]
    brand = g["splits"]["ood_brand"]["post"]["macro_f1"]["values"]
    assert avg2["macro_f1"]["values"] == pytest.approx([(a + b) / 2 for a, b in zip(ctry["values"], brand)])
    assert set(g["ood_average"]) == {"ood_country+ood_script+ood_brand", "ood_country+ood_brand"}
    assert g["ood_pools"] == ["ood_country", "ood_brand"]
    assert res["groups"]["E3 laya_ml c10"]["ood_pools"] == ["ood_country", "ood_script", "ood_brand"]
    for side in ("pre", "post"):
        assert set(g["splits"]["test_id"][side]) == set(M.METRIC_KEYS)
    assert g["splits"]["test_id"]["selective"]["coverage"]["n"] == 2
    assert g["order_invariance"] == {"mean_agreement": stat5([0.95, 0.95]), "all_passed_99": False}


def test_roles_and_the_language_matched_small_encoder(matrix):
    _, res = matrix
    gs = res["groups"]
    assert gs["E2 laya c10"]["role"] == "candidate" and gs["B3 tfidf_lr c10"]["role"] == "baseline"
    assert gs["E2 laya c10"]["matched_small_encoder"] == "B4 modernbert_base c10"
    assert gs["E3 laya_ml c10"]["matched_small_encoder"] == "B4 mmbert_small c10"
    assert gs["B4 mmbert_small c10"]["matched_small_encoder"] is None
    assert gs["E3 laya_ml c10"]["seeds"] == [11, 22, 33] and gs["E3 laya_ml c10"]["n_runs"] == 3


def test_group_ids_and_roles_for_subsets_head_only_and_zero_shot():
    def rd(name, **rec):
        from laya_poc.phase5_load import RunData
        return RunData({"run_name": name, "kind": "laya", **rec}, {})
    runs = [rd("a", arm="E6", model="laya", scheme="c10", subset=1000, head_only=False),
            rd("b", arm="E4", model="laya", scheme="c10", subset=None, head_only=True),
            rd("c", arm="B2", model="laya", scheme="c10", subset=None, head_only=False, zero_shot=True),
            rd("d", arm="E5", model="laya", scheme="c7", subset=None, head_only=False)]
    groups = make_groups(runs, ["E2", "E3"], ["E5"])
    assert {g: v.role for g, v in groups.items()} == {"B2 laya c10": "baseline", "E4 laya c10 head": "other",
                                                      "E5 laya c7": "report_also", "E6 laya c10 n1000": "other"}


def test_flip_rate_is_reported_for_multi_seed_groups_only(matrix):
    _, res = matrix
    e3 = res["groups"]["E3 laya_ml c10"]["splits"]["test_id"]
    assert 0 < e3["flip_rate"] < 1 and e3["flip_n"] == 120
    assert res["groups"]["B3 tfidf_lr c10"]["splits"]["test_id"]["flip_rate"] is None



def _sp(ids, yhat, abstained=()):
    n, k = len(ids), 10
    P = np.full((n, k), 0.01)
    P[np.arange(n), yhat] = 0.9
    ab = np.array([i in abstained for i in ids], bool)
    P[ab] = np.nan
    return SplitPreds(tuple(ids), np.zeros(n, int), P, np.where(ab, np.nan, 0.9), ab)


def test_flip_rate_aligns_rows_by_id_and_skips_rows_any_seed_abstained():
    from laya_poc.phase5_classes import group_flip
    a = _sp(["a", "b", "c", "d"], [1, 2, 3, 4])
    b = _sp(["d", "c", "b", "a"], [4, 3, 2, 5])            # same answers except row a, in another order
    c = _sp(["a", "b", "c", "d"], [1, 2, 9, 4], abstained=("c",))
    assert group_flip([a, b]) == (0.25, 4)
    assert group_flip([a, b, c]) == (pytest.approx(1 / 3), 3)
    assert group_flip([a]) == (None, None)


def test_lookalike_pairs_absent_from_a_scheme_are_skipped():
    from laya_poc.phase5_classes import lookalike_cells
    keys = option_keys("c7")
    cm = np.arange(len(keys) ** 2).reshape(len(keys), len(keys))
    cells = lookalike_cells([cm, cm], keys, [["dining", "arts"], ["retail", "civic_services"]])
    assert [(c["true"], c["pred"]) for c in cells] == [("retail", "civic_services"), ("civic_services", "retail")]
    i, j = keys.index("retail"), keys.index("civic_services")
    assert cells[0]["count"] == 2 * cm[i, j] and cells[0]["support"] == 2 * cm[i].sum()
    zero = lookalike_cells([np.zeros_like(cm)], keys, [["retail", "civic_services"]])
    assert zero[0]["rate"] is None and zero[0]["rate_runs"] is None

def test_stat5():
    assert stat5([None, "x"]) is None
    s = stat5([0.5, None, 0.7])
    assert s["mean"] == pytest.approx(0.6) and s["range"] == pytest.approx(0.2) and s["n"] == 2
    assert s["values"] == [0.5, None, 0.7]


def test_read_preds_rejects_a_width_mismatch(tmp_path):
    from laya_poc.phase5_load import read_preds
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"id": "a", "y": 0, "p": [0.5, 0.5], "answer_confidence": 0.5, "abstained": False}) + "\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="2 probabilities for 3 labels"):
        read_preds(p, 3)
    sp = read_preds(p, 2)
    assert isinstance(sp, SplitPreds) and sp.yhat.tolist() == [0]


def test_eval_labels_out_of_option_order_are_rejected(tmp_path, mcfg):
    rd = write_run(tmp_path, mcfg, "r", arm="E3", model="laya_ml", seed=11, n=20, splits=("val", "test_id"))
    ev = json.loads((rd / "eval" / "test_id.json").read_text(encoding="utf-8"))
    (rd / "eval" / "test_id.json").write_text(json.dumps({**ev, "labels": ev["labels"][::-1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="not the c10 option keys"):
        load_run(rd)


# ---------------------------------------------------------------- CLI

def test_cli_writes_the_schema_and_prints_sparse_lines(tmp_path, mcfg, capsys):
    import yaml
    root = tmp_path / "runs"
    for s in (11, 22):
        write_run(root, mcfg, f"fsq-c10-E3-laya_ml-s{s}", arm="E3", model="laya_ml", seed=s, n=50)
    write_run(root, mcfg, "fsq-c10-B3-tfidf_lr", arm="B3", model="tfidf_lr", seed=None, kind="tfidf_lr", n=50)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(mcfg), encoding="utf-8")
    out = tmp_path / "out" / "m.json"
    assert M.main(["--runs-root", str(root), "--out", str(out), "--config", str(cfg_path)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["schema"] == M.SCHEMA and set(res) >= {"groups", "bootstrap", "traps", "recomputed_check", "settings"}
    assert res["traps"]["status"] == "pending" and "id_map" not in res["traps"]
    assert {c["baseline"] for c in res["bootstrap"]["comparisons"]} == {"B3 tfidf_lr c10"}
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3 and "PASS" in lines[-1]


def test_cli_fails_with_one_line_on_an_empty_root(tmp_path, capsys):
    assert M.main(["--runs-root", str(tmp_path), "--out", str(tmp_path / "m.json")]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "no finished run" in err[0] and not (tmp_path / "m.json").exists()
