"""Phase 5 trap accuracy (spec P5 §2): pending without --traps; with an annotated trap.jsonl, accuracy on the kept
candidates joined on fsq_place_id (trap rows' own ids, or their position in the annotated CSV), multi split out."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from laya_poc import phase5_metrics as M
from laya_poc.io_utils import write_jsonl
from laya_poc.labels import option_keys
from laya_poc.phase5_load import load_run
from laya_poc.phase5_traps import blank_sha256, ids_from_csv
from laya_poc.traps import candidate_columns, pattern_names, write_candidates
from test_phase5_metrics import write_run

N = 40                                     # write_run scores 40 trap candidates
KEPT = [i for i in range(N) if i % 2 == 0]


def is_multi(i: int) -> int:
    return int(i % 5 == 0)


def candidates_frame() -> pd.DataFrame:
    keys = option_keys("c10")
    rows = [{"pattern": pattern_names()[i % 7], "fsq_place_id": f"p{i:03d}", "key": f"k{i}", "name": f"Place {i}",
             "country": "KR" if i == 4 else "GB", "label": "L1", "label_key": keys[i % 10], "multi": is_multi(i),
             "n_l1": 1 + is_multi(i), "l1s": "L1"} for i in range(N)]
    return pd.DataFrame(rows).reindex(columns=candidate_columns()).fillna("")


def annotate(csv: Path) -> None:
    """Both annotators keep the even rows (the file is re-saved as the annotation guide asks: same order)."""
    df = pd.read_csv(csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    flags = ["1" if i in KEPT else "0" for i in range(N)]
    df.assign(keep_a1=flags, keep_a2=flags).to_csv(csv, index=False, encoding="utf-8-sig", lineterminator="\n")


def trap_rows(with_ids: bool, cfg: dict) -> list[dict]:
    """What `traps merge` writes: kept in-scope candidates in CSV order (KR is not an evaluation country)."""
    from laya_poc.traps import eval_countries
    df = candidates_frame()
    scope = set(eval_countries(cfg["data"]))
    kept = [i for i in KEPT if df.loc[i, "country"] in scope]
    return [{"id": f"trap-{j:06d}", "split": "trap", "state": "{}", "label": df.loc[i, "label_key"],
             "multi": is_multi(i), "pattern": df.loc[i, "pattern"],
             **({"fsq_place_id": f"p{i:03d}"} if with_ids else {})}
            for j, i in enumerate(kept)]


@pytest.fixture()
def world(tmp_path, cfg):
    cfg = {**cfg, "phase5": {**cfg["phase5"], "bootstrap": {"resamples": 20, "ci": 0.95, "seed": 1}}}
    root = tmp_path / "work" / "runs" / "p3"
    splits = ("val", "test_id", "trap_candidates")
    for s in (11, 22):
        write_run(root, cfg, f"fsq-c10-E3-laya_ml-s{s}", arm="E3", model="laya_ml", seed=s, splits=splits, n=60)
    write_run(root, cfg, "fsq-c10-B3-tfidf_lr", arm="B3", model="tfidf_lr", seed=None, kind="tfidf_lr", splits=splits,
              n=60)
    data = tmp_path / "data"
    data.mkdir()
    csv, _ = write_candidates(candidates_frame(), data / "trap_candidates.csv")
    (tmp_path / "work" / "data_eval").mkdir()
    variant = {"params": {"candidates_sha256": blank_sha256(csv)}, "dropped": {"rejected": 0, "no_evidence": 0},
               "rows": {"trap_candidates.jsonl": N}}
    (tmp_path / "work" / "data_eval" / "variant.json").write_text(json.dumps(variant), encoding="utf-8")
    annotate(csv)
    return {"cfg": cfg, "root": root, "data": data, "csv": csv, "tmp": tmp_path}


def expected(root: Path, run: str, cfg: dict) -> dict:
    sp = load_run(root / run).preds["trap_candidates"]
    kept = {int(r["fsq_place_id"][1:]) for r in trap_rows(True, cfg)}
    idx = [int(i.split("-")[1]) for i in sp.ids]
    correct = sp.yhat == sp.y
    on = np.array([i in kept for i in idx])
    multi = np.array([is_multi(i) == 1 for i in idx])
    return {"acc": correct[on].mean(), "acc_multi": correct[on & multi].mean(),
            "acc_single": correct[on & ~multi].mean(),
            "n": int(on.sum()), "n_multi": int((on & multi).sum())}


def build(world: dict, traps: Path | None, *extra: str) -> dict:
    args = M.parse_args(["--runs-root", str(world["root"]), "--data-dir", str(world["data"]), "--out", "x",
                         *(["--traps", str(traps)] if traps else []), *extra])
    return M.build(world["cfg"], [world["root"]], M.load_traps(args, world["cfg"], [world["root"]]), {})


def check_groups(res: dict, world: dict) -> None:
    g = res["groups"]["E3 laya_ml c10"]["traps"]
    assert g["status"] == "ok" and g["basis"] == "annotated keep set"
    want = [expected(world["root"], f"fsq-c10-E3-laya_ml-s{s}", world["cfg"]) for s in (11, 22)]
    for k in ("acc", "acc_multi", "acc_single"):
        assert g[k]["values"] == pytest.approx([w[k] for w in want])
    assert g["n"] == want[0]["n"] == 19 and g["n_multi"] == want[0]["n_multi"]
    assert res["groups"]["B3 tfidf_lr c10"]["traps"]["status"] == "ok"


def test_pending_without_traps_reports_the_unannotated_candidate_accuracy(world):
    res = build(world, None)
    assert res["traps"]["status"] == "pending" and "UNANNOTATED" in res["traps"]["note"]
    g = res["groups"]["E3 laya_ml c10"]
    assert g["traps"]["status"] == "pending" and g["traps"]["basis"] == "unannotated candidates"
    evals = [load_run(world["root"] / f"fsq-c10-E3-laya_ml-s{s}").evals["trap_candidates"] for s in (11, 22)]
    assert g["traps"]["acc"]["values"] == [e["post"]["acc"] for e in evals]
    assert g["traps"]["acc_multi"] is None and g["traps"]["n"] is None


def test_trap_rows_with_place_ids_join_through_the_csv_order(world):
    traps = world["tmp"] / "trap.jsonl"
    write_jsonl(traps, trap_rows(True, world["cfg"]))
    res = build(world, traps)
    check_groups(res, world)
    t = res["traps"]
    assert t["status"] == "ok" and t["n_kept"] == 19
    assert t["n_multi"] == sum(r["multi"] for r in trap_rows(True, world["cfg"]))
    assert "trap rows' fsq_place_id" in t["join"] and "variant.json" in t["join"]


def test_trap_rows_without_place_ids_join_by_position_in_the_annotated_csv(world):
    traps = world["tmp"] / "trap.jsonl"
    write_jsonl(traps, trap_rows(False, world["cfg"]))
    res = build(world, traps)
    check_groups(res, world)
    assert "position" in res["traps"]["join"]
    bad = trap_rows(False, world["cfg"])
    bad[3] = {**bad[3], "pattern": "bank_word" if bad[3]["pattern"] != "bank_word" else "park_garden_word"}
    write_jsonl(traps, bad)
    with pytest.raises(ValueError, match="does not match kept candidate 3"):
        build(world, traps)
    write_jsonl(traps, trap_rows(False, world["cfg"])[:-1])
    with pytest.raises(ValueError, match="18 trap rows vs 19 kept"):
        build(world, traps)


def test_an_explicit_candidate_jsonl_maps_ids_without_variant_json(world):
    (world["tmp"] / "work" / "data_eval" / "variant.json").unlink()
    traps, cands = world["tmp"] / "trap.jsonl", world["tmp"] / "trap_candidates.jsonl"
    write_jsonl(traps, trap_rows(True, world["cfg"]))
    with pytest.raises(FileNotFoundError, match="--trap-candidates"):
        build(world, traps)
    write_jsonl(cands, [{"id": f"trap_candidates-{i:06d}", "fsq_place_id": f"p{i:03d}", "multi": is_multi(i)}
                        for i in range(N)])
    res = build(world, traps, "--trap-candidates", str(cands))
    check_groups(res, world)
    assert "trap_candidates.jsonl" in res["traps"]["join"]


def test_a_csv_edited_beyond_its_annotation_columns_cannot_map_ids(world):
    df = pd.read_csv(world["csv"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.loc[5, "name"] = "Renamed"
    df.to_csv(world["csv"], index=False, encoding="utf-8-sig", lineterminator="\n")
    with pytest.raises(ValueError, match="differs from the one the build scored"):
        ids_from_csv(world["csv"], world["tmp"] / "work" / "data_eval" / "variant.json")


def test_a_kept_place_that_was_never_scored_is_an_error(world):
    traps = world["tmp"] / "trap.jsonl"
    rows = trap_rows(True, world["cfg"])
    write_jsonl(traps, [*rows, {**rows[0], "fsq_place_id": "elsewhere"}])
    with pytest.raises(ValueError, match="1 kept trap places were not among the scored candidates"):
        build(world, traps)


def test_unannotated_csv_and_no_place_ids_is_an_actionable_error(world):
    (world["tmp"] / "fresh").mkdir()
    write_candidates(candidates_frame(), world["tmp"] / "fresh" / "trap_candidates.csv")
    traps = world["tmp"] / "trap.jsonl"
    write_jsonl(traps, trap_rows(False, world["cfg"]))
    with pytest.raises(ValueError, match="annotation incomplete"):
        build(world, traps, "--trap-annotations", str(world["tmp"] / "fresh" / "trap_candidates.csv"))


def test_cli_with_traps(world, tmp_path):
    import yaml
    traps = world["tmp"] / "trap.jsonl"
    write_jsonl(traps, trap_rows(True, world["cfg"]))
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(world["cfg"]), encoding="utf-8")
    out = tmp_path / "o.json"
    assert M.main(["--runs-root", str(world["root"]), "--data-dir", str(world["data"]), "--traps", str(traps),
                   "--out", str(out), "--config", str(cfg_path)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["traps"]["status"] == "ok" and set(res["traps"]) == {"status", "basis", "source", "n_kept", "n_multi",
                                                                    "join", "note"}
    assert "p000" not in out.read_text(encoding="utf-8")        # no place ids (nor rows) in the output
