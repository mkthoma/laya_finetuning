"""Phase 7 results workbook (laya_poc.phase7_workbook) from synthetic inputs: a Phase 5 report built from the
synthetic metrics of test_decision_eval, a bench_cpu v2-like JSON, a GPU timing JSON from synthetic run archives, an
optional GPU latency bench and sample-agreement stats; read back with openpyxl. Sheets, frozen headers, numbers as
numbers, the FSQ NOTICE verbatim, the pending GPU latency row, provenance, and no row id anywhere."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

from laya_poc import notice as N  # noqa: E402
from laya_poc import phase5_report as R  # noqa: E402
from laya_poc import phase7_timing as T  # noqa: E402
from laya_poc import phase7_workbook as W  # noqa: E402
from laya_poc.io_utils import sha256_file  # noqa: E402

from test_decision_eval import bench_json, metrics_json  # noqa: E402
from test_phase7_timing import b3_run, b4_run, b5_run, laya_run  # noqa: E402

ROW_ID = re.compile(r"\b[a-z][a-z0-9_]*-\d{6}\b")
SHEETS = ["README", "Decision", "Main results", "Robustness", "Bootstrap", "Per-class", "Look-alikes", "CPU",
          "GPU throughput", "GPU latency", "Samples", "Provenance"]
BUNDLE = "ab" * 32


def _dump(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def cpu_bench() -> dict:
    """bench_json's rows as a bench_cpu v2 sweep: results, machine block, onnx {model: acceptance + export}."""
    b = bench_json()
    acc = {"n": 1000, "batch_size": 32, "max_dp": 1.27e-05, "max_dp_threshold": 0.001, "argmax_agree": 1.0,
           "min_argmax_agree": 0.999, "n_disagree": 0, "max_dlogit": 4.6e-05, "seq_len_range": [158, 281],
           "passed": True, "rows": "C:/somewhere/data/test_id.jsonl", "seconds": 700.5}
    onnx = {m: {"accepted": True, "acceptance": acc, "export": {"exporter": "dynamo", "opset": 18, "seconds": 103.4,
                                                                "bytes": 1688657900}} for m in ("laya", "laya_ml")}
    machine = {**b["machine"], "python": "3.14.4", "versions": {"torch": "2.14.0+cpu", "onnxruntime": "1.30.0"}}
    return {"form": "sweep", "schema_version": 2, "results": b["rows"], "machine": machine,
            "hardware": "Test CPU (8C/16T, 32 GB RAM)", "onnx": onnx, "latency_n": 200, "batch_n": 500,
            "batch_size": 32, "rows": "C:/somewhere/data/test_id.jsonl", "errors": [], "threads_skipped": []}


def samples_stats() -> dict:
    return {"schema": "phase7_sample_agreement/1", "n_per_split": 50,
            "splits": {"test_id": {"runs": {"E2 laya c10": {"acc": 0.62, "abstained": 1, "n": 50},
                                            "B4 mmbert_small c10": {"acc": 0.66, "abstained": 0, "n": 50}},
                                   "agreement": {"E2 laya c10 vs B4 mmbert_small c10":
                                                 {"both_right": 25, "a_only": 6, "b_only": 8, "both_wrong": 11}}}},
            "ids": ["test_id-000123", "test_id-000456"], "note": "picked test_id-000789 first"}


LOCAL_CKPT = "C:\\Users\\someone\\ckpts\\best"


def gpu_bench() -> dict:
    return {"schema": "bench_gpu/1", "gpu": {"name": "NVIDIA RTX PRO 6000", "card": "G4"},
            "plan": {"ckpt": {"laya_ml": LOCAL_CKPT}, "rows": "/content/laya_poc/data/test_id.jsonl"},
            "sources": {"laya_ml": {"kind": "laya", "path": LOCAL_CKPT, "label": "local:best"}},
            "rows": [{"model": "laya_ml", "batch_size": 1, "p50_ms": 12.5, "p95_ms": 14.0, "rows_per_s": 80.0},
                     {"model": "laya_ml", "batch_size": 64, "p50_ms": 50.0, "p95_ms": 55.0, "rows_per_s": 1280.0}]}


@pytest.fixture(scope="module")
def cfg():
    from laya_poc.config import load_config
    from conftest import ROOT
    return load_config(ROOT / "config.yaml")


@pytest.fixture()
def inputs(tmp_path, cfg):
    report = R.build_report(metrics_json(), [bench_json()], cfg)
    p3, p4 = tmp_path / "arch" / "p3_colab" / "runs" / "p3", tmp_path / "arch" / "p4_colab" / "runs" / "p4"
    laya_run(p3, 11, 2.33)
    b4_run(p4, 11, 0.48)
    b5_run(p4)
    b3_run(p4)
    _dump(p3 / "env.json", {"card": "G4", "gpu_name": "NVIDIA RTX PRO 6000", "bundle_sha256": BUNDLE})
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "phase3_report_x.md").write_text("<!-- Copied from ... (notebook x, bundle fae1ee11). -->\n# R\n",
                                              encoding="utf-8")
    fp = _dump(tmp_path / "fingerprint.json", {"release": "2026-09-15", "fingerprint_sha256": "cd" * 32,
                                                "split_sizes": {"train": 25000, "test_id": 3000}})
    nt = tmp_path / "NOTICE_FSQ.txt"
    nt.write_text(N.fsq_notice("2026-09-15"), encoding="utf-8")
    return {"report": _dump(tmp_path / "report.json", report), "bench": _dump(tmp_path / "cpu.json", cpu_bench()),
            "gpu": _dump(tmp_path / "gpu_timing.json", T.build_timing([p3, p4])),
            "samples": _dump(tmp_path / "samples.json", samples_stats()),
            "bench_gpu": _dump(tmp_path / "bench_gpu.json", gpu_bench()), "notice": nt, "fingerprint": fp,
            "archives": [tmp_path / "arch" / "p3_colab", tmp_path / "arch" / "p4_colab"], "notes": notes,
            "out": tmp_path / "out" / "wb.xlsx", "report_obj": report}


def argv(inp: dict, *, samples: bool = True, bench_gpu: bool = False, archives: bool = True) -> list[str]:
    a = ["--report", str(inp["report"]), "--bench-cpu", str(inp["bench"]), "--gpu-timing", str(inp["gpu"]),
         "--notice", str(inp["notice"]), "--fingerprint", str(inp["fingerprint"]), "--notes-dir", str(inp["notes"]),
         "--out", str(inp["out"])]
    a += ["--samples-stats", str(inp["samples"])] if samples else []
    a += ["--bench-gpu", str(inp["bench_gpu"])] if bench_gpu else []
    a += [x for p in inp["archives"] for x in ("--archive", str(p))] if archives else []
    return a


def build(inp: dict, **kw):
    assert W.main(argv(inp, **kw)) == 0
    return openpyxl.load_workbook(inp["out"])


def rows(ws) -> list[tuple]:
    return list(ws.iter_rows(values_only=True))


def table(ws) -> list[dict]:
    """The first table of a sheet (header in row 1) as dicts, up to the first empty row."""
    head, out = rows(ws)[0], []
    for r in rows(ws)[1:]:
        if all(v is None for v in r):
            break
        out.append(dict(zip(head, r)))
    return out


def all_strings(wb) -> list[str]:
    return [c for ws in wb.worksheets for r in rows(ws) for c in r if isinstance(c, str)]


def test_every_sheet_with_a_frozen_bold_header(inputs):
    wb = build(inputs)
    assert wb.sheetnames == SHEETS
    for ws in wb.worksheets:
        assert ws.freeze_panes == "A2", ws.title
        assert all(c.font.bold for c in ws[1] if c.value is not None), ws.title
        assert ws.column_dimensions["A"].width >= 8


def test_readme_notice_verbatim_verdict_and_sheet_guide(inputs):
    wb = build(inputs)
    strings = [c for r in rows(wb["README"]) for c in r if isinstance(c, str)]
    notice = inputs["notice"].read_text(encoding="utf-8").rstrip("\n")
    assert notice in strings and "modified" in notice
    ws = wb["README"]
    row = next(c.row for r in ws.iter_rows() for c in r if c.value == notice)
    assert ws.row_dimensions[row].height >= 15 * notice.count("\n")          # wrapped rows sized to their text
    verdict = inputs["report_obj"]["decision"]["verdict"]
    assert any(verdict in s for s in strings)
    for name in SHEETS[1:]:
        assert name in strings


def test_notice_without_the_modified_statement_is_refused(inputs, capsys):
    inputs["notice"].write_text(N.FSQ_NOTICE, encoding="utf-8")
    assert W.main(argv(inputs)) == 1
    assert "NOTICE" in capsys.readouterr().err and not inputs["out"].exists()


def test_decision_checklist_numbers_and_verdicts(inputs):
    wb = build(inputs)
    t = table(wb["Decision"])
    dec = inputs["report_obj"]["decision"]
    e2 = dec["candidates"][0]
    c1 = next(r for r in t if r["Candidate"] == e2["id"] and r["Check"] == "criterion" and r["ID"] == "C1")
    assert isinstance(c1["Value"], float) and c1["Value"] == pytest.approx(e2["criteria"][0]["value"])
    assert c1["Threshold"] == pytest.approx(0.03) and c1["Result"] == e2["criteria"][0]["status"]
    ids = {(r["Candidate"], r["Check"], r["ID"]) for r in t}
    assert {(e2["id"], "stop", "a"), (e2["id"], "investigate", "seed_range"), (e2["id"], "verdict", None)} <= ids
    assert any(r["Check"] == "criterion (sensitivity)" for r in t)
    overall = next(r for r in t if r["Candidate"] == "overall")
    assert overall["Result"] == dec["verdict"]
    strings = [c for r in rows(wb["Decision"]) for c in r if isinstance(c, str)]
    assert "Details" in strings and "ci.B4.ci_low" in strings                 # flattened criterion leaves


def test_main_results_one_table_with_pool_column(inputs):
    wb = build(inputs)
    ws = wb["Main results"]
    t = table(ws)
    main = inputs["report_obj"]["tables"]["main"]
    want = next(r for r in main["test_id"] if r["group"] == "E2 laya c10")
    got = next(r for r in t if r["Pool"] == "test_id" and r["Group"] == "E2 laya c10")
    assert got["Macro-F1 mean"] == pytest.approx(want["macro_f1"]["mean"])
    assert got["Macro-F1 min"] == pytest.approx(want["macro_f1"]["min"]) and got["Seeds"] == 3
    assert got["ECE post"] == pytest.approx(want["ece_post"])
    assert {r["Pool"] for r in t} == {"test_id", "ood_country", "ood_script", "ood_brand"}
    col = rows(ws)[0].index("Macro-F1 mean") + 1
    assert ws.cell(row=2, column=col).number_format == "0.0000"


def test_robustness_and_bootstrap(inputs):
    wb = build(inputs)
    rob = table(wb["Robustness"])
    assert {r["Group"] for r in rob} >= {"E2 laya c10", "B3 tfidf_lr c10"}
    boot = table(wb["Bootstrap"])
    comps = inputs["report_obj"]["tables"]["bootstrap"]
    assert len(boot) == len(comps)
    assert isinstance(boot[0]["CI low"], float) and boot[0]["Candidate"] == comps[0]["candidate"]


def test_per_class_and_lookalikes(inputs):
    wb = build(inputs)
    pc = table(wb["Per-class"])
    conf = inputs["report_obj"]["tables"]["confusion"]["test_id"]
    gid = next(iter(conf["per_class"]))
    key = conf["labels"][0]
    got = next(r for r in pc if r["Pool"] == "test_id" and r["Group"] == gid and r["Class"] == key)
    want = conf["per_class"][gid][key]
    assert got["F1 mean"] == pytest.approx(want["f1"]["mean"] if isinstance(want["f1"], dict) else want["f1"])
    la = table(wb["Look-alikes"])
    assert la and {"True class", "Predicted", "Rate"} <= set(la[0])


def test_cpu_rows_machine_and_onnx_acceptance(inputs):
    wb = build(inputs)
    ws = wb["CPU"]
    t = table(ws)
    row = next(r for r in t if r["Model"] == "laya" and r["Backend"] == "onnx" and r["Threads"] == 4)
    assert row["p95 ms"] == pytest.approx(120.0) and not isinstance(row["p95 ms"], str)   # a number (120.0 -> 120)
    assert row["Hardware"] == "Test CPU (8C/16T, 32 GB RAM)"
    strings = [c for r in rows(ws) for c in r if isinstance(c, str)]
    assert "Machine" in strings and "cpu_model" in strings and "ONNX acceptance" in strings
    onnx_head = next(i for i, r in enumerate(rows(ws)) if r[0] == "ONNX acceptance") + 1
    head = rows(ws)[onnx_head]
    first = dict(zip(head, rows(ws)[onnx_head + 1]))
    assert first["Max |dp|"] == pytest.approx(1.27e-05) and first["Accepted"] is True


def test_gpu_throughput_sheet(inputs):
    wb = build(inputs)
    t = table(wb["GPU throughput"])
    e3 = next(r for r in t if r["Group"] == "E3 laya_ml c10")
    assert e3["Rows/s (median)"] == pytest.approx(3000 / 2.33, abs=0.1) and e3["Card"] == "G4"
    strings = [c for r in rows(wb["GPU throughput"]) for c in r if isinstance(c, str)]
    assert "Per group and split" in strings and "Per run" in strings and "not batch-1 latency" in " ".join(strings)


def test_gpu_latency_pending_without_bench_and_rows_with_it(inputs):
    wb = build(inputs)
    assert any("pending: run notebooks/phase7_gpu_bench.ipynb" in str(c) for r in rows(wb["GPU latency"]) for c in r)
    wb = build(inputs, bench_gpu=True)
    t = table(wb["GPU latency"])
    assert [r["p95_ms"] for r in t] == [14.0, 55.0]


def test_samples_flattened_without_row_ids_and_pending_when_absent(inputs):
    wb = build(inputs)
    t = table(wb["Samples"])
    run = next(r for r in t if r["Path"] == "splits / test_id / runs / E2 laya c10")
    assert run["acc"] == pytest.approx(0.62) and run["n"] == 50
    agree = next(r for r in t if r["Path"].endswith("E2 laya c10 vs B4 mmbert_small c10"))
    assert agree["both_right"] == 25
    assert not any(ROW_ID.search(s) for s in all_strings(wb))
    wb = build(inputs, samples=False)
    assert any("not given" in str(c) for r in rows(wb["Samples"]) for c in r)


def test_provenance(inputs, cfg):
    wb = build(inputs)
    t = table(wb["Provenance"])
    items = {r["Item"]: r["Value"] for r in t}
    assert items["Laya commit"] == cfg["laya"]["commit"] and items["Laya hub revision"] == cfg["laya"]["hub_revision"]
    assert items["Frozen data fingerprint sha256"] == "cd" * 32 and items["FSQ release"] == "2026-09-15"
    from conftest import ROOT
    assert items["config.yaml sha256"] == sha256_file(ROOT / "config.yaml")
    assert any(cfg["phase4"]["small_encoders"]["mmbert_small"]["revision"] in str(v) for v in items.values())
    assert items["Decision: lead_points"] == 3.0
    bundle = [r for r in t if r["Item"].startswith("Notebook bundle")]
    assert any(BUNDLE in str(r["Value"]) for r in bundle) and any("fae1ee11" in str(r["Value"]) for r in bundle)
    assert any(r["Item"] == "Input sha256: report" and r["Value"] == sha256_file(inputs["report"]) for r in t)


def test_bundle_not_found_is_said(inputs):
    wb = build(inputs, archives=False)
    bundle = [r for r in table(wb["Provenance"]) if r["Item"].startswith("Notebook bundle sha256 (archives)")]
    assert bundle and "not recorded" in str(bundle[0]["Value"])


def test_no_row_id_or_local_path_anywhere(inputs):
    wb = build(inputs, bench_gpu=True)
    strings = all_strings(wb)
    assert not [s for s in strings if ROW_ID.search(s)]
    assert not [s for s in strings if "someone" in s or "/content/" in s]  # the bench's local paths are dropped


def test_guard_refuses_a_row_id_in_any_cell():
    from laya_poc.phase7_xlsx import Column, Sheet, Table, check_metrics_only
    sheet = Sheet("X", [Table(None, [Column("a", "A")], [{"a": "leaked test_id-000123"}])])
    with pytest.raises(ValueError, match="row id"):
        check_metrics_only([sheet])


@pytest.mark.parametrize("cell", ["C:\\Users\\x\\ckpt", "see D:/data/run", "/home/x/ckpt", "/content/laya_poc/runs",
                                  "(/Users/x/models)"])
def test_guard_refuses_a_local_absolute_path_in_any_cell(cell):
    from laya_poc.phase7_xlsx import Column, Sheet, Table, check_metrics_only
    with pytest.raises(ValueError, match="local path"):
        check_metrics_only([Sheet("X", [Table(None, [Column("a", "A")], [{"a": cell}])])])


@pytest.mark.parametrize("cell", ["http://www.apache.org/licenses/LICENSE-2.0", "runs\\p3_colab\\runs\\p3",
                                  "docs/results/x.md", "ratio 3:1", "hf://datasets/foursquare"])
def test_guard_allows_urls_and_relative_paths(cell):
    from laya_poc.phase7_xlsx import Column, Sheet, Table, check_metrics_only
    check_metrics_only([Sheet("X", [Table(None, [Column("a", "A")], [{"a": cell}])])])


def test_cpu_rows_ignore_a_non_list_rows_field():
    from laya_poc.phase7_workbook_tables import bench_result_rows
    assert bench_result_rows({"rows": "C:/data/test_id.jsonl"}) == []
    assert bench_result_rows({"rows": 7}) == []
