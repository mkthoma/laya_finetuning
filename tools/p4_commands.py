"""The CLI commands the Phase 4 notebook runs (Phase 4 spec §0, §5, §6), rendered like the Phase 3 ones.

Same `Target` as before: the Colab GPU runtime (the defaults) or the local CPU dry run. The env check, the frozen
data build and the variants builds (c7 and traps only: no learning-curve subsets) are the Phase 3 renderings,
unchanged. Every baseline runs through `python -m laya_poc.matrix run --only B1|B3|B4|B5` (spec §5), which
dispatches each run to laya_poc.baseline_runs, small_encoder or llm_baseline and writes the Phase 3 run layout
under runs/p4. The local dry run gives B4 and B5 their own tiny stand-ins through `--init` (P4Target).
"""
from __future__ import annotations

from dataclasses import dataclass

from notebook_commands import Target, _p
from p3_commands import CARD, MATRIX, p3_commands, variant_command

REPORT_MD = "phase4_report.md"
SKIP_OPTIONAL = "--skip-optional"  # matrix run: an optional arm (B5) is recorded as "optional, disabled"
BASELINE_KINDS = ("majority", "prior", "tfidf_lr")  # runs named fsq-{scheme}-{arm}-{kind}: no model, no seed
INIT_KINDS = {"small_encoder": "encoder_init", "llm": "llm_init"}  # which stand-in a kind gets in the dry run


@dataclass(frozen=True)
class P4Target(Target):
    """A Target plus the local dry run's `matrix run --init` stand-ins: a tiny ModernBERT dir for B4 and a tiny
    causal LM dir for B5 (None on Colab: the pinned Hub models). `init` stays the tiny Laya checkpoint, whose
    tokenizer budgets the data build and the variants."""
    encoder_init: str | None = None
    llm_init: str | None = None


def arms(cfg: dict) -> list[dict]:
    return list(cfg["phase4"]["arms"])


def arm_ids(cfg: dict) -> list[str]:
    """Arm groups in config order, each once (B4 has one config row per model): B1, B3, B4, B5."""
    return list(dict.fromkeys(a["id"] for a in arms(cfg)))


def optional_arm(cfg: dict, arm_id: str) -> bool:
    """True when every config row of the arm is `optional: true` (B5): the notebook has a RUN_<arm> flag."""
    rows = [a for a in arms(cfg) if a["id"] == arm_id]
    return bool(rows) and all(a.get("optional") for a in rows)


def arm_kinds(cfg: dict, arm_id: str) -> list[str]:
    return list(dict.fromkeys(a["kind"] for a in arms(cfg) if a["id"] == arm_id))


def _schemes(arm: dict) -> list[str]:
    return list(arm.get("schemes") or [arm["scheme"]])


def _kinds(arm: dict) -> tuple[str, ...]:
    """B1 (`kind: majority`) is two runs per scheme: the majority class and the class prior (spec §1)."""
    return ("majority", "prior") if arm["kind"] == "majority" else (arm["kind"],)


def row_run_names(arm: dict) -> list[str]:
    """Run names of one config row (spec §1): fsq-{scheme}-B1-{majority|prior}, fsq-{scheme}-B3-tfidf_lr,
    fsq-c10-B4-{model}-s{seed}, fsq-c10-B5-{model}."""
    names, arm_id = [], arm["id"]
    for scheme in _schemes(arm):
        if arm["kind"] in BASELINE_KINDS:
            names += [f"fsq-{scheme}-{arm_id}-{kind}" for kind in _kinds(arm)]
        elif arm.get("seeds"):
            names += [f"fsq-{scheme}-{arm_id}-{arm['model']}-s{seed}" for seed in arm["seeds"]]
        else:
            names.append(f"fsq-{scheme}-{arm_id}-{arm['model']}")
    return names


def run_names(cfg: dict, arm_id: str) -> list[str]:
    """Run names of one arm group, in config order."""
    return [name for arm in arms(cfg) if arm["id"] == arm_id for name in row_run_names(arm)]


def all_run_names(cfg: dict) -> list[str]:
    return [name for arm in arm_ids(cfg) for name in run_names(cfg, arm)]


def optional_run_names(cfg: dict) -> list[str]:
    return [name for arm in arm_ids(cfg) if optional_arm(cfg, arm) for name in run_names(cfg, arm)]


def variant_dirs(cfg: dict) -> list[str]:
    """Directories under WORK that Step 6 builds: the 7-class labels (B1/B3 c7) and the trap candidates as rows."""
    c7 = ["data_c7"] if any("c7" in _schemes(a) for a in arms(cfg)) else []
    traps = ["data_eval"] if "trap_candidates" in cfg["phase3"]["eval_splits"] else []
    return [*c7, *traps]


def arm_init(cfg: dict, arm_id: str, t: Target) -> str | None:
    """The `--init` stand-in of an arm group in the local dry run (B4: tiny ModernBERT, B5: tiny causal LM)."""
    inits = {getattr(t, INIT_KINDS[k], None) for k in arm_kinds(cfg, arm_id) if k in INIT_KINDS} - {None}
    if len(inits) > 1:
        raise ValueError(f"{arm_id} mixes kinds {arm_kinds(cfg, arm_id)} that need different --init stand-ins")
    return next(iter(inits), None)


def matrix_run(only: str, t: Target, init: str | None = None) -> tuple[str, list]:
    """`matrix run --only <arm>`: skips finished runs, redoes an unfinished one (spec §5). Phase 4 runs live under
    runs/p4 (the matrix's --runs-root default for Phase 4 arms)."""
    return MATRIX, ["run", "--work", _p("WORK"), "--data-root", _p("WORK"), "--only", only,
                    *(["--init", init] if init else []), "--device", t.device, "--card", CARD]


def report_command() -> tuple[str, list]:
    """`matrix report` over this session's runs/p4: every group, decision criterion 1 (early read) and the
    Phase 4 exit check -> results/phase4_report.md (+ .json)."""
    return MATRIX, ["report", "--work", _p("WORK"), "--runs-root", _p("RUNS"),
                    "--out", _p("WORK", "results", REPORT_MD)]


def p4_commands(cfg: dict, t: Target) -> dict[str, tuple[str, list]]:
    """Every CLI of the Phase 4 notebook, keyed by step (variants: variant_<dir>, arms: their id): (module, args)."""
    p3 = p3_commands(cfg, t)
    cmds = {"env": p3["env"], "data": p3["data"]}  # env expects CARD; data = E1 Step 5 (--verify-frozen)
    cmds.update({f"variant_{d}": variant_command(d, t) for d in variant_dirs(cfg)})
    cmds.update({arm: matrix_run(arm, t, arm_init(cfg, arm, t)) for arm in arm_ids(cfg)})
    cmds["report"] = report_command()
    return cmds
