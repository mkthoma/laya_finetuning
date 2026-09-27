"""The CLI commands the Phase 3 notebook runs (Phase 3 spec §0-§6), rendered like the smoke and E1 ones.

Same `Target` as before: the Colab GPU runtime (the defaults) or the local CPU dry run (tiny checkpoint, synthetic
pool). The card is the notebook variable CARD (Step 1; config phase3.card), so a user on another GPU changes one
line. The data build and parity are the E1 and Phase 1 renderings, unchanged; the variants and matrix CLIs are
rendered exactly as the Phase 3 spec defines them (§3, §5).
"""
from __future__ import annotations

from e1_commands import e1_commands
from notebook_commands import Expr, Target, _p, commands

CARD = Expr("CARD")  # Step 1 defines it: the card profile every run trains with
MATRIX, VARIANTS = "laya_poc.matrix", "laya_poc.variants"
B2 = "B2"  # the zero-shot arm: evaluation of the shipped checkpoints, no training


def arm_ids(cfg: dict) -> list[str]:
    """Trained arm ids in config order, each once (E5 has one config row per model)."""
    return list(dict.fromkeys(a["id"] for a in cfg["phase3"]["arms"]))


def subset_sizes(cfg: dict) -> list[int]:
    """The E6 learning-curve train subsets (every arm's train_subset), ascending."""
    return sorted({n for a in cfg["phase3"]["arms"] for n in a.get("train_subset") or ()})


def run_names(cfg: dict, arm_id: str) -> list[str]:
    """Run names of one arm (spec §1): fsq-{scheme}-{arm}-{model}-s{seed}[-n{subset}][-head]; B2: -zs (c10)."""
    p3 = cfg["phase3"]
    if arm_id == B2:
        return [f"fsq-c10-B2-{m}-zs" for m in p3.get("zero_shot") or ()]
    names = []
    for arm in (a for a in p3["arms"] if a["id"] == arm_id):
        head = "-head" if arm.get("freeze_encoder") else ""
        for seed in arm["seeds"]:
            for n in arm.get("train_subset") or [None]:
                names.append(f"fsq-{arm['scheme']}-{arm_id}-{arm['model']}-s{seed}{f'-n{n}' if n else ''}{head}")
    return names


def all_run_names(cfg: dict) -> list[str]:
    """Every configured run: B2 first, then the trained arms in notebook order."""
    return [name for arm in (B2, *arm_ids(cfg)) for name in run_names(cfg, arm)]


def variant_dirs(cfg: dict) -> list[str]:
    """Directories under WORK that Step 6 builds (spec §0.3): c7 labels, E6 subsets, trap candidates as rows."""
    p3 = cfg["phase3"]
    c7 = ["data_c7"] if any(a["scheme"] == "c7" for a in p3["arms"]) else []
    traps = ["data_eval"] if "trap_candidates" in p3["eval_splits"] else []
    return [*c7, *(f"data_lc{n}" for n in subset_sizes(cfg)), *traps]


def _tokenizer_args(t: Target) -> list[str]:
    """build_data's tokenizer flags: every variant row is budget-checked with the same tokenizers (spec §3)."""
    return [*(["--init-tokenizer", t.init] if t.init else []), *(["--models", *t.data_models] if t.data_models else [])]


def variant_command(name: str, t: Target) -> tuple[str, list]:
    """`python -m laya_poc.variants ...` that writes WORK/<name> from the frozen WORK/data."""
    src, out = ["--data-dir", _p("DATA")], ["--out", _p("WORK", name)]
    if name == "data_c7":
        return VARIANTS, ["c7", *src, *out, *_tokenizer_args(t)]
    if name == "data_eval":
        return VARIANTS, ["traps", *src, *out, *_tokenizer_args(t)]
    n = name.removeprefix("data_lc")
    if not n.isdigit():
        raise ValueError(f"unknown variant directory {name!r}")
    return VARIANTS, ["subset", *src, "--n", n, *out, *_tokenizer_args(t)]


def matrix_run(only: str, t: Target) -> tuple[str, list]:
    """`matrix run --only <arm>`: skips finished runs, resumes an interrupted one (spec §5)."""
    return MATRIX, ["run", "--work", _p("WORK"), "--data-root", _p("WORK"), "--only", only,
                    *(["--init", t.init] if t.init else []), "--device", t.device, "--card", CARD]


def p3_commands(cfg: dict, t: Target) -> dict[str, tuple[str, list]]:
    """Every CLI of the Phase 3 notebook, keyed by step (variants: variant_<dir>): (module, args)."""
    smoke, work = commands(cfg, t), ["--work", _p("WORK")]
    cmds = {"env": ("laya_poc.env_check", ["--out", _p("RUNS", "env.json"),
                                           *(["--require-gpu"] if t.require_gpu else []), "--expect-card", CARD]),
            "data": e1_commands(cfg, t)["data"]}  # identical to E1 Step 5: --verify-frozen
    cmds.update({f"variant_{d}": variant_command(d, t) for d in variant_dirs(cfg)})
    cmds.update({k: smoke[k] for k in ("parity_laya", "parity_laya_ml")})  # design §7.7: again on each new card
    cmds["plan"] = (MATRIX, ["plan", *work])
    if cfg["phase3"].get("zero_shot"):
        cmds[B2] = matrix_run(B2, t)
    cmds.update({arm: matrix_run(arm, t) for arm in arm_ids(cfg)})
    cmds["report"] = (MATRIX, ["report", *work, "--out", _p("WORK", "results", "phase3_report.md")])
    return cmds
