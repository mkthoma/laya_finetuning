"""B5: zero-shot reference LLM scored by option-key log-likelihood (design §5.13 item 6, §7.10 "Reference LLM").

    python -m laya_poc.llm_baseline --model qwen3_4b --scheme c10 --data-dir D --run-dir R [--extra name=abs ...]
        [--splits ...] [--device cuda|cpu] [--init <abs local causal-LM dir>] [--batch-size N]
        [--prompt-style chat|plain] [--scoring packed|per_key] [--verify-n N] [--config yaml]

The LM (config phase4.llms, pinned revision; bf16 on compute capability >= 8, fp16 below, fp32 on CPU) reads one
fixed plain-text instruction (Laya's question, each option key with its labels.criteria description, the compact
JSON state) as a user turn of its chat template with thinking off (Qwen3's empty <think></think>; "plain": text +
"Answer:", keys scored as " <key>"). Key score = sum over its tokens of log p(token | prompt + its earlier
tokens), teacher-forced, nothing generated; P = softmax over keys. "packed" batching: ONE row per item = prompt +
every key's tokens but its last, under a 4D mask (prompt causal; a key token sees the prompt and its own earlier
tokens, never another key) with positions restarting after the prompt per key: one forward = the K per-key
forwards; the vocabulary projection runs only at scored positions; items are length-sorted into batches. The
first --verify-n val items are also scored "per_key" (a right-padded sequence per (item, key), standard 2D mask);
a mismatch beyond the dtype tolerance switches the run to per_key, with a warning. Subsets (config
phase4.llm_eval): val = its first val_for_temperature labelled rows (T fitted on them; the run's val split);
stripped_test, trap_candidates in full; other splits subset_per_pool rows in sha256("seed:split:id") order (all
if fewer), in file order. Outputs via baseline_eval (Phase 3 layout, payloads add subset: true) + calibration.json;
no order invariance. The HF token comes from $HF_TOKEN only.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import labels as L
from .calibrate import log_softmax
from .config import load_config
from .env_check import run_cli, write_json
from .evaluate import check_question, label_indices, load_rows
from .evaluate_splits import parse_extras, split_paths

VAL, STRIPPED = "val", "stripped_test"
FULL_SPLITS = frozenset({STRIPPED, "trap_candidates"})   # scored in full (config phase4.llm_eval)
STYLES, SCORERS = ("chat", "plain"), ("packed", "per_key")
PLAIN_CUE, CPU_BATCH, VERIFY_N, PROGRESS_SECONDS = "Answer:", 4, 4, 60
VERIFY_TOL = {"fp32": 1e-3, "half": 0.5}                 # max |packed - per_key| in nats on the verify items
PROMPT, KEY, PAD = -1, 0, -2                              # segment ids in a packed row (keys are 0..K-1)


# ---------------------------------------------------------------- prompt

def instruction(scheme: str, state: str) -> str:
    """The fixed plain-text instruction for one record."""
    options = "\n".join(f"{k}: {d}" for k, d in L.criteria(scheme).items())
    return (f"{L.INSTRUCTIONS}\n\nOptions (key: description):\n{options}\n\nRecord (JSON): {state}\n\n"
            f"Answer with exactly one option key.")


def render(tok: Any, text: str, style: str) -> str:
    """The prompt string up to the answer cue (see the module doc)."""
    if style == "plain":
        return f"{text}\n{PLAIN_CUE}"
    if not getattr(tok, "chat_template", None):
        raise ValueError("--prompt-style chat needs a tokenizer with a chat template; use --prompt-style plain")
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def encode(tok: Any, text: str, style: str) -> list[int]:
    """Token ids of a rendered prompt (the chat template already carries the special tokens it needs)."""
    return list(tok(text, add_special_tokens=style == "plain")["input_ids"])


def key_token_ids(tok: Any, prompt: str, keys: Sequence[str], style: str) -> list[list[int]]:
    """Each key's token ids, checked to be exactly what the tokenizer produces after the prompt (every prompt
    ends in the same fixed text), so a score is the likelihood of the key string as the model would read it."""
    lead, base = (" " if style == "plain" else ""), encode(tok, prompt, style)
    ids = [list(tok(lead + k, add_special_tokens=False)["input_ids"]) for k in keys]
    for k, kid in zip(keys, ids):
        if not kid or encode(tok, prompt + lead + k, style) != base + kid:
            raise ValueError(f"key {k!r} does not tokenise the same after the prompt ({style} style)")
    return ids


# ---------------------------------------------------------------- scoring (torch)

def pair_index(keys: Sequence[Sequence[int]]) -> tuple[list[int], list[int], list[int]]:
    """(source row, target token, key) per scored token. Row 0 = the last prompt token (predicts every key's
    first token); row 1 + off_j + t - 1 = key j's token t - 1 in the packed continuation (predicts token t)."""
    src, tgt, key_of, off = [], [], [], 0
    for j, k in enumerate(keys):
        for t, tok_id in enumerate(k):
            src.append(0 if t == 0 else off + t)
            tgt.append(tok_id)
            key_of.append(j)
        off += len(k) - 1
    return src, tgt, key_of


def packed_batch(prompts: Sequence[Sequence[int]], keys: Sequence[Sequence[int]], pad_id: int) -> dict[str, Any]:
    """input_ids / position_ids / segment ids / prompt lengths of one packed forward, right-padded."""
    import torch

    cont = [t for k in keys for t in k[:-1]]
    seg_c = [j for j, k in enumerate(keys) for _ in k[:-1]]
    off_c = [t for k in keys for t in range(len(k) - 1)]
    width = max(len(p) for p in prompts) + len(cont)
    ids, pos, seg = [], [], []
    for p in prompts:
        pad = width - len(p) - len(cont)
        ids.append([*p, *cont, *[pad_id] * pad])
        pos.append([*range(len(p)), *(len(p) + t for t in off_c), *[0] * pad])
        seg.append([PROMPT] * len(p) + seg_c + [PAD] * pad)
    return {"input_ids": torch.tensor(ids), "position_ids": torch.tensor(pos), "seg": torch.tensor(seg),
            "lengths": torch.tensor([len(p) for p in prompts])}


def packed_allowed(seg: Any) -> Any:
    """[B, L, L] bool: prompt tokens causal; a key token sees the prompt and its own key's earlier tokens;
    pads see only themselves (no fully masked row, so no NaN)."""
    import torch

    idx = torch.arange(seg.shape[1], device=seg.device)
    causal, eye = idx[:, None] >= idx[None, :], idx[:, None] == idx[None, :]
    kv_prompt, same_key = (seg == PROMPT)[:, None, :], seg[:, :, None] == seg[:, None, :]
    q_prompt, q_key, q_pad = (seg == PROMPT)[:, :, None], (seg >= KEY)[:, :, None], (seg == PAD)[:, :, None]
    return (q_prompt & kv_prompt & causal) | (q_key & (kv_prompt | (same_key & causal))) | (q_pad & eye)


def additive_mask(allowed: Any, dtype: Any) -> Any:
    """[B, 1, L, L] additive float mask (0 / dtype min): what eager and SDPA attention both consume as-is."""
    import torch

    mask = torch.zeros(allowed.shape[0], 1, *allowed.shape[1:], dtype=dtype, device=allowed.device)
    return mask.masked_fill(~allowed[:, None], torch.finfo(dtype).min)


def _logp_rows(model: Any, hidden: Any, seq_idx: Any, pos_idx: Any) -> Any:
    """log-softmax over the vocabulary at the selected hidden states only (float32)."""
    import torch

    return torch.log_softmax(model.get_output_embeddings()(hidden[seq_idx, pos_idx]).float(), dim=-1)


def packed_scores(model: Any, prompts: Sequence[Sequence[int]], keys: Sequence[Sequence[int]],
                  pad_id: int) -> np.ndarray:
    """[B, K] summed key-token log-likelihoods from ONE forward over the packed rows."""
    import torch

    dev = model.device
    b = {k: v.to(dev) for k, v in packed_batch(prompts, keys, pad_id).items()}
    hidden = model.base_model(input_ids=b["input_ids"], position_ids=b["position_ids"], use_cache=False,
                              attention_mask=additive_mask(packed_allowed(b["seg"]), model.dtype)).last_hidden_state
    rows = b["lengths"][:, None] - 1 + torch.arange(1 + sum(len(k) - 1 for k in keys), device=dev)
    logp = _logp_rows(model, hidden, torch.arange(len(prompts), device=dev)[:, None], rows)     # [B, R, V]
    src, tgt, key_of = (torch.tensor(v, device=dev) for v in pair_index(keys))
    out = torch.zeros(len(prompts), len(keys), dtype=torch.float64, device=dev)
    return out.index_add_(1, key_of, logp[:, src, tgt].double()).cpu().numpy()


def per_key_scores(model: Any, prompts: Sequence[Sequence[int]], keys: Sequence[Sequence[int]],
                   pad_id: int) -> np.ndarray:
    """[B, K] the slow reference: one right-padded sequence per (item, key), the model's standard 2D mask."""
    import torch

    dev = model.device
    seqs = [[*p, *k[:-1]] for p in prompts for k in keys]
    width = max(len(s) for s in seqs)
    ids = torch.tensor([s + [pad_id] * (width - len(s)) for s in seqs], device=dev)
    att = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=dev)
    hidden = model.base_model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    pairs = [(i * len(keys) + j, len(p) - 1 + t, tok_id, i, j) for i, p in enumerate(prompts)
             for j, k in enumerate(keys) for t, tok_id in enumerate(k)]
    n, pos, tgt, item, key = (torch.tensor(v, device=dev) for v in zip(*pairs))
    vals = _logp_rows(model, hidden, n, pos)[torch.arange(len(pairs), device=dev), tgt].double()
    out = torch.zeros(len(prompts), len(keys), dtype=torch.float64, device=dev)
    return out.index_put_((item, key), vals, accumulate=True).cpu().numpy()


@dataclass(frozen=True)
class Scorer:
    model: Any
    keys: list[list[int]]
    pad_id: int
    batch_size: int          # items per forward; per_key runs batch_size // K items (K sequences each)
    method: str = "packed"

    def raw(self, prompts: Sequence[Sequence[int]], label: str = "score") -> np.ndarray:
        """[N, K] summed key log-likelihoods: length-sorted batches, results in input order."""
        import torch

        fn: Callable[..., np.ndarray] = packed_scores if self.method == "packed" else per_key_scores
        step = self.batch_size if self.method == "packed" else max(1, self.batch_size // len(self.keys))
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        out, t0, shown = np.zeros((len(prompts), len(self.keys))), time.perf_counter(), time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(order), step):
                idx = order[start:start + step]
                out[idx] = fn(self.model, [prompts[i] for i in idx], self.keys, self.pad_id)
                if time.perf_counter() - shown >= PROGRESS_SECONDS:
                    shown = time.perf_counter()
                    print(f"llm_baseline: {label} {start + len(idx)}/{len(order)} items ({shown - t0:.0f}s)",
                          flush=True)
        return out


def verify(scorer: Scorer, prompts: Sequence[Sequence[int]], half: bool) -> dict[str, Any]:
    """packed vs per_key on a few items (guards the custom 4D mask against a transformers behaviour change)."""
    import torch

    with torch.inference_mode():
        a = packed_scores(scorer.model, prompts, scorer.keys, scorer.pad_id)
        b = per_key_scores(scorer.model, prompts, scorer.keys, scorer.pad_id)
    d, tol = float(np.abs(a - b).max()), VERIFY_TOL["half" if half else "fp32"]
    return {"n": len(prompts), "max_abs_diff": d, "tol": tol, "passed": bool(d <= tol)}


# ---------------------------------------------------------------- model and subsets

def pick_dtype(device: str) -> Any:
    """bf16 on compute capability >= 8, fp16 below, fp32 on CPU (the fp16-everywhere rule is Laya's, #443)."""
    import torch

    cc_major = 0 if device == "cpu" else torch.cuda.get_device_capability()[0]
    return torch.float32 if device == "cpu" else torch.bfloat16 if cc_major >= 8 else torch.float16


def load_llm(source: str, revision: str | None, device: str) -> tuple[Any, Any]:
    """(tokenizer, eval-mode causal LM on `device`) from the Hub id at the pinned revision or a local dir."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but no CUDA device (use --device cpu for local runs)")
    kw = {"token": os.environ.get("HF_TOKEN") or None, **({"revision": revision} if revision else {})}
    tok = AutoTokenizer.from_pretrained(source, **kw)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=pick_dtype(device), attn_implementation="sdpa", **kw)
    return tok, model.to(device).eval()


def sample_order(split: str, ids: Sequence[str], seed: int) -> list[int]:
    """Row indices in sha256("seed:split:id") order: stable across machines and Python versions."""
    return sorted(range(len(ids)), key=lambda i: hashlib.sha256(f"{seed}:{split}:{ids[i]}".encode()).hexdigest())


def select_rows(split: str, rows: Sequence[dict], llm_eval: dict[str, Any]) -> tuple[list[dict], dict[str, Any]]:
    """(rows to score, in file order; subset note for the payload) per config phase4.llm_eval."""
    if split == VAL:
        n = int(llm_eval["val_for_temperature"])
        keep, rule = [r for r in rows if r.get("label") is not None][:n], f"first {n} labelled rows"
    elif split in FULL_SPLITS:
        keep, rule = list(rows), "all rows"
    else:
        n, seed = int(llm_eval["subset_per_pool"]), int(llm_eval["subset_seed"])
        chosen = set(sample_order(split, [r["id"] for r in rows], seed)[:n])
        keep, rule = [r for i, r in enumerate(rows) if i in chosen], f"sha256({seed}:{split}:id) first {n}"
    return keep, {"subset": True, "subset_rule": rule, "n_source": len(rows)}


def load_splits(args: argparse.Namespace, cfg: dict, scheme: str) -> dict[str, tuple[list[dict], dict]]:
    """{split: (selected rows, subset note)}, val first; every file is checked before the model loads."""
    splits = list(dict.fromkeys([VAL, *(args.splits or cfg["phase3"]["eval_splits"])]))
    paths, _ = split_paths(splits, args.data_dir, parse_extras(args.extra), [])
    out = {}
    for name, path in paths.items():
        rows = load_rows(path)
        check_question(rows, L.question(scheme))
        keep, note = select_rows(name, rows, cfg["phase4"]["llm_eval"])
        if not keep:
            raise ValueError(f"{name}: no rows selected ({note['subset_rule']}) from {path}")
        out[name] = (keep, {**note, "rows": str(path)})
    return out


# ---------------------------------------------------------------- run

def encode_rows(tok: Any, rows: Sequence[dict], scheme: str, style: str) -> list[list[int]]:
    return [encode(tok, render(tok, instruction(scheme, r["state"]), style), style) for r in rows]


def make_scorer(tok: Any, model: Any, args: argparse.Namespace, cfg: dict,
                val_rows: Sequence[dict]) -> tuple[Scorer, dict[str, Any]]:
    """Key tokens (boundary-checked), batch size, and the packed-vs-per_key check on the first val items."""
    import torch

    style, keys = args.prompt_style, L.option_keys(args.scheme)
    key_ids = key_token_ids(tok, render(tok, instruction(args.scheme, val_rows[0]["state"]), style), keys, style)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    bs = args.batch_size or (CPU_BATCH if args.device == "cpu" else int(cfg.get("eval", {}).get("batch_size", 64)))
    scorer, check = Scorer(model, key_ids, pad, bs, args.scoring), None
    if args.scoring == "packed" and args.verify_n > 0:
        check = verify(scorer, encode_rows(tok, val_rows[:args.verify_n], args.scheme, style),
                       half=model.dtype != torch.float32)
        print(f"llm_baseline: packed vs per_key on {check['n']} val items: max |d| {check['max_abs_diff']:.2e} nats "
              f"(tol {check['tol']:g}) {'OK' if check['passed'] else 'MISMATCH: WARNING: scoring per_key instead'}",
              flush=True)
        scorer = scorer if check["passed"] else Scorer(model, key_ids, pad, bs, "per_key")
    return scorer, {"prompt_style": style, "prompt_template": render(tok, instruction(args.scheme, "{state}"), style),
                    "scoring": scorer.method, "verify": check, "batch_size": bs,
                    "key_tokens": dict(zip(keys, key_ids))}


def score_and_write(be: Any, ctx: dict[str, Any], name: str, rows: list[dict], note: dict,
                    logp: np.ndarray | None, tau: Any) -> dict:
    """Score one split (unless already scored) and write eval/<split>.json + preds via baseline_eval."""
    t0 = time.perf_counter()
    if logp is None:
        logp = log_softmax(ctx["scorer"].raw(encode_rows(ctx["tok"], rows, ctx["scheme"], ctx["style"]), name))
    meta = {**note, "device": ctx["device"], "batch_size": ctx["scorer"].batch_size,
            "seconds": round(time.perf_counter() - t0 + note.get("seconds", 0.0), 2)}
    payload = be.write_outputs(ctx["run_dir"] / be.EVAL_DIR, ctx["run_dir"] / be.PREDS_DIR, rows=rows,
                               keys=ctx["keys"], scores=logp, T=ctx["T"], split=name, model=ctx["model"],
                               ckpt=ctx["ckpt"], evidence_fields=ctx["evidence_fields"], abstain_cfg=ctx["abstain"],
                               tau=tau, meta=meta)
    f1 = "n/a" if (payload.get("post") or {}).get("macro_f1") is None else f"{payload['post']['macro_f1']:.4f}"
    print(f"llm_baseline: {name}: {payload['n']}/{note['n_source']} rows, macro-F1 {f1} at T={ctx['T']:.3f}, "
          f"{payload['n_abstained_no_evidence']} gated ({meta['seconds']:.0f}s)", flush=True)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from . import baseline_eval as be

    cfg, t0 = load_config(args.config), time.perf_counter()
    spec = ((cfg.get("phase4") or {}).get("llms") or {}).get(args.model)
    if spec is None:
        raise ValueError(f"--model {args.model!r} is not in config phase4.llms")
    args = argparse.Namespace(**{**vars(args), "scheme": args.scheme or cfg["labels"]["scheme"]})
    splits, ckpt = load_splits(args, cfg, args.scheme), args.init or f"{spec['id']}@{spec['revision']}"
    print(f"llm_baseline: {args.model} ({ckpt}) on {', '.join(f'{s} {len(r)}' for s, (r, _) in splits.items())} "
          f"rows, scheme {args.scheme}, {args.device}", flush=True)
    tok, model = load_llm(args.init or spec["id"], None if args.init else spec["revision"], args.device)
    load_s = round(time.perf_counter() - t0, 2)
    scorer, prompt = make_scorer(tok, model, args, cfg, splits[VAL][0])
    (val_rows, val_note), t_val = splits[VAL], time.perf_counter()
    val_logp = log_softmax(scorer.raw(encode_rows(tok, val_rows, args.scheme, args.prompt_style), VAL))
    y = np.array(label_indices(val_rows, L.option_keys(args.scheme)), dtype=int)   # the val subset is labelled
    T, clamped = be.fit_T(val_logp, y)
    ctx = {"scorer": scorer, "tok": tok, "scheme": args.scheme, "style": args.prompt_style, "device": args.device,
           "run_dir": Path(args.run_dir), "keys": L.option_keys(args.scheme), "T": T, "model": args.model,
           "ckpt": ckpt, "evidence_fields": cfg["serialise"]["evidence_fields"], "abstain": cfg.get("abstain")}
    tau = score_and_write(be, ctx, VAL, val_rows, {**val_note, "seconds": time.perf_counter() - t_val}, val_logp,
                          None)["tau"]
    for name, (rows, note) in list(splits.items())[1:]:
        score_and_write(be, ctx, name, rows, note, None, tau)
    cuda = args.device == "cuda"
    extra = {"llm": args.model, "id": spec["id"], "revision": None if args.init else spec["revision"],
             "init": args.init, **prompt, "dtype": str(model.dtype).removeprefix("torch."), "device": args.device,
             "gpu": torch.cuda.get_device_name() if cuda else None,
             "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None,
             "subsets": {s: {"n": len(r), "n_source": n["n_source"], "rule": n["subset_rule"]}
                         for s, (r, n) in splits.items()},
             "load_seconds": load_s, "seconds": round(time.perf_counter() - t0, 2)}
    record = be.calibration_record(T=T, clamped=clamped, n=len(y), extra=extra)
    write_json(Path(args.run_dir) / be.CALIBRATION_FILE, record)
    return {"splits": list(splits), "T": T, "clamped": clamped, "seconds": extra["seconds"]}


# ---------------------------------------------------------------- CLI

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="laya_poc.llm_baseline", description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="config phase4.llms entry, e.g. qwen3_4b")
    p.add_argument("--scheme", choices=sorted(L.SCHEMES), default=None, help="default: config labels.scheme")
    p.add_argument("--data-dir", required=True, help="<split>.jsonl for every split not given by --extra")
    p.add_argument("--run-dir", required=True, help="writes eval/, preds/ and calibration.json here")
    p.add_argument("--extra", nargs="+", action="extend", default=None, metavar="NAME=ABS_PATH",
                   help="split files outside --data-dir, e.g. trap_candidates=<abs jsonl>")
    p.add_argument("--splits", nargs="+", default=None, help="default: config phase3.eval_splits (val always)")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--init", default=None, help="absolute local causal-LM dir (weights + tokenizer); default: Hub")
    p.add_argument("--batch-size", type=int, default=None, help="items per forward (default: eval.batch_size / 4)")
    p.add_argument("--prompt-style", choices=STYLES, default="chat")
    p.add_argument("--scoring", choices=SCORERS, default="packed")
    p.add_argument("--verify-n", type=int, default=VERIFY_N, help="val items checked packed vs per_key (0: off)")
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    if args.init and not (Path(args.init).is_absolute() and Path(args.init).is_dir()):
        p.error(f"--init must be an existing absolute directory, got {args.init!r}")
    if (args.batch_size is not None and args.batch_size < 1) or args.verify_n < 0:
        p.error("--batch-size must be >= 1 and --verify-n >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    def body() -> int:
        res = run(args)
        print(f"llm_baseline: wrote {len(res['splits'])} split reports to {Path(args.run_dir) / 'eval'} "
              f"(T={res['T']:.4f}{', CLAMPED' if res['clamped'] else ''}, {res['seconds']:.0f}s)")
        return 0

    return run_cli("llm_baseline", body, Path(args.run_dir) / "llm_baseline")


if __name__ == "__main__":
    sys.exit(main())
