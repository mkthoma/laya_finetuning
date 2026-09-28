<!-- Copied from runs/e1/gate_report.md of the Colab run on 2026-09-26 (notebooks/e1_first_run.local.ipynb, bundle 8a20f473). Metrics only: no FSQ records.
     HARDWARE DEVIATION (deliberate): run on a Colab G4 runtime = NVIDIA RTX PRO 6000 Blackwell (cc 12.0, 95 GB), chosen for speed instead of the planned free T4.
     E1 still used the T4 recipe (micro-batch 8 x accumulation 4, gradient checkpointing, fp16 autocast + GradScaler), so the
     gate criteria hold; wall-clock (0.28 h), s/micro-step and VRAM are NOT T4 figures, and the §7.7 CPU-vs-GPU parity check
     was not re-run on this card (it passed on the T4 in Phase 1). The CPU benchmark ran on the Colab host at 1-2 threads. -->

# Laya PoC: Phase 2 gate (E1)

Verdict: **PASS** (4/4 gate criteria met)

## Gate (design doc §6.2)

| # | Criterion | Value | Threshold | Result | Note |
|---|---|---|---|---|---|
| 1 | end-to-end (train -> T fit -> val eval -> CPU bench; resume) | 5/5 stages | stop_reason in ('epochs', 'early_stop', 'max_micro_steps'); export_check passed; val eval; >= 1 bench thread; >= 1 resume | PASS | training stop_reason 'epochs'; temperature fit: export_check passed=True, T 1.0939; val eval n=3000; CPU bench thread settings [1, 2]; 1 resume(s) in train/log.jsonl (forced crash at micro-step 2002, exit -9) |
| 2 | numerics (finite loss and gradients, GradScaler scale) | 0 non-finite losses, 0 non-finite grads applied, min scale 1024.0 | 0 non-finite losses; nonfinite_grad_applied == 0; min GradScaler scale >= 1.0 | PASS | 6 fp16 step(s) skipped by GradScaler (expected, not a failure) |
| 3 | beats trivial (zero-shot, majority) | 0.5761 | E1 >= zero-shot laya +10.0 pts and >= B1 majority +20.0 pts (val) | PASS | 10-class macro-F1 (Event: 18915 labelled ID places, >= labels.event_min_id_pool 300): E1 0.5761; zero-shot 0.3025 (+27.4 pts); majority 0.0187 (+55.7 pts) |
| 4 | near TF-IDF+LR | 0.5761 | E1 >= TF-IDF+LR (B3) - 5.0 pts (val) | PASS | 10-class macro-F1 (Event: 18915 labelled ID places, >= labels.event_min_id_pool 300): E1 0.5761 vs TF-IDF+LR 0.4892 (+8.7 pts) |

## E1 run (information)

| Item | Value |
|---|---|
| Headline metric | 10-class macro-F1 (Event: 18915 labelled ID places, >= labels.event_min_id_pool 300) |
| E1 val (post-T) | macro-F1 0.5761 (9-class 0.6016), acc 0.5937, ECE 0.0639; acc@80 0.6696, acc@90 0.6326; n 3000, unlabelled 0, abstained without evidence 0 |
| E1 wall-clock | 0.27 h of work over 2 processes (0.28 h from the first start to the end) |
| Seconds per micro-step; peak VRAM | 0.064 s mean (0.055 median) at MB 8; peak VRAM 8.52 GB reserved |
| Epochs / stop | epochs; 4 planned epochs; 13376 micro / 3344 opt steps; 14 evals; 0 truncated train items; padding ratio 1.01 |
| Best checkpoint | opt step 3250: val macro-F1 0.5761 (training eval) |
| Temperature; ECE pre -> post (val) | T 1.0939 (export_check T 1.0939, clamped False); ECE 0.0857 -> 0.0639 |
| TF-IDF+LR (B3) | C 8, T 0.9727; val macro-F1 0.4892 (9-class 0.5193), acc 0.5023, ECE 0.0198 |
| Zero-shot laya (val) | macro-F1 0.3025 (9-class 0.3119), acc 0.3067, ECE 0.1237 |
| Zero-shot laya-multilingual (val) | not run |
| Data fingerprint | 64eafe66ad8a5eb14743afb043bad1c814fac8ef69b344fcc1f16d88c124f3db; matches the frozen manifest docs/results/data_fingerprint_2026-09-15.json |
| Trap candidates | 700 candidates; annotation pending: data/trap_candidates.csv -> 2 annotators -> python -m laya_poc.traps merge |

## CPU benchmark (info: p95 <= 500 ms, >= 8 rec/s is judged in Phase 5)

| Threads | Cold start (s) | p50 (ms) | p95 (ms) | Batch rec/s | Peak RSS (GB) | In budget |
|---|---|---|---|---|---|---|
| 1 | 2.6 | 1526.2 | 1850.9 | 0.6 | 3.21 | no |
| 2 | 2.5 | 921.6 | 1092.3 | 1.2 | 3.21 | no |

## Input notes

- eval_zeroshot_ml_val.json: not run (optional)
