<!-- Copied from the Step 12 output of the Colab T4 run on 2026-09-26 (notebooks/smoke_test.local.ipynb, bundle c1d8e555). Metrics only: no FSQ records. -->

# Laya PoC: Phase 1 smoke test report

Verdict: **PASS** (8/8 exit criteria met)

## Smoke test (design doc §7.13)

| Item | Value |
|---|---|
| GPU / CC / driver | Tesla T4 / 7.5 / 580.82.07 |
| Laya commit / version; transformers; torch | 4066d5d5fbf0 / 0.3.20; 5.16.1; 2.11.0+cu128 (CUDA 12.8) |
| Parity `laya`: max \|Δp\|, argmax agreement | max\|dp\| 0.0062, agree 1.000, padded 0.0019 (200 rows, test cuda float16) |
| Parity `laya-multilingual` | max\|dp\| 0.0067, agree 1.000, padded 0.0025 (200 rows, test cuda float16) |
| Zero-shot on 20 records: correct / 20 (each checkpoint) | laya 3/20, laya_ml 2/20 |
| Loss at step 0 / 120 / 250; NaN? | mean loss_ce micro 1..10: 2.2731 / 111..130: 2.0014 / 241..250: 1.7410; val_ce 2.0452 (initial) -> 1.7155 (final); NaN/inf: none |
| Resume: step resumed from; Δloss vs control | micro-step 120; relative Δ 0.0018 (micro 121..140: resumed 1.9409 vs control 1.9443; lr_enc, lr_head, scale match at 33 opt steps) |
| Peak VRAM (GB); s/step at MB 8 | 8.50 GB (10^9 B) reserved (8.34 allocated); 0.431 s/micro-step (mean) at MB 8 |
| Extrapolated E1 wall-clock (h) = s/step × steps | 1.94 h = train 1.60 h (0.431 s mean × 13376 micro-steps (ceil(25000 × 1.07 / 8) × 4 epochs)) + eval 0.19 h (14 evals × 3000 rows at 16.7 ms/row, from the control final eval) + checkpoints 0.14 h (7 checkpoints × 71.9 s, one per 15 min) |
| Optional typed-decisions 1-epoch accuracy | not run |

## Exit checklist (design doc §6.2 Phase 1, restated per review §D.2)

| # | Criterion | Value | Threshold | Result | Note |
|---|---|---|---|---|---|
| 1 | parity laya | max\|dp\| 0.0062, agree 1.000, padded 0.0019 | max\|dp\| <= 0.02, agree >= 0.995, no NaN, padded max\|dp\| <= 0.02 | PASS | 200 rows, test cuda float16 |
| 2 | parity laya_ml | max\|dp\| 0.0067, agree 1.000, padded 0.0025 | max\|dp\| <= 0.02, agree >= 0.995, no NaN, padded max\|dp\| <= 0.02 | PASS | 200 rows, test cuda float16 |
| 3 | crash run exited non-zero | -9 | returncode != 0 (-9 or 137 expected) after crash_injected at micro-step 122 | PASS |  |
| 4 | resumed from last checkpoint | 120 | == 120 (kill at 122, ckpt every 40) | PASS | from /content/laya_poc/runs/smoke/resumed/ckpt/step0000030.pt |
| 5 | resume vs control (loss_ce, lr, scaler scale) | 0.0018 | \|mean resumed - mean control\| / mean control <= 0.02; lr_enc, lr_head (rel 1e-12) and GradScaler scale equal at every opt step after the resume point | PASS | micro 121..140: resumed 1.9409 vs control 1.9443; lr_enc, lr_head, scale match at 33 opt steps |
| 6 | val_ce drop (control, initial -> final eval) | 0.1612 | (initial val_ce - final val_ce) / initial val_ce >= 0.1 (control, held-out val) | PASS | val_ce 2.0452 (opt 0, n=200) -> 1.7155 (opt 63, n=200); doc §6.2 target 20% (info): not met; info (not graded): train mean loss_ce micro 1..30 2.2006 -> 221..250 1.7016 |
| 7 | numerics (finite loss, scaler scale >= 1) | 0 non-finite, min scale 8192.0 | 0 non-finite losses; min GradScaler scale >= 1 | PASS |  |
| 8 | peak VRAM reserved | 8.5020 | <= 14.0 GB (10^9 bytes), torch.cuda.max_memory_reserved | PASS | control 8.502, resumed 8.481 |

## Additional information

| Item | Value |
|---|---|
| Description tokens | C10 option descriptions tokenise to up to 15 tokens (the doc assumed <= 10); the ten options total 134 tokens incl. [MASK], under head_max_len - 16, so Laya never trims them |
| Export / calibration round trip | PASS: T=0.8388; round-trip max\|dp\| 0.00013; ECE 0.0691 -> 0.0916; training eval ok; budgets ok |
| Smoke data (split sizes) | {"train": 2000, "val": 200, "test_id": 200, "ood_country": 0, "ood_script": 0, "ood_brand": 0} |
| Environment warnings | none |
