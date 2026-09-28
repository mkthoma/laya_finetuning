# GPU scoring throughput and training time (Phase 3/4 Colab archives)

Generated 2026-09-28T12:58:43+00:00 from 27 finished runs under `runs\p3_colab\runs\p3`, `runs\p4_colab\runs\p4`. Cards: G4 = NVIDIA RTX PRO 6000 Blackwell Server Edition. Metrics only: read from done.json, timing.json, eval/<split>.json, train/summary.json, train/log.jsonl and calibration.json; no FSQ row is read or written.

**One-pass throughput (rows/s)**: rows forwarded in ONE batched scoring pass of an eval split / that pass's wall-clock seconds, as the run's eval JSON recorded it: batched scoring at the eval batch size on the Colab card, end to end incl. tokenisation, batching and host-device copies (warm model, one process); not batch-1 latency. The summary takes the median over a group's runs (seeds) and its eval splits with >= 1000 scored rows (val, test_id and the OOD pools; trap_candidates is smaller and stripped_test has no scored row). **Train min**: median training minutes per run. **Total min**: done.json `seconds`: the run's whole wall clock (every orchestrator step, subprocess start included).

Which field is one pass (read in the writers' code):

- Laya (E*, B2): seconds_post: one post-T predict_batch pass over the answered rows (gated rows are not forwarded); `seconds` covers both passes and is not used. Precision: the laya package's CUDA autocast, fp16 because the Colab notebooks' setup cell sets LAYA_CUDA_AMP=fp16 (the run files do not record it).
- B4 small encoders: seconds: one score_states pass over every row (tokenise + length-sorted batches, fp16 autocast), timed before T is fitted; seconds_post/pre are 0.
- B5 LLM: seconds: one scoring pass over every row of the eval subset (tokenise + teacher-forced key scoring); on val it also includes the T fit.
- B3/B1 seconds_post: one scoring pass on the Colab host CPU (the GPU is idle); `seconds` is only the metric computation.
- Training: Laya: timing.json `train` (train_single subprocess: imports, checkpoint load, training, in-training val evals, best save); B4: train/summary.json `seconds` (training loop incl. per-epoch val scoring, no model load); B3: TF-IDF fit + C grid on CPU (calibration.json); B1, B2, B5 train nothing.

† = derived with a caveat, see the notes column.

## Summary per group

| Group | Card | Device | Precision | Batch | Runs | One-pass rows/s (median) | Range | Points | Train min (median/run) | Total min (median/run) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 majority c10 | G4 | cpu | cpu (sklearn / numpy) | n/a | 1 | n/a † | n/a to n/a | 0 | n/a | 0.0 | scored on the Colab host CPU (the GPU is idle) |
| B1 majority c7 | G4 | cpu | cpu (sklearn / numpy) | n/a | 1 | n/a † | n/a to n/a | 0 | n/a | 0.0 | scored on the Colab host CPU (the GPU is idle) |
| B1 prior c10 | G4 | cpu | cpu (sklearn / numpy) | n/a | 1 | n/a † | n/a to n/a | 0 | n/a | 0.0 | scored on the Colab host CPU (the GPU is idle) |
| B1 prior c7 | G4 | cpu | cpu (sklearn / numpy) | n/a | 1 | n/a † | n/a to n/a | 0 | n/a | 0.0 | scored on the Colab host CPU (the GPU is idle) |
| B2 laya c10 | G4 | cuda | fp16 autocast (LAYA_CUDA_AMP, notebook) | 64 | 1 | 667 | 595 to 677 | 5 | n/a | 0.9 |  |
| B2 laya_ml c10 | G4 | cuda | fp16 autocast (LAYA_CUDA_AMP, notebook) | 64 | 1 | 1,293 | 1,156 to 1,399 | 5 | n/a | 0.5 |  |
| B3 tfidf_lr c10 | G4 | cpu | cpu (sklearn / numpy) | n/a | 1 | 6,977 † | 4,651 to 11,111 | 5 | 0.4 | 0.5 | scored on the Colab host CPU (the GPU is idle); one-pass times under 1 s (recorded to 0.01 s): rates carry about 1-2 % rounding |
| B3 tfidf_lr c7 | G4 | cpu | cpu (sklearn / numpy) | n/a | 1 | 6,977 † | 4,651 to 11,765 | 5 | 0.4 | 0.5 | scored on the Colab host CPU (the GPU is idle); one-pass times under 1 s (recorded to 0.01 s): rates carry about 1-2 % rounding |
| B4 mmbert_small c10 | G4 | cuda | fp16 autocast | 64 | 3 | 6,244 † | 5,128 to 6,897 | 15 | 1.3 | 1.5 | one-pass times under 1 s (recorded to 0.01 s): rates carry about 1-2 % rounding |
| B4 modernbert_base c10 | G4 | cuda | fp16 autocast | 64 | 3 | 4,839 † | 3,704 to 5,263 | 15 | 1.4 | 1.5 | one-pass times under 1 s (recorded to 0.01 s): rates carry about 1-2 % rounding |
| B5 qwen3_4b c10 | G4 | cuda | bfloat16 | 64 | 1 | 12.7 † | 11.2 to 13.9 | 4 | n/a | 13.7 | per_key scoring (the packed check failed): 6 items (60 sequences) per forward at batch_size 64 |
| E2 laya c10 | G4 | cuda | fp16 autocast (LAYA_CUDA_AMP, notebook) | 64 | 3 | 671 | 595 to 682 | 15 | 8.6 | 9.7 |  |
| E3 laya_ml c10 | G4 | cuda | fp16 autocast (LAYA_CUDA_AMP, notebook) | 64 | 3 | 1,288 | 1,156 to 1,399 | 15 | 4.5 | 5.2 |  |
| E4 laya c10 head | G4 | cuda | fp16 autocast (LAYA_CUDA_AMP, notebook) | 64 | 1 | 670 | 595 to 680 | 5 | 2.2 | 3.3 |  |
| E5 laya c7 | G4 | cuda | fp16 autocast (LAYA_CUDA_AMP, notebook) | 64 | 1 | 872 | 752 to 884 | 5 | 7.0 | 7.8 |  |
| E5 laya_ml c7 | G4 | cuda | fp16 autocast (LAYA_CUDA_AMP, notebook) | 64 | 1 | 1,630 | 1,418 to 1,786 | 5 | 2.9 | 3.5 |  |
| E6 laya c10 n1000 | G4 | cuda | fp16 autocast (LAYA_CUDA_AMP, notebook) | 64 | 1 | 671 | 595 to 682 | 5 | 1.1 | 2.2 |  |
| E6 laya c10 n10000 | G4 | cuda | fp16 autocast (LAYA_CUDA_AMP, notebook) | 64 | 1 | 671 | 595 to 682 | 5 | 3.9 | 5.0 |  |
| E6 laya c10 n3000 | G4 | cuda | fp16 autocast (LAYA_CUDA_AMP, notebook) | 64 | 1 | 671 | 595 to 682 | 5 | 1.6 | 2.6 |  |

## Scoring throughput per group and split

| Group | Split | Rows forwarded | Scored | Runs | One-pass s (median) | Rows/s (median) | Batch | Device | Card | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 majority c10 | ood_brand | 2000 | 2000 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 majority c10 | ood_country | 1998 | 1998 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 majority c10 | ood_script | 2000 | 2000 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 majority c10 | stripped_test | 1000 | 0 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 majority c10 | test_id | 3000 | 3000 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 majority c10 | trap_candidates | 700 | 700 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 majority c10 | val | 3000 | 3000 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 majority c7 | ood_brand | 2000 | 2000 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 majority c7 | ood_country | 1998 | 1998 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 majority c7 | ood_script | 2000 | 2000 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 majority c7 | stripped_test | 1000 | 0 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 majority c7 | test_id | 3000 | 3000 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 majority c7 | trap_candidates | 700 | 700 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 majority c7 | val | 3000 | 3000 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 prior c10 | ood_brand | 2000 | 2000 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 prior c10 | ood_country | 1998 | 1998 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 prior c10 | ood_script | 2000 | 2000 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 prior c10 | stripped_test | 1000 | 0 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 prior c10 | test_id | 3000 | 3000 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 prior c10 | trap_candidates | 700 | 700 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 prior c10 | val | 3000 | 3000 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 prior c7 | ood_brand | 2000 | 2000 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 prior c7 | ood_country | 1998 | 1998 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 prior c7 | ood_script | 2000 | 2000 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 prior c7 | stripped_test | 1000 | 0 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 prior c7 | test_id | 3000 | 3000 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 prior c7 | trap_candidates | 700 | 700 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B1 prior c7 | val | 3000 | 3000 | 1 | 0.00 | n/a † | n/a | cpu | G4 | pass below the 0.01 s timer resolution: no rate |
| B2 laya c10 | ood_brand | 2000 | 2000 | 1 | 3.36 | 595 | 64 | cuda | G4 |  |
| B2 laya c10 | ood_country | 1998 | 1998 | 1 | 2.95 | 677 | 64 | cuda | G4 |  |
| B2 laya c10 | ood_script | 2000 | 2000 | 1 | 2.96 | 676 | 64 | cuda | G4 |  |
| B2 laya c10 | stripped_test | 0 | 0 | 1 | 0.00 | n/a † | 64 | cuda | G4 | no row forwarded (the gate abstains on every row) |
| B2 laya c10 | test_id | 3000 | 3000 | 1 | 4.50 | 667 | 64 | cuda | G4 |  |
| B2 laya c10 | trap_candidates | 700 | 700 | 1 | 1.05 | 667 | 64 | cuda | G4 |  |
| B2 laya c10 | val | 3000 | 3000 | 1 | 4.78 | 628 | 64 | cuda | G4 |  |
| B2 laya_ml c10 | ood_brand | 2000 | 2000 | 1 | 1.73 | 1,156 | 64 | cuda | G4 |  |
| B2 laya_ml c10 | ood_country | 1998 | 1998 | 1 | 1.52 | 1,314 | 64 | cuda | G4 |  |
| B2 laya_ml c10 | ood_script | 2000 | 2000 | 1 | 1.43 | 1,399 | 64 | cuda | G4 |  |
| B2 laya_ml c10 | stripped_test | 0 | 0 | 1 | 0.00 | n/a † | 64 | cuda | G4 | no row forwarded (the gate abstains on every row) |
| B2 laya_ml c10 | test_id | 3000 | 3000 | 1 | 2.32 | 1,293 | 64 | cuda | G4 |  |
| B2 laya_ml c10 | trap_candidates | 700 | 700 | 1 | 0.55 | 1,273 | 64 | cuda | G4 |  |
| B2 laya_ml c10 | val | 3000 | 3000 | 1 | 2.57 | 1,167 | 64 | cuda | G4 |  |
| B3 tfidf_lr c10 | ood_brand | 2000 | 2000 | 1 | 0.43 | 4,651 | n/a | cpu | G4 |  |
| B3 tfidf_lr c10 | ood_country | 1998 | 1998 | 1 | 0.27 | 7,400 | n/a | cpu | G4 |  |
| B3 tfidf_lr c10 | ood_script | 2000 | 2000 | 1 | 0.18 | 11,111 | n/a | cpu | G4 |  |
| B3 tfidf_lr c10 | stripped_test | 1000 | 0 | 1 | 0.02 | 50,000 | n/a | cpu | G4 |  |
| B3 tfidf_lr c10 | test_id | 3000 | 3000 | 1 | 0.45 | 6,667 | n/a | cpu | G4 |  |
| B3 tfidf_lr c10 | trap_candidates | 700 | 700 | 1 | 0.11 | 6,364 | n/a | cpu | G4 |  |
| B3 tfidf_lr c10 | val | 3000 | 3000 | 1 | 0.43 | 6,977 | n/a | cpu | G4 |  |
| B3 tfidf_lr c7 | ood_brand | 2000 | 2000 | 1 | 0.43 | 4,651 | n/a | cpu | G4 |  |
| B3 tfidf_lr c7 | ood_country | 1998 | 1998 | 1 | 0.27 | 7,400 | n/a | cpu | G4 |  |
| B3 tfidf_lr c7 | ood_script | 2000 | 2000 | 1 | 0.17 | 11,765 | n/a | cpu | G4 |  |
| B3 tfidf_lr c7 | stripped_test | 1000 | 0 | 1 | 0.02 | 50,000 | n/a | cpu | G4 |  |
| B3 tfidf_lr c7 | test_id | 3000 | 3000 | 1 | 0.44 | 6,818 | n/a | cpu | G4 |  |
| B3 tfidf_lr c7 | trap_candidates | 700 | 700 | 1 | 0.11 | 6,364 | n/a | cpu | G4 |  |
| B3 tfidf_lr c7 | val | 3000 | 3000 | 1 | 0.43 | 6,977 | n/a | cpu | G4 |  |
| B4 mmbert_small c10 | ood_brand | 2000 | 2000 | 3 | 0.39 | 5,128 | 64 | cuda | G4 |  |
| B4 mmbert_small c10 | ood_country | 1998 | 1998 | 3 | 0.32 | 6,244 | 64 | cuda | G4 |  |
| B4 mmbert_small c10 | ood_script | 2000 | 2000 | 3 | 0.29 | 6,897 | 64 | cuda | G4 |  |
| B4 mmbert_small c10 | stripped_test | 1000 | 0 | 3 | 0.11 | 9,091 | 64 | cuda | G4 |  |
| B4 mmbert_small c10 | test_id | 3000 | 3000 | 3 | 0.48 | 6,250 | 64 | cuda | G4 |  |
| B4 mmbert_small c10 | trap_candidates | 700 | 700 | 3 | 0.12 | 5,833 | 64 | cuda | G4 |  |
| B4 mmbert_small c10 | val | 3000 | 3000 | 3 | 0.49 | 6,122 | 64 | cuda | G4 |  |
| B4 modernbert_base c10 | ood_brand | 2000 | 2000 | 3 | 0.54 | 3,704 | 64 | cuda | G4 |  |
| B4 modernbert_base c10 | ood_country | 1998 | 1998 | 3 | 0.40 | 4,995 | 64 | cuda | G4 |  |
| B4 modernbert_base c10 | ood_script | 2000 | 2000 | 3 | 0.38 | 5,263 | 64 | cuda | G4 |  |
| B4 modernbert_base c10 | stripped_test | 1000 | 0 | 3 | 0.10 | 10,000 | 64 | cuda | G4 |  |
| B4 modernbert_base c10 | test_id | 3000 | 3000 | 3 | 0.62 | 4,839 | 64 | cuda | G4 |  |
| B4 modernbert_base c10 | trap_candidates | 700 | 700 | 3 | 0.15 | 4,667 | 64 | cuda | G4 |  |
| B4 modernbert_base c10 | val | 3000 | 3000 | 3 | 0.62 | 4,839 | 64 | cuda | G4 |  |
| B5 qwen3_4b c10 | ood_brand | 2000 | 2000 | 1 | 179.27 | 11.2 | 64 | cuda | G4 |  |
| B5 qwen3_4b c10 | ood_country | 1998 | 1998 | 1 | 155.89 | 12.8 | 64 | cuda | G4 |  |
| B5 qwen3_4b c10 | ood_script | 2000 | 2000 | 1 | 143.83 | 13.9 | 64 | cuda | G4 |  |
| B5 qwen3_4b c10 | stripped_test | 1000 | 0 | 1 | 60.74 | 16.5 | 64 | cuda | G4 |  |
| B5 qwen3_4b c10 | test_id | 2000 | 2000 | 1 | 159.23 | 12.6 | 64 | cuda | G4 |  |
| B5 qwen3_4b c10 | trap_candidates | 700 | 700 | 1 | 54.61 | 12.8 | 64 | cuda | G4 |  |
| B5 qwen3_4b c10 | val | 500 | 500 | 1 | 39.65 | 12.6 † | 64 | cuda | G4 | one-pass seconds include the temperature fit on this split |
| E2 laya c10 | ood_brand | 2000 | 2000 | 3 | 3.36 | 595 | 64 | cuda | G4 |  |
| E2 laya c10 | ood_country | 1998 | 1998 | 3 | 2.93 | 682 | 64 | cuda | G4 |  |
| E2 laya c10 | ood_script | 2000 | 2000 | 3 | 2.95 | 678 | 64 | cuda | G4 |  |
| E2 laya c10 | stripped_test | 0 | 0 | 3 | 0.00 | n/a † | 64 | cuda | G4 | no row forwarded (the gate abstains on every row) |
| E2 laya c10 | test_id | 3000 | 3000 | 3 | 4.47 | 671 | 64 | cuda | G4 |  |
| E2 laya c10 | trap_candidates | 700 | 700 | 3 | 1.04 | 673 | 64 | cuda | G4 |  |
| E2 laya c10 | val | 3000 | 3000 | 3 | 4.74 | 633 | 64 | cuda | G4 |  |
| E3 laya_ml c10 | ood_brand | 2000 | 2000 | 3 | 1.73 | 1,156 | 64 | cuda | G4 |  |
| E3 laya_ml c10 | ood_country | 1998 | 1998 | 3 | 1.53 | 1,306 | 64 | cuda | G4 |  |
| E3 laya_ml c10 | ood_script | 2000 | 2000 | 3 | 1.44 | 1,389 | 64 | cuda | G4 |  |
| E3 laya_ml c10 | stripped_test | 0 | 0 | 3 | 0.00 | n/a † | 64 | cuda | G4 | no row forwarded (the gate abstains on every row) |
| E3 laya_ml c10 | test_id | 3000 | 3000 | 3 | 2.33 | 1,288 | 64 | cuda | G4 |  |
| E3 laya_ml c10 | trap_candidates | 700 | 700 | 3 | 0.55 | 1,273 | 64 | cuda | G4 |  |
| E3 laya_ml c10 | val | 3000 | 3000 | 3 | 2.59 | 1,158 | 64 | cuda | G4 |  |
| E4 laya c10 head | ood_brand | 2000 | 2000 | 1 | 3.36 | 595 | 64 | cuda | G4 |  |
| E4 laya c10 head | ood_country | 1998 | 1998 | 1 | 2.94 | 680 | 64 | cuda | G4 |  |
| E4 laya c10 head | ood_script | 2000 | 2000 | 1 | 2.95 | 678 | 64 | cuda | G4 |  |
| E4 laya c10 head | stripped_test | 0 | 0 | 1 | 0.00 | n/a † | 64 | cuda | G4 | no row forwarded (the gate abstains on every row) |
| E4 laya c10 head | test_id | 3000 | 3000 | 1 | 4.48 | 670 | 64 | cuda | G4 |  |
| E4 laya c10 head | trap_candidates | 700 | 700 | 1 | 1.04 | 673 | 64 | cuda | G4 |  |
| E4 laya c10 head | val | 3000 | 3000 | 1 | 4.74 | 633 | 64 | cuda | G4 |  |
| E5 laya c7 | ood_brand | 2000 | 2000 | 1 | 2.66 | 752 | 64 | cuda | G4 |  |
| E5 laya c7 | ood_country | 1998 | 1998 | 1 | 2.26 | 884 | 64 | cuda | G4 |  |
| E5 laya c7 | ood_script | 2000 | 2000 | 1 | 2.28 | 877 | 64 | cuda | G4 |  |
| E5 laya c7 | stripped_test | 0 | 0 | 1 | 0.00 | n/a † | 64 | cuda | G4 | no row forwarded (the gate abstains on every row) |
| E5 laya c7 | test_id | 3000 | 3000 | 1 | 3.44 | 872 | 64 | cuda | G4 |  |
| E5 laya c7 | trap_candidates | 700 | 700 | 1 | 0.81 | 864 | 64 | cuda | G4 |  |
| E5 laya c7 | val | 3000 | 3000 | 1 | 3.71 | 809 | 64 | cuda | G4 |  |
| E5 laya_ml c7 | ood_brand | 2000 | 2000 | 1 | 1.41 | 1,418 | 64 | cuda | G4 |  |
| E5 laya_ml c7 | ood_country | 1998 | 1998 | 1 | 1.20 | 1,665 | 64 | cuda | G4 |  |
| E5 laya_ml c7 | ood_script | 2000 | 2000 | 1 | 1.12 | 1,786 | 64 | cuda | G4 |  |
| E5 laya_ml c7 | stripped_test | 0 | 0 | 1 | 0.00 | n/a † | 64 | cuda | G4 | no row forwarded (the gate abstains on every row) |
| E5 laya_ml c7 | test_id | 3000 | 3000 | 1 | 1.84 | 1,630 | 64 | cuda | G4 |  |
| E5 laya_ml c7 | trap_candidates | 700 | 700 | 1 | 0.43 | 1,628 | 64 | cuda | G4 |  |
| E5 laya_ml c7 | val | 3000 | 3000 | 1 | 2.10 | 1,429 | 64 | cuda | G4 |  |
| E6 laya c10 n1000 | ood_brand | 2000 | 2000 | 1 | 3.36 | 595 | 64 | cuda | G4 |  |
| E6 laya c10 n1000 | ood_country | 1998 | 1998 | 1 | 2.93 | 682 | 64 | cuda | G4 |  |
| E6 laya c10 n1000 | ood_script | 2000 | 2000 | 1 | 2.95 | 678 | 64 | cuda | G4 |  |
| E6 laya c10 n1000 | stripped_test | 0 | 0 | 1 | 0.00 | n/a † | 64 | cuda | G4 | no row forwarded (the gate abstains on every row) |
| E6 laya c10 n1000 | test_id | 3000 | 3000 | 1 | 4.47 | 671 | 64 | cuda | G4 |  |
| E6 laya c10 n1000 | trap_candidates | 700 | 700 | 1 | 1.04 | 673 | 64 | cuda | G4 |  |
| E6 laya c10 n1000 | val | 3000 | 3000 | 1 | 4.74 | 633 | 64 | cuda | G4 |  |
| E6 laya c10 n10000 | ood_brand | 2000 | 2000 | 1 | 3.36 | 595 | 64 | cuda | G4 |  |
| E6 laya c10 n10000 | ood_country | 1998 | 1998 | 1 | 2.93 | 682 | 64 | cuda | G4 |  |
| E6 laya c10 n10000 | ood_script | 2000 | 2000 | 1 | 2.95 | 678 | 64 | cuda | G4 |  |
| E6 laya c10 n10000 | stripped_test | 0 | 0 | 1 | 0.00 | n/a † | 64 | cuda | G4 | no row forwarded (the gate abstains on every row) |
| E6 laya c10 n10000 | test_id | 3000 | 3000 | 1 | 4.47 | 671 | 64 | cuda | G4 |  |
| E6 laya c10 n10000 | trap_candidates | 700 | 700 | 1 | 1.04 | 673 | 64 | cuda | G4 |  |
| E6 laya c10 n10000 | val | 3000 | 3000 | 1 | 4.74 | 633 | 64 | cuda | G4 |  |
| E6 laya c10 n3000 | ood_brand | 2000 | 2000 | 1 | 3.36 | 595 | 64 | cuda | G4 |  |
| E6 laya c10 n3000 | ood_country | 1998 | 1998 | 1 | 2.93 | 682 | 64 | cuda | G4 |  |
| E6 laya c10 n3000 | ood_script | 2000 | 2000 | 1 | 2.95 | 678 | 64 | cuda | G4 |  |
| E6 laya c10 n3000 | stripped_test | 0 | 0 | 1 | 0.00 | n/a † | 64 | cuda | G4 | no row forwarded (the gate abstains on every row) |
| E6 laya c10 n3000 | test_id | 3000 | 3000 | 1 | 4.47 | 671 | 64 | cuda | G4 |  |
| E6 laya c10 n3000 | trap_candidates | 700 | 700 | 1 | 1.04 | 673 | 64 | cuda | G4 |  |
| E6 laya c10 n3000 | val | 3000 | 3000 | 1 | 4.75 | 632 | 64 | cuda | G4 |  |

## Per run

Eval step s: the Laya evaluate subprocess (both passes on every split plus order invariance, model load included); the baselines run one process, so only its total is recorded.

| Run | Group | Seed | Card | Device | Train precision | Train s | Scoring s (one pass, all splits) | Eval step s | Export check s | Load s | Total s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fsq-c10-B1-majority | B1 majority c10 | n/a | G4 | cpu | n/a | n/a | 0.00 | n/a | n/a | n/a | 1.2 |
| fsq-c7-B1-majority | B1 majority c7 | n/a | G4 | cpu | n/a | n/a | 0.00 | n/a | n/a | n/a | 1.1 |
| fsq-c10-B1-prior | B1 prior c10 | n/a | G4 | cpu | n/a | n/a | 0.00 | n/a | n/a | n/a | 1.2 |
| fsq-c7-B1-prior | B1 prior c7 | n/a | G4 | cpu | n/a | n/a | 0.00 | n/a | n/a | n/a | 1.1 |
| fsq-c10-B2-laya-zs | B2 laya c10 | n/a | G4 | cuda | n/a | n/a | 19.60 | 54.0 | n/a | n/a | 54.0 |
| fsq-c10-B2-laya_ml-zs | B2 laya_ml c10 | n/a | G4 | cuda | n/a | n/a | 10.12 | 31.8 | n/a | n/a | 31.8 |
| fsq-c10-B3-tfidf_lr | B3 tfidf_lr c10 | n/a | G4 | cpu | n/a | 24.9 | 1.89 | n/a | n/a | n/a | 29.2 |
| fsq-c7-B3-tfidf_lr | B3 tfidf_lr c7 | n/a | G4 | cpu | n/a | 25.9 | 1.87 | n/a | n/a | n/a | 30.1 |
| fsq-c10-B4-mmbert_small-s11 | B4 mmbert_small c10 | 11 | G4 | cuda | fp16 AMP | 79.8 | 2.17 | n/a | n/a | n/a | 92.7 |
| fsq-c10-B4-mmbert_small-s22 | B4 mmbert_small c10 | 22 | G4 | cuda | fp16 AMP | 78.4 | 2.21 | n/a | n/a | n/a | 87.8 |
| fsq-c10-B4-mmbert_small-s33 | B4 mmbert_small c10 | 33 | G4 | cuda | fp16 AMP | 78.9 | 2.20 | n/a | n/a | n/a | 88.2 |
| fsq-c10-B4-modernbert_base-s11 | B4 modernbert_base c10 | 11 | G4 | cuda | fp16 AMP | 83.2 | 2.81 | n/a | n/a | n/a | 95.6 |
| fsq-c10-B4-modernbert_base-s22 | B4 modernbert_base c10 | 22 | G4 | cuda | fp16 AMP | 83.2 | 2.81 | n/a | n/a | n/a | 92.3 |
| fsq-c10-B4-modernbert_base-s33 | B4 modernbert_base c10 | 33 | G4 | cuda | fp16 AMP | 80.9 | 2.82 | n/a | n/a | n/a | 89.9 |
| fsq-c10-B5-qwen3_4b | B5 qwen3_4b c10 | n/a | G4 | cuda | n/a | n/a | 793.22 | n/a | n/a | 21.7 | 819.9 |
| fsq-c10-E2-laya-s11 | E2 laya c10 | 11 | G4 | cuda | fp16 AMP | 517.3 | 19.51 | 53.7 | 10.0 | n/a | 581.0 |
| fsq-c10-E2-laya-s22 | E2 laya c10 | 22 | G4 | cuda | fp16 AMP | 515.2 | 19.45 | 53.6 | 10.1 | n/a | 578.9 |
| fsq-c10-E2-laya-s33 | E2 laya c10 | 33 | G4 | cuda | fp16 AMP | 430.2 | 19.49 | 53.6 | 10.1 | n/a | 493.9 |
| fsq-c10-E3-laya_ml-s11 | E3 laya_ml c10 | 11 | G4 | cuda | fp16 AMP | 222.6 | 10.17 | 31.9 | 9.4 | n/a | 264.0 |
| fsq-c10-E3-laya_ml-s22 | E3 laya_ml c10 | 22 | G4 | cuda | fp16 AMP | 267.9 | 10.13 | 31.9 | 9.4 | n/a | 309.1 |
| fsq-c10-E3-laya_ml-s33 | E3 laya_ml c10 | 33 | G4 | cuda | fp16 AMP | 286.4 | 10.17 | 31.9 | 9.4 | n/a | 327.7 |
| fsq-c10-E4-laya-s11-head | E4 laya c10 head | 11 | G4 | cuda | fp16 AMP | 131.8 | 19.51 | 53.8 | 10.1 | n/a | 195.6 |
| fsq-c7-E5-laya-s11 | E5 laya c7 | 11 | G4 | cuda | fp16 AMP | 417.3 | 15.16 | 42.6 | 9.3 | n/a | 469.2 |
| fsq-c7-E5-laya_ml-s11 | E5 laya_ml c7 | 11 | G4 | cuda | fp16 AMP | 176.7 | 8.10 | 26.6 | 9.0 | n/a | 212.3 |
| fsq-c10-E6-laya-s11-n1000 | E6 laya c10 n1000 | 11 | G4 | cuda | fp16 AMP | 66.1 | 19.49 | 53.7 | 10.1 | n/a | 129.8 |
| fsq-c10-E6-laya-s11-n10000 | E6 laya c10 n10000 | 11 | G4 | cuda | fp16 AMP | 235.7 | 19.49 | 53.7 | 10.1 | n/a | 299.4 |
| fsq-c10-E6-laya-s11-n3000 | E6 laya c10 n3000 | 11 | G4 | cuda | fp16 AMP | 94.5 | 19.50 | 53.7 | 10.1 | n/a | 158.4 |
