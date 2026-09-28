# Laya PoC: Phase 5 evaluation and decision

> **Run notes (2026-09-28).** Accuracy, calibration and robustness come from the Phase 3 (G4) and Phase 4 (G4)
> archives, recomputed by `laya_poc.phase5_metrics` (the macro-F1 matches the eval JSONs on all 162 run-splits).
> The judged CPU benchmark ran on the maintainer's laptop, which is **not** the doc's reference 4-vCPU cloud host.
> The machine was an i5-1135G7 with 4 physical cores and 8 threads, 16 GB RAM, Windows 11, on a quiet machine. The run used
> `--thread-cap logical`, so the 8-thread rows use hyperthreads. The row counts were reduced: latency n=200 and batch
> n=500 (config: 500 and 1000). The ONNX acceptance check used the full 1000 rows. mmBERT-small was re-run on its own
> after a download-filter bug (its pinned revision ships only `pytorch_model.bin`). Its two rows were merged into the same bench file, with the
> same machine and plan. The 4 physical cores are likely at least as fast as 4 cloud vCPUs, so the C5 throughput misses
> (laya_ml 3.2 rec/s against a floor of 8) are not expected to reverse on the reference host. To judge C5 on doc-spec hardware, run
> `notebooks/phase5_cpu_bench.ipynb` on a Colab CPU runtime and pass its JSON as the first `--bench`.
> Traps are pending annotation (`docs/trap_annotation_guide.md`); the report shows that they cannot change the verdict.

Generated 2026-09-28T11:40:46+00:00 from the phase5_metrics output of 2026-09-28T09:23:02+00:00 (recomputed macro-F1 matches the eval JSONs: yes). Judged benchmark machine: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4C/8T, 15.69 GB RAM). Trap set: pending. Post-T metrics unless marked pre; multi-seed groups show `mean (min to max)` over seeds. Metrics only: no FSQ rows.

Verdict: **INVESTIGATE** (final). Final: the pending inputs (traps) cannot change it. PASS is out of reach whatever the pending inputs show: every candidate already fails (E2 laya c10: C2, C3, C5; E3 laya_ml c10: C1, C2, C3, C5).

## Main results (design §7.13)

Mean ± range is shown as `mean (min to max)` over seeds. Gap vs ID = macro-F1(test_id) - macro-F1(pool) per run, then averaged; `(excluded)`: the pool is not judged for that model (`laya` cannot read Thai). B5 scores evaluation subsets. E6 train subsets are in the Phase 3/4 report.

### test_id

| Arm | Macro-F1 (mean ± range) | Acc | ECE pre → post | Brier | NLL | Acc@80 | Acc@90 | Gap vs ID |
|---|---|---|---|---|---|---|---|---|
| Majority (B1) | 0.0205 | 0.1143 | 0.8857 → 0.8857 | 1.7713 | 24.4719 | 0.1154 | 0.1163 | - |
| Prior (B1) | 0.0205 | 0.1143 | 0.0133 → 0.0133 | 0.8999 | 2.3022 | 0.1154 | 0.1163 | - |
| Zero-shot `laya` (B2) | 0.3126 | 0.3147 | 0.1212 → 0.1212 | 0.8247 | 2.0865 | 0.3550 | 0.3363 | - |
| Zero-shot `laya-ml` (B2) | 0.2823 | 0.2847 | 0.3449 → 0.3449 | 0.9836 | 2.6766 | 0.3175 | 0.3011 | - |
| TF-IDF+LR (B3) | 0.4866 | 0.4967 | 0.0214 → 0.0223 | 0.6412 | 1.4983 | 0.5596 | 0.5274 | - |
| ModernBERT-base (B4), 3 seeds | 0.5598 (0.5570 to 0.5627) | 0.5748 | 0.0278 → 0.0278 | 0.5580 | 1.3239 | 0.6553 | 0.6141 | - |
| mmBERT-small (B4), 3 seeds | 0.5959 (0.5858 to 0.6047) | 0.6152 | 0.0346 → 0.0264 | 0.5090 | 1.1875 | 0.7019 | 0.6590 | - |
| Head-only `laya` (E4) | 0.3177 | 0.3197 | 0.0395 → 0.0283 | 0.7995 | 1.9834 | 0.3613 | 0.3419 | - |
| FT `laya` (E2), 3 seeds | 0.5639 (0.5602 to 0.5688) | 0.5817 | 0.0723 → 0.0617 | 0.5493 | 1.3067 | 0.6688 | 0.6237 | - |
| FT `laya-multilingual` (E3), 3 seeds | 0.5885 (0.5842 to 0.5915) | 0.6057 | 0.0435 → 0.0439 | 0.5259 | 1.2533 | 0.6951 | 0.6498 | - |
| Reference LLM qwen3_4b (B5, eval subsets) | 0.5022 | 0.5060 | 0.4483 → 0.0614 | 0.6341 | 1.5545 | 0.5837 | 0.5417 | - |
| Majority (B1) [c7] | 0.0493 | 0.2087 | 0.7913 → 0.7913 | 1.5827 | 21.8653 | 0.2083 | 0.2067 | - |
| Prior (B1) [c7] | 0.0493 | 0.2087 | 0.0084 → 0.0084 | 0.8478 | 1.9139 | 0.2083 | 0.2067 | - |
| TF-IDF+LR (B3) [c7] | 0.5433 | 0.5450 | 0.0601 → 0.0193 | 0.5878 | 1.2621 | 0.6096 | 0.5796 | - |
| FT `laya` (E5, reported only) [c7] | 0.6235 | 0.6220 | 0.0472 → 0.0406 | 0.5060 | 1.1004 | 0.7033 | 0.6633 | - |
| FT `laya-multilingual` (E5, reported only) [c7] | 0.6348 | 0.6300 | 0.0383 → 0.0487 | 0.4953 | 1.0729 | 0.7108 | 0.6726 | - |

### ood_country

| Arm | Macro-F1 (mean ± range) | Acc | ECE pre → post | Brier | NLL | Acc@80 | Acc@90 | Gap vs ID |
|---|---|---|---|---|---|---|---|---|
| Majority (B1) | 0.0191 | 0.1056 | 0.8944 → 0.8944 | 1.7888 | 24.7130 | 0.1032 | 0.1045 | 0.0014 |
| Prior (B1) | 0.0191 | 0.1056 | 0.0045 → 0.0045 | 0.8999 | 2.3022 | 0.1032 | 0.1045 | 0.0014 |
| Zero-shot `laya` (B2) | 0.2393 | 0.2588 | 0.1529 → 0.1529 | 0.8808 | 2.2341 | 0.2821 | 0.2735 | 0.0733 |
| Zero-shot `laya-ml` (B2) | 0.2484 | 0.2462 | 0.4220 → 0.4220 | 1.0793 | 2.9289 | 0.2658 | 0.2596 | 0.0338 |
| TF-IDF+LR (B3) | 0.3425 | 0.3654 | 0.0469 → 0.0531 | 0.7664 | 1.9177 | 0.4134 | 0.3891 | 0.1441 |
| ModernBERT-base (B4), 3 seeds | 0.4118 (0.4083 to 0.4140) | 0.4268 | 0.0748 → 0.0752 | 0.7121 | 1.7315 | 0.4859 | 0.4551 | 0.1480 |
| mmBERT-small (B4), 3 seeds | 0.5022 (0.4999 to 0.5039) | 0.5264 | 0.0288 → 0.0489 | 0.6169 | 1.4609 | 0.5962 | 0.5622 | 0.0936 |
| Head-only `laya` (E4) | 0.2395 | 0.2608 | 0.0252 → 0.0365 | 0.8400 | 2.1032 | 0.2889 | 0.2774 | 0.0783 |
| FT `laya` (E2), 3 seeds | 0.4300 (0.4233 to 0.4349) | 0.4491 | 0.1055 → 0.0862 | 0.6934 | 1.6738 | 0.5057 | 0.4764 | 0.1339 |
| FT `laya-multilingual` (E3), 3 seeds | 0.4940 (0.4901 to 0.5001) | 0.5160 | 0.0558 → 0.0663 | 0.6289 | 1.5103 | 0.5866 | 0.5494 | 0.0945 |
| Reference LLM qwen3_4b (B5, eval subsets) | 0.4486 | 0.4530 | 0.4934 → 0.0664 | 0.6816 | 1.6743 | 0.5241 | 0.4858 | 0.0536 |
| Majority (B1) [c7] | 0.0522 | 0.2237 | 0.7763 → 0.7763 | 1.5526 | 21.4493 | 0.2195 | 0.2168 | -0.0029 |
| Prior (B1) [c7] | 0.0522 | 0.2237 | 0.0235 → 0.0235 | 0.8469 | 1.9106 | 0.2195 | 0.2168 | -0.0029 |
| TF-IDF+LR (B3) [c7] | 0.3939 | 0.4104 | 0.0312 → 0.0554 | 0.7158 | 1.5902 | 0.4572 | 0.4352 | 0.1494 |
| FT `laya` (E5, reported only) [c7] | 0.4701 | 0.4850 | 0.0981 → 0.0942 | 0.6600 | 1.4302 | 0.5366 | 0.5086 | 0.1534 |
| FT `laya-multilingual` (E5, reported only) [c7] | 0.5427 | 0.5470 | 0.0532 → 0.0663 | 0.5827 | 1.2634 | 0.6148 | 0.5820 | 0.0921 |

### ood_script

| Arm | Macro-F1 (mean ± range) | Acc | ECE pre → post | Brier | NLL | Acc@80 | Acc@90 | Gap vs ID |
|---|---|---|---|---|---|---|---|---|
| Majority (B1) | 0.0197 | 0.1095 | 0.8905 → 0.8905 | 1.7810 | 24.6054 | 0.1100 | 0.1100 | 0.0008 |
| Prior (B1) | 0.0197 | 0.1095 | 0.0084 → 0.0084 | 0.8999 | 2.3020 | 0.1100 | 0.1100 | 0.0008 |
| Zero-shot `laya` (B2) | 0.2331 | 0.2415 | 0.1250 → 0.1250 | 0.8658 | 2.2520 | 0.2712 | 0.2583 | 0.0795 (excluded) |
| Zero-shot `laya-ml` (B2) | 0.2732 | 0.3005 | 0.3854 → 0.3854 | 1.0148 | 2.7994 | 0.3237 | 0.3122 | 0.0090 |
| TF-IDF+LR (B3) | 0.2786 | 0.2955 | 0.1351 → 0.1425 | 0.8313 | 2.3073 | 0.3331 | 0.3111 | 0.2080 |
| ModernBERT-base (B4), 3 seeds | 0.3270 (0.3136 to 0.3352) | 0.3367 | 0.1327 → 0.1325 | 0.7925 | 2.0480 | 0.3781 | 0.3535 | 0.2328 |
| mmBERT-small (B4), 3 seeds | 0.4826 (0.4766 to 0.4891) | 0.5142 | 0.0376 → 0.0636 | 0.6342 | 1.5002 | 0.5806 | 0.5461 | 0.1133 |
| Head-only `laya` (E4) | 0.2344 | 0.2400 | 0.0442 → 0.0400 | 0.8384 | 2.1298 | 0.2706 | 0.2533 | 0.0834 (excluded) |
| FT `laya` (E2), 3 seeds | 0.3244 (0.3175 to 0.3320) | 0.3270 | 0.1402 → 0.1248 | 0.7739 | 1.9680 | 0.3829 | 0.3519 | 0.2395 (excluded) |
| FT `laya-multilingual` (E3), 3 seeds | 0.5058 (0.5033 to 0.5101) | 0.5400 | 0.0491 → 0.0507 | 0.5950 | 1.4175 | 0.6217 | 0.5824 | 0.0827 |
| Reference LLM qwen3_4b (B5, eval subsets) | 0.4627 | 0.4970 | 0.4569 → 0.0663 | 0.6432 | 1.5898 | 0.5725 | 0.5317 | 0.0396 |
| Majority (B1) [c7] | 0.0513 | 0.2190 | 0.7810 → 0.7810 | 1.5620 | 21.5798 | 0.2238 | 0.2233 | -0.0020 |
| Prior (B1) [c7] | 0.0513 | 0.2190 | 0.0188 → 0.0188 | 0.8499 | 1.9212 | 0.2238 | 0.2233 | -0.0020 |
| TF-IDF+LR (B3) [c7] | 0.3039 | 0.3085 | 0.1495 → 0.1873 | 0.8249 | 1.9871 | 0.3400 | 0.3239 | 0.2393 |
| FT `laya` (E5, reported only) [c7] | 0.3815 | 0.3620 | 0.1913 → 0.1859 | 0.7638 | 1.7012 | 0.4181 | 0.3900 | 0.2420 (excluded) |
| FT `laya-multilingual` (E5, reported only) [c7] | 0.5829 | 0.5775 | 0.0524 → 0.0573 | 0.5551 | 1.1962 | 0.6494 | 0.6122 | 0.0519 |

### ood_brand

| Arm | Macro-F1 (mean ± range) | Acc | ECE pre → post | Brier | NLL | Acc@80 | Acc@90 | Gap vs ID |
|---|---|---|---|---|---|---|---|---|
| Majority (B1) | 0.0191 | 0.1055 | 0.8945 → 0.8945 | 1.7890 | 24.7159 | 0.1050 | 0.1056 | 0.0014 |
| Prior (B1) | 0.0191 | 0.1055 | 0.0044 → 0.0044 | 0.8995 | 2.3003 | 0.1050 | 0.1056 | 0.0014 |
| Zero-shot `laya` (B2) | 0.2345 | 0.2655 | 0.2073 → 0.2073 | 0.9002 | 2.1862 | 0.2938 | 0.2800 | 0.0781 |
| Zero-shot `laya-ml` (B2) | 0.2078 | 0.2345 | 0.4088 → 0.4088 | 1.0919 | 3.0324 | 0.2387 | 0.2383 | 0.0744 |
| TF-IDF+LR (B3) | 0.4493 | 0.5245 | 0.0394 → 0.0401 | 0.6195 | 1.4452 | 0.5900 | 0.5539 | 0.0373 |
| ModernBERT-base (B4), 3 seeds | 0.5211 (0.4982 to 0.5408) | 0.5987 | 0.0976 → 0.0969 | 0.5215 | 1.2760 | 0.6877 | 0.6391 | 0.0387 |
| mmBERT-small (B4), 3 seeds | 0.5783 (0.5652 to 0.5876) | 0.6412 | 0.1052 → 0.0831 | 0.4647 | 1.0360 | 0.7194 | 0.6807 | 0.0176 |
| Head-only `laya` (E4) | 0.2369 | 0.2675 | 0.0778 → 0.0912 | 0.8466 | 2.0554 | 0.3063 | 0.2878 | 0.0808 |
| FT `laya` (E2), 3 seeds | 0.5881 (0.5566 to 0.6261) | 0.6762 | 0.0970 → 0.1059 | 0.4326 | 1.0264 | 0.7712 | 0.7256 | -0.0242 |
| FT `laya-multilingual` (E3), 3 seeds | 0.5511 (0.5180 to 0.5726) | 0.6177 | 0.0934 → 0.0936 | 0.5173 | 1.2065 | 0.6900 | 0.6561 | 0.0374 |
| Reference LLM qwen3_4b (B5, eval subsets) | 0.4006 | 0.4150 | 0.5562 → 0.1319 | 0.7676 | 1.8590 | 0.4575 | 0.4294 | 0.1017 |
| Majority (B1) [c7] | 0.0566 | 0.2470 | 0.7530 → 0.7530 | 1.5060 | 20.8062 | 0.2462 | 0.2478 | -0.0073 |
| Prior (B1) [c7] | 0.0566 | 0.2470 | 0.0468 → 0.0468 | 0.8809 | 2.0293 | 0.2462 | 0.2478 | -0.0073 |
| TF-IDF+LR (B3) [c7] | 0.5385 | 0.5725 | 0.0728 → 0.0496 | 0.5778 | 1.2480 | 0.6294 | 0.6000 | 0.0048 |
| FT `laya` (E5, reported only) [c7] | 0.6227 | 0.6315 | 0.1195 → 0.1146 | 0.4910 | 1.0587 | 0.7375 | 0.6828 | 0.0008 |
| FT `laya-multilingual` (E5, reported only) [c7] | 0.6234 | 0.6165 | 0.0980 → 0.0955 | 0.5182 | 1.1619 | 0.6906 | 0.6572 | 0.0113 |

## Traps, abstention and stability

Trap accuracy on the annotated keep set, multi-category items split out; before annotation the unannotated-candidate accuracy is shown for information. Stripped: no-evidence rows, abstention at the val tau and the share answered with confidence > 0.8. Flip rate: rows whose argmax differs across seeds. Order invariance: mean argmax agreement over shuffled field orders.

| Arm | Trap acc (all / multi) | Stripped: abstain rate | Stripped: false-confident | Flip rate (test_id) | Order invariance | Seed range macro-F1 test_id (points) | Seed range ECE test_id |
|---|---|---|---|---|---|---|---|
| Majority (B1) | pending (unannotated candidates: 0.1029) | 0.0000 | 1.0000 | n/a | n/a | - | - |
| Prior (B1) | pending (unannotated candidates: 0.1029) | 0.0000 | 0.0000 | n/a | n/a | - | - |
| Zero-shot `laya` (B2) | pending (unannotated candidates: 0.1914) | 0.3380 | 0.0000 | n/a | 0.7104 (all >= 0.99: no) | - | - |
| Zero-shot `laya-ml` (B2) | pending (unannotated candidates: 0.1600) | 0.3020 | 0.2480 | n/a | 0.7586 (all >= 0.99: no) | - | - |
| TF-IDF+LR (B3) | pending (unannotated candidates: 0.3129) | 0.0940 | 0.0000 | n/a | 0.8816 (all >= 0.99: no) | - | - |
| ModernBERT-base (B4), 3 seeds | pending (unannotated candidates: 0.3657) | 1.0000 | 0.0000 | 0.2767 | 0.8991 (all >= 0.99: no) | 0.6 | 0.0067 |
| mmBERT-small (B4), 3 seeds | pending (unannotated candidates: 0.3695) | 1.0000 | 0.0000 | 0.2553 | 0.9507 (all >= 0.99: no) | 1.9 | 0.0143 |
| Head-only `laya` (E4) | pending (unannotated candidates: 0.1971) | 0.5340 | 0.0000 | n/a | 0.7062 (all >= 0.99: no) | - | - |
| FT `laya` (E2), 3 seeds | pending (unannotated candidates: 0.3819) | 1.0000 | 0.0000 | 0.3127 | 0.8944 (all >= 0.99: no) | 0.9 | 0.0136 |
| FT `laya-multilingual` (E3), 3 seeds | pending (unannotated candidates: 0.3662) | 1.0000 | 0.0000 | 0.2717 | 0.9244 (all >= 0.99: no) | 0.7 | 0.0141 |
| Reference LLM qwen3_4b (B5, eval subsets) | pending (unannotated candidates: 0.2586) | 1.0000 | 0.0000 | n/a | n/a | - | - |
| Majority (B1) [c7] | pending (unannotated candidates: 0.1571) | 0.0000 | 1.0000 | n/a | n/a | - | - |
| Prior (B1) [c7] | pending (unannotated candidates: 0.1571) | 0.0000 | 0.0000 | n/a | n/a | - | - |
| TF-IDF+LR (B3) [c7] | pending (unannotated candidates: 0.3500) | 0.2140 | 0.0000 | n/a | 0.9072 (all >= 0.99: no) | - | - |
| FT `laya` (E5, reported only) [c7] | pending (unannotated candidates: 0.4143) | 1.0000 | 0.0000 | n/a | 0.8964 (all >= 0.99: no) | - | - |
| FT `laya-multilingual` (E5, reported only) [c7] | pending (unannotated candidates: 0.4200) | 1.0000 | 0.0000 | n/a | 0.9270 (all >= 0.99: no) | - | - |

## CPU (design §7.11)

fp32; batch-1 latency and batched throughput (batch 32, length-sorted); the first --bench is the judged machine (criterion 5 at 4 threads, stop (c) at 8 threads with ONNX).

| Model | Backend | Threads | Cold start (s) | p50 (ms) | p95 (ms) | Batch rec/s | Peak RAM (GB) | Hardware |
|---|---|---|---|---|---|---|---|---|
| laya | torch | 4 | 15.72 | 1214.9 | 1564.4 | 1.1 | 2.80 | 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4C/8T, 15.69 GB RAM) (judged) |
| laya | torch | 8 | 8.00 | 1006.0 | 2549.2 | 1.0 | 2.80 | 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4C/8T, 15.69 GB RAM) (judged) |
| laya | onnx | 4 | 23.50 | 855.5 | 1130.8 | 0.8 | 4.04 | 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4C/8T, 15.69 GB RAM) (judged) |
| laya | onnx | 8 | 10.68 | 843.0 | 1217.2 | 1.0 | 4.05 | 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4C/8T, 15.69 GB RAM) (judged) |
| laya_ml | torch | 4 | 17.07 | 338.9 | 471.2 | 3.2 | 2.31 | 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4C/8T, 15.69 GB RAM) (judged) |
| laya_ml | torch | 8 | 13.25 | 326.7 | 443.6 | 4.2 | 2.31 | 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4C/8T, 15.69 GB RAM) (judged) |
| laya_ml | onnx | 4 | 13.69 | 308.3 | 408.5 | 2.6 | 2.40 | 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4C/8T, 15.69 GB RAM) (judged) |
| laya_ml | onnx | 8 | 10.90 | 268.1 | 351.5 | 3.0 | 2.40 | 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4C/8T, 15.69 GB RAM) (judged) |
| modernbert_base | hf | 4 | 8.55 | 100.4 | 167.1 | 13.6 | 1.14 | 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4C/8T, 15.69 GB RAM) (judged) |
| modernbert_base | hf | 8 | 7.75 | 136.9 | 257.7 | 13.7 | 1.15 | 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4C/8T, 15.69 GB RAM) (judged) |
| mmbert_small | hf | 4 | 10.55 | 49.2 | 75.0 | 30.9 | 0.84 | 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4C/8T, 15.69 GB RAM) (judged) |
| mmbert_small | hf | 8 | 9.95 | 61.8 | 100.2 | 31.4 | 0.84 | 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4C/8T, 15.69 GB RAM) (judged) |

ONNX acceptance check (judged machine; argmax agreement and max |dp| vs PyTorch fp32): laya passed; laya_ml passed.

Throughput relative to the language-matched small encoder (faster Laya backend):

| Laya model | Small encoder | Threads | Laya rec/s | Encoder rec/s | Ratio |
|---|---|---|---|---|---|
| laya | modernbert_base | 4 | 1.1 | 13.6 | 0.08x |
| laya_ml | mmbert_small | 4 | 3.2 | 30.9 | 0.10x |

## Decision checklist (design §5.12)

Seed means. C1's B4 is the language-matched small encoder (config phase5.small_encoder_match); the strict variant is a sensitivity check. Brackets: bootstrap 95% CI of the lead. A near miss fails by less than half the criterion's margin. Stop is global (every candidate must hit one); the overall verdict is PASS if any candidate passes, else STOP, INVESTIGATE, or NO PASS (no formal stop condition met — LDS judgement).

### E2 laya c10 (seeds 11, 22, 33; OOD pools ood_country, ood_brand; ood_script excluded: cannot read Thai by design)

| Check | Number | Threshold | Result | Note |
|---|---|---|---|---|
| C1 OOD lead over B3 and the language-matched B4 | +4.3 pts (B3 tfidf_lr c10 +11.3 [+9.8, +12.8]; B4 modernbert_base c10 +4.3 [+3.1, +5.3]) | >= +3.0 pts over both | PASS |  |
| C1 strict (sensitivity): best B4 of either language | -3.1 pts (B3 tfidf_lr c10 +11.3 [+9.8, +12.8]; B4 mmbert_small c10 -3.1 [-4.2, -2.1]) | >= +3.0 pts over both | FAIL | reported only; not in the verdict |
| C2 ID->OOD gap on each included pool | 13.4 pts (worst ood_country; ood_country 13.4, ood_brand -2.4) | <= 5.0 pts on each pool | FAIL |  |
| C3 post-T ECE on test_id and the OOD average | 0.0960 (test_id 0.0617; OOD avg 0.0960) | <= 0.05 on both | FAIL |  |
| C4 trap accuracy >= best baseline | n/a | >= the best baseline | PENDING (traps) | trap set not annotated yet (traps merge -> data/trap.jsonl) |
| C5 CPU budget (4 threads, fp32) | p95 1131 ms, 0.8 rec/s (onnx, 4 threads) | p95 <= 500 ms and >= 8 rec/s | FAIL |  |
| stop (a) below B3 and the matched B4 on test_id and the OOD average | below on test_id: no; on the OOD average: no |  | no |  |
| stop (b) post-T ECE > 0.10 (test_id or OOD average) | test_id 0.0617; OOD avg 0.0960 |  | no |  |
| stop (c) p95 > 1000 ms with ONNX on 8 threads | p95 1217 ms (8 threads) |  | YES |  |
| investigate (i) misses a single criterion by less than half its margin | fails: C2, C3, C5 |  | no |  |
| investigate (ii) seed range of a headline macro-F1 > 3 points | ood_brand 7.0, ood_average 3.6 |  | YES |  |
| investigate (iii) the gate failed once and a clear bug was fixed |  |  | no | not applicable: the Phase 2 gate passed first time (PASS 4/4); no gate failure was fixed |

Candidate verdict: **STOP**

### E3 laya_ml c10 (seeds 11, 22, 33; OOD pools ood_country, ood_script, ood_brand)

| Check | Number | Threshold | Result | Note |
|---|---|---|---|---|
| C1 OOD lead over B3 and the language-matched B4 | -0.4 pts (B3 tfidf_lr c10 +16.0 [+14.7, +17.3]; B4 mmbert_small c10 -0.4 [-1.2, +0.5]) | >= +3.0 pts over both | FAIL |  |
| C1 strict (sensitivity): best B4 of either language | -0.4 pts (B3 tfidf_lr c10 +16.0 [+14.7, +17.3]; B4 mmbert_small c10 -0.4 [-1.2, +0.5]) | >= +3.0 pts over both | FAIL | reported only; not in the verdict |
| C2 ID->OOD gap on each included pool | 9.4 pts (worst ood_country; ood_country 9.4, ood_script 8.3, ood_brand 3.7) | <= 5.0 pts on each pool | FAIL |  |
| C3 post-T ECE on test_id and the OOD average | 0.0702 (test_id 0.0439; OOD avg 0.0702) | <= 0.05 on both | FAIL (near miss) |  |
| C4 trap accuracy >= best baseline | n/a | >= the best baseline | PENDING (traps) | trap set not annotated yet (traps merge -> data/trap.jsonl) |
| C5 CPU budget (4 threads, fp32) | p95 408 ms, 2.6 rec/s (onnx, 4 threads) | p95 <= 500 ms and >= 8 rec/s | FAIL |  |
| stop (a) below B3 and the matched B4 on test_id and the OOD average | below on test_id: no; on the OOD average: no |  | no |  |
| stop (b) post-T ECE > 0.10 (test_id or OOD average) | test_id 0.0439; OOD avg 0.0702 |  | no |  |
| stop (c) p95 > 1000 ms with ONNX on 8 threads | p95 352 ms (8 threads) |  | no |  |
| investigate (i) misses a single criterion by less than half its margin | fails: C1, C2, C3 (near), C5 |  | no |  |
| investigate (ii) seed range of a headline macro-F1 > 3 points | ood_brand 5.5 |  | YES |  |
| investigate (iii) the gate failed once and a clear bug was fixed |  |  | no | not applicable: the Phase 2 gate passed first time (PASS 4/4); no gate failure was fixed |

Candidate verdict: **INVESTIGATE**

### Overall

Verdict: **INVESTIGATE** (final). Final: the pending inputs (traps) cannot change it. PASS is out of reach whatever the pending inputs show: every candidate already fails (E2 laya c10: C2, C3, C5; E3 laya_ml c10: C1, C2, C3, C5).

| Pending input | Items | Can change the verdict |
|---|---|---|
| traps | E2 laya c10 C4, E3 laya_ml c10 C4 | no |

## Bootstrap 95% CIs

Paired item bootstrap of the seed-mean macro-F1 difference (candidate - baseline), design §5.11; ood_average uses the candidate's criterion-1 pools.

| Candidate | Baseline | Pool | Diff (points) | 95% CI (points) | P(diff > 0) | P(diff >= 3 points) | Rows |
|---|---|---|---|---|---|---|---|
| E2 laya c10 | B1 majority c10 | test_id | +54.3 | [+52.7, +56.0] | 1.0000 | 1.0000 | all |
| E2 laya c10 | B1 majority c10 | ood_country | +41.1 | [+39.0, +43.1] | 1.0000 | 1.0000 | all |
| E2 laya c10 | B1 majority c10 | ood_script | +30.5 | [+28.7, +32.1] | 1.0000 | 1.0000 | all |
| E2 laya c10 | B1 majority c10 | ood_brand | +56.9 | [+54.5, +59.3] | 1.0000 | 1.0000 | all |
| E2 laya c10 | B1 majority c10 | ood_average (ood_country+ood_brand) | +49.0 | [+47.2, +50.6] | 1.0000 | 1.0000 | all |
| E2 laya c10 | B3 tfidf_lr c10 | test_id | +7.7 | [+6.1, +9.5] | 1.0000 | 1.0000 | all |
| E2 laya c10 | B3 tfidf_lr c10 | ood_country | +8.7 | [+6.7, +10.8] | 1.0000 | 1.0000 | all |
| E2 laya c10 | B3 tfidf_lr c10 | ood_script | +4.6 | [+2.9, +6.3] | 1.0000 | 0.9600 | all |
| E2 laya c10 | B3 tfidf_lr c10 | ood_brand | +13.9 | [+11.8, +16.1] | 1.0000 | 1.0000 | all |
| E2 laya c10 | B3 tfidf_lr c10 | ood_average (ood_country+ood_brand) | +11.3 | [+9.8, +12.8] | 1.0000 | 1.0000 | all |
| E2 laya c10 | B4 mmbert_small c10 | test_id | -3.2 | [-4.4, -1.9] | 0.0000 | 0.0000 | all |
| E2 laya c10 | B4 mmbert_small c10 | ood_country | -7.2 | [-8.8, -5.6] | 0.0000 | 0.0000 | all |
| E2 laya c10 | B4 mmbert_small c10 | ood_script | -15.8 | [-17.7, -14.0] | 0.0000 | 0.0000 | all |
| E2 laya c10 | B4 mmbert_small c10 | ood_brand | +1.0 | [-0.4, +2.4] | 0.9370 | 0.0010 | all |
| E2 laya c10 | B4 mmbert_small c10 | ood_average (ood_country+ood_brand) | -3.1 | [-4.2, -2.1] | 0.0000 | 0.0000 | all |
| E2 laya c10 | B4 modernbert_base c10 | test_id | +0.4 | [-0.7, +1.5] | 0.7810 | 0.0000 | all |
| E2 laya c10 | B4 modernbert_base c10 | ood_country | +1.8 | [+0.2, +3.4] | 0.9880 | 0.0730 | all |
| E2 laya c10 | B4 modernbert_base c10 | ood_script | -0.3 | [-1.6, +1.0] | 0.3610 | 0.0000 | all |
| E2 laya c10 | B4 modernbert_base c10 | ood_brand | +6.7 | [+5.3, +8.1] | 1.0000 | 1.0000 | all |
| E2 laya c10 | B4 modernbert_base c10 | ood_average (ood_country+ood_brand) | +4.3 | [+3.1, +5.3] | 1.0000 | 0.9900 | all |
| E2 laya c10 | B5 qwen3_4b c10 | test_id | +5.8 | [+3.6, +8.0] | 1.0000 | 0.9960 | baseline's evaluated ids |
| E2 laya c10 | B5 qwen3_4b c10 | ood_country | -1.9 | [-4.1, +0.2] | 0.0440 | 0.0000 | all |
| E2 laya c10 | B5 qwen3_4b c10 | ood_script | -13.8 | [-16.2, -11.5] | 0.0000 | 0.0000 | all |
| E2 laya c10 | B5 qwen3_4b c10 | ood_brand | +18.8 | [+16.7, +20.9] | 1.0000 | 1.0000 | all |
| E2 laya c10 | B5 qwen3_4b c10 | ood_average (ood_country+ood_brand) | +8.4 | [+6.9, +10.0] | 1.0000 | 1.0000 | all |
| E3 laya_ml c10 | B1 majority c10 | test_id | +56.8 | [+55.1, +58.4] | 1.0000 | 1.0000 | all |
| E3 laya_ml c10 | B1 majority c10 | ood_country | +47.5 | [+45.4, +49.5] | 1.0000 | 1.0000 | all |
| E3 laya_ml c10 | B1 majority c10 | ood_script | +48.6 | [+46.6, +50.4] | 1.0000 | 1.0000 | all |
| E3 laya_ml c10 | B1 majority c10 | ood_brand | +53.2 | [+50.7, +55.5] | 1.0000 | 1.0000 | all |
| E3 laya_ml c10 | B1 majority c10 | ood_average (ood_country+ood_script+ood_brand) | +49.8 | [+48.5, +50.9] | 1.0000 | 1.0000 | all |
| E3 laya_ml c10 | B3 tfidf_lr c10 | test_id | +10.2 | [+8.6, +12.1] | 1.0000 | 1.0000 | all |
| E3 laya_ml c10 | B3 tfidf_lr c10 | ood_country | +15.1 | [+13.2, +17.3] | 1.0000 | 1.0000 | all |
| E3 laya_ml c10 | B3 tfidf_lr c10 | ood_script | +22.7 | [+20.6, +24.8] | 1.0000 | 1.0000 | all |
| E3 laya_ml c10 | B3 tfidf_lr c10 | ood_brand | +10.2 | [+7.6, +12.7] | 1.0000 | 1.0000 | all |
| E3 laya_ml c10 | B3 tfidf_lr c10 | ood_average (ood_country+ood_script+ood_brand) | +16.0 | [+14.7, +17.3] | 1.0000 | 1.0000 | all |
| E3 laya_ml c10 | B4 mmbert_small c10 | test_id | -0.7 | [-1.7, +0.3] | 0.0750 | 0.0000 | all |
| E3 laya_ml c10 | B4 mmbert_small c10 | ood_country | -0.8 | [-2.2, +0.5] | 0.1420 | 0.0000 | all |
| E3 laya_ml c10 | B4 mmbert_small c10 | ood_script | +2.3 | [+0.8, +3.8] | 1.0000 | 0.1890 | all |
| E3 laya_ml c10 | B4 mmbert_small c10 | ood_brand | -2.7 | [-4.3, -1.2] | 0.0000 | 0.0000 | all |
| E3 laya_ml c10 | B4 mmbert_small c10 | ood_average (ood_country+ood_script+ood_brand) | -0.4 | [-1.2, +0.5] | 0.2000 | 0.0000 | all |
| E3 laya_ml c10 | B4 modernbert_base c10 | test_id | +2.9 | [+1.5, +4.3] | 1.0000 | 0.4610 | all |
| E3 laya_ml c10 | B4 modernbert_base c10 | ood_country | +8.2 | [+6.5, +10.0] | 1.0000 | 1.0000 | all |
| E3 laya_ml c10 | B4 modernbert_base c10 | ood_script | +17.9 | [+16.1, +19.8] | 1.0000 | 1.0000 | all |
| E3 laya_ml c10 | B4 modernbert_base c10 | ood_brand | +3.0 | [+1.1, +4.8] | 0.9980 | 0.5200 | all |
| E3 laya_ml c10 | B4 modernbert_base c10 | ood_average (ood_country+ood_script+ood_brand) | +9.7 | [+8.7, +10.7] | 1.0000 | 1.0000 | all |
| E3 laya_ml c10 | B5 qwen3_4b c10 | test_id | +7.9 | [+5.8, +10.1] | 1.0000 | 1.0000 | baseline's evaluated ids |
| E3 laya_ml c10 | B5 qwen3_4b c10 | ood_country | +4.5 | [+2.5, +6.6] | 1.0000 | 0.9250 | all |
| E3 laya_ml c10 | B5 qwen3_4b c10 | ood_script | +4.3 | [+2.2, +6.6] | 1.0000 | 0.8840 | all |
| E3 laya_ml c10 | B5 qwen3_4b c10 | ood_brand | +15.1 | [+12.7, +17.5] | 1.0000 | 1.0000 | all |
| E3 laya_ml c10 | B5 qwen3_4b c10 | ood_average (ood_country+ood_script+ood_brand) | +8.0 | [+6.7, +9.2] | 1.0000 | 1.0000 | all |

## Per-class P / R / F1 (candidates and the best baseline)

Cells: P / R / F1, seed means. Best baseline: B4 mmbert_small c10 (highest test_id macro-F1).

### test_id

| Class | E2 laya c10 | E3 laya_ml c10 | B4 mmbert_small c10 |
|---|---|---|---|
| arts | 0.479 / 0.392 / 0.430 | 0.469 / 0.422 / 0.444 | 0.527 / 0.401 / 0.455 |
| services | 0.490 / 0.563 / 0.523 | 0.473 / 0.588 / 0.521 | 0.538 / 0.593 / 0.563 |
| community | 0.604 / 0.544 / 0.572 | 0.668 / 0.543 / 0.597 | 0.619 / 0.605 / 0.611 |
| dining | 0.635 / 0.656 / 0.645 | 0.696 / 0.659 / 0.675 | 0.664 / 0.668 / 0.666 |
| event | 0.268 / 0.567 / 0.363 | 0.266 / 0.550 / 0.358 | 0.273 / 0.596 / 0.371 |
| health | 0.755 / 0.798 / 0.776 | 0.814 / 0.815 / 0.814 | 0.799 / 0.839 / 0.819 |
| outdoors | 0.572 / 0.500 / 0.533 | 0.581 / 0.532 / 0.554 | 0.566 / 0.541 / 0.551 |
| retail | 0.466 / 0.455 / 0.459 | 0.505 / 0.529 / 0.516 | 0.511 / 0.492 / 0.501 |
| sports | 0.690 / 0.670 / 0.679 | 0.792 / 0.669 / 0.725 | 0.770 / 0.685 / 0.725 |
| travel | 0.696 / 0.630 / 0.659 | 0.687 / 0.679 / 0.683 | 0.709 / 0.689 / 0.698 |

### ood_country

| Class | E2 laya c10 | E3 laya_ml c10 | B4 mmbert_small c10 |
|---|---|---|---|
| arts | 0.382 / 0.318 / 0.344 | 0.378 / 0.404 / 0.390 | 0.483 / 0.406 / 0.441 |
| services | 0.310 / 0.475 / 0.374 | 0.384 / 0.525 / 0.440 | 0.388 / 0.472 / 0.425 |
| community | 0.487 / 0.344 / 0.403 | 0.620 / 0.449 / 0.518 | 0.557 / 0.506 / 0.529 |
| dining | 0.502 / 0.556 / 0.527 | 0.563 / 0.586 / 0.572 | 0.570 / 0.603 / 0.586 |
| event | 0.142 / 0.420 / 0.210 | 0.124 / 0.435 / 0.193 | 0.157 / 0.377 / 0.217 |
| health | 0.640 / 0.583 / 0.610 | 0.712 / 0.694 / 0.699 | 0.691 / 0.703 / 0.695 |
| outdoors | 0.455 / 0.452 / 0.453 | 0.533 / 0.491 / 0.511 | 0.479 / 0.538 / 0.505 |
| retail | 0.346 / 0.332 / 0.336 | 0.410 / 0.393 / 0.400 | 0.420 / 0.397 / 0.408 |
| sports | 0.590 / 0.594 / 0.591 | 0.725 / 0.604 / 0.657 | 0.724 / 0.625 / 0.671 |
| travel | 0.570 / 0.382 / 0.451 | 0.635 / 0.502 / 0.560 | 0.618 / 0.492 / 0.546 |

### ood_script

| Class | E2 laya c10 | E3 laya_ml c10 | B4 mmbert_small c10 |
|---|---|---|---|
| arts | 0.271 / 0.410 / 0.325 | 0.405 / 0.380 / 0.391 | 0.403 / 0.424 / 0.412 |
| services | 0.508 / 0.304 / 0.378 | 0.496 / 0.423 / 0.452 | 0.662 / 0.376 / 0.478 |
| community | 0.459 / 0.144 / 0.218 | 0.469 / 0.429 / 0.444 | 0.387 / 0.531 / 0.447 |
| dining | 0.303 / 0.470 / 0.367 | 0.615 / 0.775 / 0.685 | 0.585 / 0.760 / 0.661 |
| event | 0.018 / 0.333 / 0.033 | 0.069 / 0.404 / 0.118 | 0.044 / 0.228 / 0.073 |
| health | 0.695 / 0.350 / 0.466 | 0.886 / 0.781 / 0.830 | 0.863 / 0.643 / 0.735 |
| outdoors | 0.313 / 0.149 / 0.195 | 0.410 / 0.405 / 0.405 | 0.354 / 0.380 / 0.364 |
| retail | 0.480 / 0.201 / 0.282 | 0.500 / 0.395 / 0.440 | 0.581 / 0.381 / 0.457 |
| sports | 0.743 / 0.502 / 0.599 | 0.784 / 0.715 / 0.748 | 0.713 / 0.619 / 0.663 |
| travel | 0.364 / 0.435 / 0.381 | 0.540 / 0.551 / 0.545 | 0.553 / 0.521 / 0.536 |

### ood_brand

| Class | E2 laya c10 | E3 laya_ml c10 | B4 mmbert_small c10 |
|---|---|---|---|
| arts | 0.419 / 0.202 / 0.269 | 0.286 / 0.179 / 0.219 | 0.369 / 0.250 / 0.298 |
| services | 0.400 / 0.840 / 0.540 | 0.315 / 0.886 / 0.463 | 0.341 / 0.865 / 0.488 |
| community | 0.926 / 0.637 / 0.754 | 0.920 / 0.523 / 0.664 | 0.947 / 0.634 / 0.757 |
| dining | 0.791 / 0.852 / 0.820 | 0.910 / 0.842 / 0.874 | 0.890 / 0.889 / 0.890 |
| event | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| health | 0.904 / 0.800 / 0.849 | 0.961 / 0.725 / 0.826 | 0.963 / 0.752 / 0.844 |
| outdoors | 0.568 / 0.698 / 0.626 | 0.533 / 0.508 / 0.514 | 0.533 / 0.603 / 0.559 |
| retail | 0.464 / 0.678 / 0.551 | 0.466 / 0.716 / 0.565 | 0.406 / 0.654 / 0.497 |
| sports | 0.753 / 0.907 / 0.821 | 0.845 / 0.912 / 0.877 | 0.874 / 0.873 / 0.873 |
| travel | 0.934 / 0.509 / 0.650 | 0.903 / 0.374 / 0.510 | 0.975 / 0.409 / 0.576 |

## Look-alike confusions (design §5.11 pairs)

Rate = confusions / true-class support over the group's runs (count/support).

### test_id

| True → predicted | E2 laya c10 | E3 laya_ml c10 | B4 mmbert_small c10 |
|---|---|---|---|
| dining → arts | 0.049 (50/1029) | 0.040 (41/1029) | 0.032 (33/1029) |
| arts → dining | 0.094 (85/906) | 0.082 (74/906) | 0.089 (81/906) |
| sports → outdoors | 0.043 (48/1110) | 0.038 (42/1110) | 0.044 (49/1110) |
| outdoors → sports | 0.052 (47/906) | 0.028 (25/906) | 0.031 (28/906) |
| services → community | 0.036 (34/942) | 0.037 (35/942) | 0.039 (37/942) |
| community → services | 0.064 (60/936) | 0.075 (70/936) | 0.045 (42/936) |
| retail → services | 0.217 (208/957) | 0.212 (203/957) | 0.191 (183/957) |
| services → retail | 0.143 (135/942) | 0.152 (143/942) | 0.132 (124/942) |

### ood_country

| True → predicted | E2 laya c10 | E3 laya_ml c10 | B4 mmbert_small c10 |
|---|---|---|---|
| dining → arts | 0.046 (29/633) | 0.082 (52/633) | 0.041 (26/633) |
| arts → dining | 0.109 (65/594) | 0.089 (53/594) | 0.098 (58/594) |
| sports → outdoors | 0.055 (37/675) | 0.052 (35/675) | 0.076 (51/675) |
| outdoors → sports | 0.060 (42/705) | 0.034 (24/705) | 0.045 (32/705) |
| services → community | 0.058 (38/657) | 0.049 (32/657) | 0.070 (46/657) |
| community → services | 0.159 (109/684) | 0.123 (84/684) | 0.086 (59/684) |
| retail → services | 0.289 (195/675) | 0.249 (168/675) | 0.241 (163/675) |
| services → retail | 0.161 (106/657) | 0.172 (113/657) | 0.180 (118/657) |

### ood_script

| True → predicted | E2 laya c10 | E3 laya_ml c10 | B4 mmbert_small c10 |
|---|---|---|---|
| dining → arts | 0.105 (69/657) | 0.037 (24/657) | 0.052 (34/657) |
| arts → dining | 0.145 (91/627) | 0.118 (74/627) | 0.121 (76/627) |
| sports → outdoors | 0.044 (29/657) | 0.055 (36/657) | 0.117 (77/657) |
| outdoors → sports | 0.025 (16/639) | 0.055 (35/639) | 0.050 (32/639) |
| services → community | 0.051 (31/612) | 0.077 (47/612) | 0.160 (98/612) |
| community → services | 0.020 (14/702) | 0.048 (34/702) | 0.017 (12/702) |
| retail → services | 0.062 (44/711) | 0.134 (95/711) | 0.045 (32/711) |
| services → retail | 0.067 (41/612) | 0.147 (90/612) | 0.108 (66/612) |

### ood_brand

| True → predicted | E2 laya c10 | E3 laya_ml c10 | B4 mmbert_small c10 |
|---|---|---|---|
| dining → arts | 0.000 (0/633) | 0.002 (1/633) | 0.006 (4/633) |
| arts → dining | 0.000 (0/168) | 0.000 (0/168) | 0.000 (0/168) |
| sports → outdoors | 0.009 (6/636) | 0.002 (1/636) | 0.008 (5/636) |
| outdoors → sports | 0.095 (6/63) | 0.079 (5/63) | 0.016 (1/63) |
| services → community | 0.013 (9/675) | 0.010 (7/675) | 0.000 (0/675) |
| community → services | 0.160 (129/807) | 0.351 (283/807) | 0.223 (180/807) |
| retail → services | 0.148 (93/627) | 0.164 (103/627) | 0.187 (117/627) |
| services → retail | 0.073 (49/675) | 0.059 (40/675) | 0.084 (57/675) |

