# Laya PoC: verdict memo

| | |
|---|---|
| Date | 28 September 2026 |
| For | Lead Data Scientist (LDS) and reviewer (REV): decision and sign-off |
| Status | Final for review. Two inputs are still open and neither can change the verdict: trap annotation (C4) and a doc-spec CPU run (C5) |
| Detail | [Phase 5 report](phase5_report_2026-09-28.md), [results workbook](phase7_results_workbook_2026-09-28.xlsx), [Phase 3/4 report](phase34_report_2026-09-28.md) |

## 1. Verdict

**§5.12 verdict: INVESTIGATE (final). Recommendation: do not take Laya to an internal trial.**

**What we tested.** We fine-tuned both Laya checkpoints on 25k public Foursquare place records. Each model had to put a record into one of 10 categories. We tested on held-out records and on three pools of unseen data: new countries, Thai script and new brands. We used three seeds each and compared against baselines trained on the same splits.

**Accuracy.** Fine-tuning works. Multilingual `laya_ml` reaches 0.589 macro-F1 on held-out data (61% top-1 accuracy), up from 0.28 zero-shot, and it beats TF-IDF by 16 points on unseen data. But a plainly fine-tuned **mmBERT-small** (140M parameters, 80 s of training) matches it: 0.596 macro-F1 (62% accuracy). mmBERT-small is also better calibrated and **about 10× faster on CPU**.

**Pass criteria.** Neither Laya checkpoint meets any of criteria 1, 2, 3 or 5. The criteria are measured on held-out ("in-distribution", ID) data and on the unseen-data ("out-of-distribution", OOD) pools. PASS is out of reach whatever the open inputs show.

**Why INVESTIGATE and not STOP.** The formal verdict is INVESTIGATE because two things hold:
- English `laya` hits stop condition (c), but `laya_ml` hits no stop condition.
- `laya_ml`'s score on the new-brands pool varies by more than 3 points across seeds.

**Why we recommend against spending the investigation.** The allowed investigation is at most 2 days and targets seed variance. It cannot close a −0.4-point lead where +3 is needed, a 9.4-point ID→OOD gap against a 5-point limit, or a 2.5× throughput shortfall. We recommend the LDS closes the PoC on this evidence rather than spending that iteration.

## 2. Accuracy and unseen data

Macro-F1 is the unweighted mean of the per-class F1 scores, with 10 classes. Fine-tuned rows show the mean over 3 seeds. Post-T ECE is the expected calibration error after temperature scaling (lower is better).

| Model | Test (ID) | New countries | Thai script | New brands | OOD avg | ECE test / OOD |
|---|---|---|---|---|---|---|
| **`laya_ml` fine-tuned (E3)** | 0.589 | 0.494 | 0.506 | 0.551 | **0.517** | 0.044 / 0.070 |
| `laya` (English) fine-tuned (E2) | 0.564 | 0.430 | 0.324 ¹ | 0.588 | 0.509 ¹ | 0.062 / 0.096 |
| **mmBERT-small fine-tuned (B4)** | **0.596** | **0.502** | 0.483 | 0.578 | **0.521** | **0.026 / 0.065** |
| ModernBERT-base fine-tuned (B4) | 0.560 | 0.412 | 0.327 | 0.521 | 0.420 | 0.028 / 0.102 |
| Qwen3-4B zero-shot LLM (B5) ² | 0.502 | 0.449 | 0.463 | 0.401 | 0.437 | 0.061 / 0.088 |
| Char TF-IDF + LR (B3) | 0.487 | 0.343 | 0.279 | 0.449 | 0.357 | 0.022 / 0.079 |
| `laya` zero-shot / `laya_ml` zero-shot (B2) | 0.313 / 0.282 | | | | | |

¹ English `laya` cannot read Thai by design, so `ood_script` is excluded from its criteria. Its OOD average covers new countries and new brands only.

² Scored on evaluation subsets, not the full splits.

**How it performed on unseen data.**
- **The gap is similar for both models.** `laya_ml` loses 9.4 points going to new countries, 8.3 on Thai and 3.7 on new brands; the limit is ≤5 on each pool. mmBERT-small's drops are similar: 9.4, 11.3 and 1.8.
- **Where Laya wins and loses.** On Thai script `laya_ml` is the best model, 2.3 points ahead of mmBERT-small (95% CI [+0.8, +3.8]). On new brands it trails by 2.7 [−4.3, −1.2]. Over all three pools it is level: −0.4 [−1.2, +0.5].
- **English Laya against its matched encoder.** English `laya` beats its language-matched encoder, ModernBERT-base, by 4.3 points on unseen data [+3.1, +5.3]. It still trails mmBERT-small by 3.1.
- **Seed stability is weak on new brands.** English `laya` ranges 0.557–0.626 across seeds (7.0 points) and `laya_ml` 0.518–0.573 (5.5 points). The encoders stay within 2 points.

**Where both models go wrong.** In the seed-11 runs on the held-out test set, the two models agree on right or wrong for 85% of records. Each gets about 220 of 3,000 records right that the other misses. Most errors are between look-alike categories:
- 19–29% of retail records are labelled services, depending on the pool;
- on new brands, `laya_ml` labels 35% of community records as services (mmBERT-small 22%).

**Sample inferences.** A spot-check of sampled errors shows many records where the FSQ label itself looks doubtful, for example a street address labelled as a civic place. So part of the ~60% accuracy ceiling may be label noise. This is not measured; the design's label-quality audit would quantify it. The examples themselves are in a local file only (`runs/phase7/sample_inferences.md`), because FSQ rows may not leave the team (Appendix D).

## 3. Calibration, abstention, stability

- **Calibration.** After temperature scaling, `laya_ml` misses C3 on unseen data (OOD ECE 0.070 against a limit of 0.05). It meets it on held-out data (0.044). English `laya` misses on both.
- **Abstention.** On 1,000 records with every field except the country removed, all models abstain. That is the evidence gate working. Without the gate, both Laya models answer near-uniformly (mean confidence 0.10), which is the right behaviour. TF-IDF is confidently wrong (0.45).
- **Consistency.** Answers change when the record's fields are shuffled for 7–11% of records for Laya and 5% for mmBERT-small; the design wants at most 1%. Across seeds, 27–31% of test answers change (26% for mmBERT-small).

## 4. Speed: CPU and GPU

**CPU** (criterion 5; doc budget p95 ≤500 ms per record and ≥8 records/s at 4 threads, fp32). Measured on the maintainer's laptop (i5-1135G7, 4 cores / 8 threads), which is not the doc's 4-vCPU cloud host.

| Model | Backend | p50 / p95 per record | Batched records/s | vs matched encoder |
|---|---|---|---|---|
| `laya` | ONNX | 855 / 1131 ms | 0.8 (1.1 PyTorch) | 0.08× |
| `laya_ml` | ONNX | 308 / 408 ms | 2.6 (3.2 PyTorch) | 0.10× |
| ModernBERT-base | PyTorch | 100 / 167 ms | 13.6 | |
| mmBERT-small | PyTorch | 49 / 75 ms | 30.9 | |

- **ONNX export is exact.** Both Laya ONNX exports agree with PyTorch on all 1,000 checked records (max |Δp| 1.3e-05).
- **`laya_ml` fails on throughput.** It meets the latency limit but reaches only a third of the throughput floor.
- **English `laya` fails on both, and hits stop condition (c).** Its p95 is 1,217 ms even with ONNX on 8 threads.

**GPU** (Colab G4, RTX PRO 6000 Blackwell; fp16 autocast; end to end from the record string to probabilities).
- **Per-record latency** comes from the Phase 7 GPU benchmark: one call at a time, 500 records, after 20 warm-up calls ([runbook](../phase7_gpu_runbook.md)).
- **Batched throughput** is at batch 64.
- **Training time** comes from the Phase 3/4 runs ([detail](phase7_gpu_timing_2026-09-28.md)).

| Model | p50 / p95 per record | Batched records/s | Peak VRAM | Train + evaluate, one run |
|---|---|---|---|---|
| `laya` | 9.6 / 9.7 ms | 668 | 2.9 GiB | 9.7 min |
| `laya_ml` | 7.8 / 8.0 ms | 1,305 | 1.9 GiB | 5.2 min |
| ModernBERT-base | 6.6 / 7.3 ms | 4,745 | 1.0 GiB | 1.5 min |
| mmBERT-small | 6.9 / 7.5 ms | 6,426 | 0.7 GiB | 1.5 min |
| Qwen3-4B zero-shot | not measured | 12.7 ³ | | 13.7 min (scoring only) |

³ Scored with 6 items (60 sequences) per forward pass, because the packed-scoring check failed. Measured in the Phase 4 run.

- **GPU speed does not block Laya.** Every model answers a single record in under 10 ms. At batch 1, fixed per-call overhead dominates, so Laya is only 1.1–1.4× slower.
- **The gap widens with batching.** Batched, the encoders are 5–10× faster (1,305 vs 6,426 records/s for the multilingual pair).
- **The benchmark agrees with the evaluation runs.** Its batched rates match those measured during the Phase 3/4 evaluations (668 vs 671, 1,305 vs 1,288 and 6,426 vs 6,244 records/s). The GPU cost of Laya is a hardware bill, not a blocker. CPU is where it fails (C5).

## 5. Decision checklist (§5.12, seed means)

| | `laya_ml` (E3) | `laya` (E2) |
|---|---|---|
| C1 OOD lead ≥ +3 over TF-IDF and the language-matched encoder | FAIL: −0.4 vs mmBERT-small (+16.0 vs TF-IDF) | PASS: +4.3 vs ModernBERT-base (+11.3 vs TF-IDF); strict check vs mmBERT-small −3.1 |
| C2 ID→OOD gap ≤5 on each pool | FAIL: 9.4 worst (new countries) | FAIL: 13.4 |
| C3 post-T ECE ≤0.05, test and OOD | FAIL (near miss): 0.044 / 0.070 | FAIL: 0.062 / 0.096 |
| C4 trap accuracy ≥ best baseline | pending annotation (cannot change verdict) | pending |
| C5 CPU p95 ≤500 ms and ≥8 rec/s | FAIL: 408 ms, 3.2 rec/s | FAIL: 1131 ms, 1.1 rec/s |
| Stop (a) below both baselines / (b) ECE > 0.10 / (c) p95 > 1 s on ONNX at 8 threads | no / no / no | no / no / **yes** |
| Investigate: seed range > 3 points | yes (new brands 5.5) | yes (new brands 7.0) |
| **Candidate verdict** | **INVESTIGATE** | **STOP** |

**Three interpretations need LDS sign-off:**
1. C1 compares each Laya checkpoint with its language-matched small encoder. The strict best-encoder variant is reported as a sensitivity check.
2. Stop is global: STOP applies only if every candidate stops.
3. On a 4-core host, the 8-thread stop (c) row uses hyperthreads.

None of these changes the recommendation.

## 6. Limits and next steps

**Limits.**
- **Public data only.** A pass on public place data would not have shown performance on the target data (§5.12). This result is a no, and it does not need that caveat to hold.
- **CPU hardware.** The CPU numbers are from a laptop at reduced row counts (200 latency rows, 500 batch rows). Four physical cores are likely at least as fast as 4 cloud vCPUs, so the throughput misses should not reverse.
- **Traps.** The trap set (700 candidates) is not yet annotated.
- **Training size.** Only `laya` was trained on the learning-curve subsets. Its macro-F1 still rises from 10k to 25k records (0.528 → 0.566). More data would likely lift every model, not Laya alone.

**Next steps.**
1. The LDS signs off the verdict and the three interpretations, and decides whether to close the PoC or run the 2-day investigation. We recommend closing.
2. If record normalisation is still wanted, take the fine-tuned mmBERT-small approach into the "what a pass does not show" checks (§5.12) on target data: label audit, token budget, target look-alikes, calibration re-fit and a CPU check on the target hardware.
3. Optional:
   - annotate the trap set (`docs/trap_annotation_guide.md`) to complete C4;
   - run `notebooks/phase5_cpu_bench.ipynb` on Colab to judge C5 on doc-spec hardware.

## 7. Provenance

| Item | Value |
|---|---|
| Code | branch `phase-5`; decision thresholds fixed in `config.yaml` before any results |
| Laya | NandhaKishorM/laya @ 4066d5d (v0.3.20); Hub `convaiinnovations/laya` @ 55cf4c4 (root and `multilingual/`) |
| Baselines | ModernBERT-base @ 8949b909, mmBERT-small @ abc32620, Qwen3-4B @ 1cfa9a72 |
| Data | FSQ OS Places release 2026-09-15; frozen build fingerprint 64eafe66…3db ([manifest](data_fingerprint_2026-09-15.json)); train 25,000, val 3,000, test 3,000, OOD 1,998 / 2,000 / 2,000, stripped 1,000, 700 trap candidates held out |
| Seeds | 11, 22, 33 (fine-tuned Laya and B4); one run each for B1, B3, B5 and the ablations |
| Hardware | training, GPU scoring and the GPU latency benchmark on a Colab G4 (RTX PRO 6000 Blackwell; torch 2.11 + CUDA 12.8); CPU on an i5-1135G7 laptop |
| Tests | full suite passing on `phase-5` |

**FSQ attribution** (design Appendix D):

```
Copyright 2024 Foursquare Labs, Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Derived from Foursquare Open Source Places release 2026-09-15; modified: fields removed, records filtered and sampled.
```
