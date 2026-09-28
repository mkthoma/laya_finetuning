<!-- Copied from results/phase3_report.md of the Colab G4 run on 2026-09-27 (notebooks/phase3_matrix.local.ipynb, bundle fae1ee11). Metrics only: no FSQ records.
     Per-run table: docs/results/phase3_runs_2026-09-27.csv. The early read against §5.12 at the end is NOT the Phase 5 verdict. -->

# Laya PoC: Phase 3 seed matrix

Generated 2026-09-27T19:19:01+00:00; card(s) G4; 14 finished of 14 configured runs. Phase 3 exit check: **PASS** (14/14).

Post-T metrics unless marked pre; groups of several seeds show `mean (min to max)` over seeds. B2 rows are the zero-shot hub checkpoints (shipped temperature). Metrics only: no FSQ rows.

## Accuracy: macro-F1 (post-T)

| Group | Runs | Seeds | val | val (9-class) | test_id | ood_country | ood_script | ood_brand |
|---|---|---|---|---|---|---|---|---|
| B2 laya c10 (zero-shot, shipped T) | 1 | - | 0.3025 | 0.3119 | 0.3126 | 0.2393 | 0.2331 | 0.2345 |
| B2 laya_ml c10 (zero-shot, shipped T) | 1 | - | 0.2813 | 0.2951 | 0.2823 | 0.2484 | 0.2732 | 0.2078 |
| E2 laya c10 | 3 | 11, 22, 33 | 0.5660 (0.5617 to 0.5706) | 0.5929 (0.5860 to 0.5987) | 0.5639 (0.5602 to 0.5688) | 0.4300 (0.4233 to 0.4349) | 0.3244 (0.3175 to 0.3320) | 0.5881 (0.5566 to 0.6261) |
| E3 laya_ml c10 | 3 | 11, 22, 33 | 0.5832 (0.5781 to 0.5875) | 0.6166 (0.6109 to 0.6203) | 0.5885 (0.5842 to 0.5915) | 0.4940 (0.4901 to 0.5001) | 0.5058 (0.5033 to 0.5101) | 0.5511 (0.5180 to 0.5726) |
| E4 laya c10 head-only | 1 | 11 | 0.3099 | 0.3192 | 0.3177 | 0.2395 | 0.2344 | 0.2369 |
| E5 laya c7 | 1 | 11 | 0.6191 | n/a | 0.6235 | 0.4701 | 0.3815 | 0.6227 |
| E5 laya_ml c7 | 1 | 11 | 0.6332 | n/a | 0.6348 | 0.5427 | 0.5829 | 0.6234 |
| E6 laya c10 n=1000 | 1 | 11 | 0.4484 | 0.4693 | 0.4481 | 0.3443 | 0.2877 | 0.4233 |
| E6 laya c10 n=3000 | 1 | 11 | 0.4779 | 0.4996 | 0.4736 | 0.3718 | 0.2907 | 0.4807 |
| E6 laya c10 n=10000 | 1 | 11 | 0.5278 | 0.5573 | 0.5249 | 0.3998 | 0.3278 | 0.5154 |

## ID->OOD gap

Gap = macro-F1(test_id) - macro-F1(pool) per run, then averaged (design §5.11). The decision rule (§5.12, judged in Phase 5) wants <= 5 points on each included pool; `laya` cannot read Thai by design, so its ood_script gap is excluded there and only reported.

| Group | ood_country | ood_script | ood_brand |
|---|---|---|---|
| B2 laya c10 (zero-shot, shipped T) | 0.0733 | 0.0795 (excluded) | 0.0781 |
| B2 laya_ml c10 (zero-shot, shipped T) | 0.0338 | 0.0090 | 0.0744 |
| E2 laya c10 | 0.1339 (0.1252 to 0.1456) | 0.2395 (0.2369 to 0.2426) (excluded) | -0.0242 (-0.0659 to 0.0060) |
| E3 laya_ml c10 | 0.0945 (0.0914 to 0.0979) | 0.0827 (0.0742 to 0.0876) | 0.0374 (0.0171 to 0.0662) |
| E4 laya c10 head-only | 0.0783 | 0.0834 (excluded) | 0.0808 |
| E5 laya c7 | 0.1534 | 0.2420 (excluded) | 0.0008 |
| E5 laya_ml c7 | 0.0921 | 0.0519 | 0.0113 |
| E6 laya c10 n=1000 | 0.1038 | 0.1604 (excluded) | 0.0248 |
| E6 laya c10 n=3000 | 0.1018 | 0.1829 (excluded) | -0.0071 |
| E6 laya c10 n=10000 | 0.1252 | 0.1971 (excluded) | 0.0095 |

## Calibration: T and ECE pre -> post

T: fitted on val by export_check for trained runs; the shipped temperature for zero-shot. ECE: 15 equal-width bins on answer_confidence, before and after the temperature.

| Group | T | val ECE pre -> post | test_id ECE pre -> post |
|---|---|---|---|
| B2 laya c10 (zero-shot, shipped T) | 1.0000 | 0.1237 -> 0.1237 | 0.1212 -> 0.1212 |
| B2 laya_ml c10 (zero-shot, shipped T) | 1.0000 | 0.3513 -> 0.3513 | 0.3449 -> 0.3449 |
| E2 laya c10 | 1.0658 (1.0274 to 1.1037) | 0.0719 (0.0550 to 0.0905) -> 0.0570 (0.0503 to 0.0697) | 0.0723 (0.0569 to 0.0875) -> 0.0617 (0.0552 to 0.0688) |
| E3 laya_ml c10 | 0.9562 (0.9134 to 0.9893) | 0.0491 (0.0393 to 0.0603) -> 0.0490 (0.0310 to 0.0614) | 0.0435 (0.0417 to 0.0466) -> 0.0439 (0.0348 to 0.0489) |
| E4 laya c10 head-only | 0.8780 | 0.0355 -> 0.0238 | 0.0395 -> 0.0283 |
| E5 laya c7 | 1.0149 | 0.0463 -> 0.0430 | 0.0472 -> 0.0406 |
| E5 laya_ml c7 | 0.9530 | 0.0446 -> 0.0433 | 0.0383 -> 0.0487 |
| E6 laya c10 n=1000 | 1.1068 | 0.0491 -> 0.0284 | 0.0556 -> 0.0253 |
| E6 laya c10 n=3000 | 1.0751 | 0.0488 -> 0.0423 | 0.0641 -> 0.0412 |
| E6 laya c10 n=10000 | 1.0940 | 0.0686 -> 0.0566 | 0.0796 -> 0.0573 |

## Traps, no evidence and field-order invariance

Trap accuracy is on every trap CANDIDATE with its automatic label (annotation pending), so it is not the §5.11 trap metric; Phase 5 restricts it to the annotated keep set via fsq_place_id. tau is chosen on val (abstain.target_acc / min_coverage).

| Group | Trap acc (unannotated candidates) | Stripped: abstain rate at tau | Stripped: false-confident (conf > 0.8) | Order invariance (mean agreement) | All >= 0.99 |
|---|---|---|---|---|---|
| B2 laya c10 (zero-shot, shipped T) | 0.1914 | 0.3380 | 0.0000 | 0.7104 | no |
| B2 laya_ml c10 (zero-shot, shipped T) | 0.1600 | 0.3020 | 0.2480 | 0.7586 | no |
| E2 laya c10 | 0.3819 (0.3657 to 0.3929) | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) | 0.8944 (0.8860 to 0.9008) | no |
| E3 laya_ml c10 | 0.3662 (0.3629 to 0.3686) | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) | 0.9244 (0.9188 to 0.9296) | no |
| E4 laya c10 head-only | 0.1971 | 0.5340 | 0.0000 | 0.7062 | no |
| E5 laya c7 | 0.4143 | 1.0000 | 0.0000 | 0.8964 | no |
| E5 laya_ml c7 | 0.4200 | 1.0000 | 0.0000 | 0.9270 | no |
| E6 laya c10 n=1000 | 0.2671 | 1.0000 | 0.0000 | 0.7954 | no |
| E6 laya c10 n=3000 | 0.2971 | 1.0000 | 0.0000 | 0.8444 | no |
| E6 laya c10 n=10000 | 0.3671 | 1.0000 | 0.0000 | 0.8784 | no |

## Seed variance (design §5.12: Investigate when a macro-F1 range over seeds exceeds 3 points)

Flagged: E2 laya c10, E3 laya_ml c10.

| Group | Metric | Seeds | Range (points) | Flag |
|---|---|---|---|---|
| E2 laya c10 | val_macro_f1 | 3 | 0.9 | ok |
| E2 laya c10 | test_id_macro_f1 | 3 | 0.9 | ok |
| E2 laya c10 | ood_country_macro_f1 | 3 | 1.2 | ok |
| E2 laya c10 | ood_script_macro_f1 | 3 | 1.4 | ok |
| E2 laya c10 | ood_brand_macro_f1 | 3 | 7.0 | INVESTIGATE |
| E3 laya_ml c10 | val_macro_f1 | 3 | 0.9 | ok |
| E3 laya_ml c10 | test_id_macro_f1 | 3 | 0.7 | ok |
| E3 laya_ml c10 | ood_country_macro_f1 | 3 | 1.0 | ok |
| E3 laya_ml c10 | ood_script_macro_f1 | 3 | 0.7 | ok |
| E3 laya_ml c10 | ood_brand_macro_f1 | 3 | 5.5 | INVESTIGATE |

## Learning curve (E6: train subsets + the full-size run of the same seed)

| Curve | n (train rows) | Run | val macro-F1 | test_id macro-F1 |
|---|---|---|---|---|
| E6 laya c10 s11 | 1000 | fsq-c10-E6-laya-s11-n1000 | 0.4484 | 0.4481 |
| E6 laya c10 s11 | 3000 | fsq-c10-E6-laya-s11-n3000 | 0.4779 | 0.4736 |
| E6 laya c10 s11 | 10000 | fsq-c10-E6-laya-s11-n10000 | 0.5278 | 0.5249 |
| E6 laya c10 s11 | 26750 (full) | fsq-c10-E2-laya-s11 | 0.5658 | 0.5602 |

## Runs

| Run | Card | Epochs | Stop | Best opt | T | val macro-F1 | test_id macro-F1 | Order inv. | Train (min) | Run (min) | export_check |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fsq-c10-B2-laya-zs | G4 | n/a | n/a | n/a | 1.0000 | 0.3025 | 0.3126 | 0.7104 | n/a | 0.9 | - |
| fsq-c10-B2-laya_ml-zs | G4 | n/a | n/a | n/a | 1.0000 | 0.2813 | 0.2823 | 0.7586 | n/a | 0.5 | - |
| fsq-c10-E2-laya-s11 | G4 | 4.00 | epochs | 3000 | 1.1037 | 0.5658 | 0.5602 | 0.8860 | 8.4 | 9.7 | passed |
| fsq-c10-E2-laya-s22 | G4 | 4.00 | epochs | 3000 | 1.0662 | 0.5706 | 0.5688 | 0.9008 | 8.3 | 9.6 | passed |
| fsq-c10-E2-laya-s33 | G4 | 3.29 | early_stop | 2000 | 1.0274 | 0.5617 | 0.5626 | 0.8964 | 6.9 | 8.2 | passed |
| fsq-c10-E3-laya_ml-s11 | G4 | 2.99 | early_stop | 1750 | 0.9134 | 0.5875 | 0.5842 | 0.9296 | 3.4 | 4.4 | passed |
| fsq-c10-E3-laya_ml-s22 | G4 | 3.59 | early_stop | 2250 | 0.9661 | 0.5842 | 0.5897 | 0.9188 | 4.2 | 5.2 | passed |
| fsq-c10-E3-laya_ml-s33 | G4 | 3.89 | early_stop | 2500 | 0.9893 | 0.5781 | 0.5915 | 0.9248 | 4.5 | 5.5 | passed |
| fsq-c10-E4-laya-s11-head | G4 | 2.09 | early_stop | 1000 | 0.8780 | 0.3099 | 0.3177 | 0.7062 | 1.9 | 3.3 | passed |
| fsq-c10-E6-laya-s11-n1000 | G4 | 3.88 | early_stop | 99 | 1.1068 | 0.4484 | 0.4481 | 0.7954 | 1.0 | 2.2 | passed |
| fsq-c10-E6-laya-s11-n10000 | G4 | 4.00 | epochs | 1110 | 1.0940 | 0.5278 | 0.5249 | 0.8784 | 3.8 | 5.0 | passed |
| fsq-c10-E6-laya-s11-n3000 | G4 | 3.59 | early_stop | 264 | 1.0751 | 0.4779 | 0.4736 | 0.8444 | 1.5 | 2.6 | passed |
| fsq-c7-E5-laya-s11 | G4 | 3.89 | early_stop | 2500 | 1.0149 | 0.6191 | 0.6235 | 0.8964 | 6.7 | 7.8 | passed |
| fsq-c7-E5-laya_ml-s11 | G4 | 2.69 | early_stop | 1500 | 0.9530 | 0.6332 | 0.6348 | 0.9270 | 2.7 | 3.5 | passed |

## Phase 3 exit check

Criterion: every configured run has a best/ checkpoint (trained runs), a temperature and val metrics in results/runs.csv (design §6.2 Phase 3 exit).

Verdict: **PASS** (14/14 runs complete)

| Run | done.json | best/ | T | val metrics | Result |
|---|---|---|---|---|---|
| fsq-c10-B2-laya-zs | yes | n/a | yes | yes | ok |
| fsq-c10-B2-laya_ml-zs | yes | n/a | yes | yes | ok |
| fsq-c10-E2-laya-s11 | yes | yes | yes | yes | ok |
| fsq-c10-E2-laya-s22 | yes | yes | yes | yes | ok |
| fsq-c10-E2-laya-s33 | yes | yes | yes | yes | ok |
| fsq-c10-E3-laya_ml-s11 | yes | yes | yes | yes | ok |
| fsq-c10-E3-laya_ml-s22 | yes | yes | yes | yes | ok |
| fsq-c10-E3-laya_ml-s33 | yes | yes | yes | yes | ok |
| fsq-c10-E4-laya-s11-head | yes | yes | yes | yes | ok |
| fsq-c7-E5-laya-s11 | yes | yes | yes | yes | ok |
| fsq-c7-E5-laya_ml-s11 | yes | yes | yes | yes | ok |
| fsq-c10-E6-laya-s11-n1000 | yes | yes | yes | yes | ok |
| fsq-c10-E6-laya-s11-n3000 | yes | yes | yes | yes | ok |
| fsq-c10-E6-laya-s11-n10000 | yes | yes | yes | yes | ok |


## Early read against the decision rule (§5.12) — not the Phase 5 verdict

Added 2026-09-27. B3 (char TF-IDF + LR) was scored on the OOD pools locally on the same frozen data
(`python -m laya_poc.baselines --splits val test_id ood_country ood_script ood_brand`): c10 macro-F1 val 0.4892,
test_id 0.4866, ood_country 0.3425, ood_script 0.2786, ood_brand 0.4493; post-T ECE test_id 0.0223, ood_country
0.0531, ood_script 0.1425, ood_brand 0.0401. Post-T ECE of the Laya arms on the OOD pools (mean over seeds) was
computed from the per-split eval JSONs.

| §5.12 criterion | `laya` (E2) | `laya-multilingual` (E3) | Still needed |
|---|---|---|---|
| 1. OOD macro-F1 >= both trained baselines + 3 pts (OOD average; `laya` excludes ood_script) | 0.509 vs TF-IDF 0.396: +11.3 pts | 0.517 vs TF-IDF 0.357: +16.0 pts | B4 fine-tuned small encoder (Phase 4) |
| 2. ID->OOD gap <= 5 pts on each included pool | FAIL: country 13.4, brand -2.4 | FAIL: country 9.5, script 8.3, brand 3.7 | — (TF-IDF's gaps: 14.4 / 20.8 / 3.7) |
| 3. post-T ECE <= 0.05 on test_id and the OOD average | FAIL: 0.062 / 0.096 | test_id PASS 0.044; OOD FAIL 0.070 | — |
| 4. trap accuracy >= best baseline | pending | pending | trap annotation (data/trap_candidates.csv) |
| 5. CPU p95 <= 500 ms and >= 8 rec/s (4 vCPU) | FAIL on the laptop: 1184 ms, 1.0 rec/s | 403 ms PASS, 2.7 rec/s FAIL | ONNX + 4-8 vCPU run (Phase 5) |

Other findings: head-only (E4) is no better than zero-shot (0.310 vs 0.303 val), so full fine-tuning is essential;
the learning curve is still rising (1k 0.448, 3k 0.478, 10k 0.528, 26.75k 0.566 val macro-F1); the 7-class
collapse gains ~5 pts; field-order invariance is 0.89 (`laya`) / 0.92 (`laya-multilingual`) against the §5.11
expectation of >= 0.99; tau for 95 % answered accuracy covers only ~24 % of val, so the 80 %-coverage fallback
applies (answered accuracy ~0.69); trained models abstain on 100 % of stripped (no-evidence) records with no
false-confident answers; ood_brand seed range 7.0 / 5.5 pts (brand rows are clustered on ~200 names).
