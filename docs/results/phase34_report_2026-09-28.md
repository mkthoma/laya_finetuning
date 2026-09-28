<!-- Merged locally on 2026-09-28 from the Phase 3 (2026-09-27) and Phase 4 (2026-09-28) Colab G4 archives:
     python -m laya_poc.matrix report --runs-root runs/p3_colab/runs/p3 --runs-root runs/p4_colab/runs/p4 --archived
     Metrics only: no FSQ records. Per-run table: docs/results/phase34_runs_2026-09-28.csv.
     Decision criterion 1 below is an EARLY READ: Phase 5 applies the full §5.12 decision rule. -->

# Laya PoC: Phase 3 seed matrix and Phase 4 baselines

Generated 2026-09-28T07:23:00+00:00; card(s) G4; 27 finished of 27 configured runs. Phase 3 exit check: **PASS** (14/14). Phase 4 exit check: **PASS** (13/13).

Post-T metrics unless marked pre; groups of several seeds show `mean (min to max)` over seeds. B2 rows are the zero-shot hub checkpoints (shipped temperature). B1/B3/B4/B5 rows are the Phase 4 baselines (majority/prior, char TF-IDF+LR, fine-tuned small encoders, the optional reference LLM on evaluation subsets). Metrics only: no FSQ rows.

## Accuracy: macro-F1 (post-T)

| Group | Runs | Seeds | val | val (9-class) | test_id | ood_country | ood_script | ood_brand |
|---|---|---|---|---|---|---|---|---|
| B1 majority c10 | 1 | - | 0.0187 | 0.0208 | 0.0205 | 0.0191 | 0.0197 | 0.0191 |
| B1 majority c7 | 1 | - | 0.0527 | n/a | 0.0493 | 0.0522 | 0.0513 | 0.0566 |
| B1 prior c10 | 1 | - | 0.0187 | 0.0208 | 0.0205 | 0.0191 | 0.0197 | 0.0191 |
| B1 prior c7 | 1 | - | 0.0527 | n/a | 0.0493 | 0.0522 | 0.0513 | 0.0566 |
| B2 laya c10 (zero-shot, shipped T) | 1 | - | 0.3025 | 0.3119 | 0.3126 | 0.2393 | 0.2331 | 0.2345 |
| B2 laya_ml c10 (zero-shot, shipped T) | 1 | - | 0.2813 | 0.2951 | 0.2823 | 0.2484 | 0.2732 | 0.2078 |
| B3 tfidf_lr c10 | 1 | - | 0.4892 | 0.5193 | 0.4866 | 0.3425 | 0.2786 | 0.4493 |
| B3 tfidf_lr c7 | 1 | - | 0.5319 | n/a | 0.5433 | 0.3939 | 0.3039 | 0.5385 |
| B4 mmbert_small c10 | 3 | 11, 22, 33 | 0.5964 (0.5910 to 0.6021) | 0.6246 (0.6224 to 0.6275) | 0.5959 (0.5858 to 0.6047) | 0.5022 (0.4999 to 0.5039) | 0.4826 (0.4766 to 0.4891) | 0.5783 (0.5652 to 0.5876) |
| B4 modernbert_base c10 | 3 | 11, 22, 33 | 0.5445 (0.5398 to 0.5491) | 0.5731 (0.5693 to 0.5773) | 0.5598 (0.5570 to 0.5627) | 0.4118 (0.4083 to 0.4140) | 0.3270 (0.3136 to 0.3352) | 0.5211 (0.4982 to 0.5408) |
| B5 qwen3_4b c10 (zero-shot reference, eval subsets) | 1 | - | 0.4751 | 0.4952 | 0.5022 | 0.4486 | 0.4627 | 0.4006 |
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
| B1 majority c10 | 0.0014 | 0.0008 | 0.0014 |
| B1 majority c7 | -0.0029 | -0.0020 | -0.0073 |
| B1 prior c10 | 0.0014 | 0.0008 | 0.0014 |
| B1 prior c7 | -0.0029 | -0.0020 | -0.0073 |
| B2 laya c10 (zero-shot, shipped T) | 0.0733 | 0.0795 (excluded) | 0.0781 |
| B2 laya_ml c10 (zero-shot, shipped T) | 0.0338 | 0.0090 | 0.0744 |
| B3 tfidf_lr c10 | 0.1441 | 0.2080 | 0.0373 |
| B3 tfidf_lr c7 | 0.1494 | 0.2393 | 0.0048 |
| B4 mmbert_small c10 | 0.0936 (0.0829 to 0.1008) | 0.1133 (0.1080 to 0.1227) | 0.0176 (-0.0018 to 0.0394) |
| B4 modernbert_base c10 | 0.1480 (0.1455 to 0.1496) | 0.2328 (0.2243 to 0.2434) | 0.0387 (0.0163 to 0.0646) |
| B5 qwen3_4b c10 (zero-shot reference, eval subsets) | 0.0536 | 0.0396 | 0.1017 |
| E2 laya c10 | 0.1339 (0.1252 to 0.1456) | 0.2395 (0.2369 to 0.2426) (excluded) | -0.0242 (-0.0659 to 0.0060) |
| E3 laya_ml c10 | 0.0945 (0.0914 to 0.0979) | 0.0827 (0.0742 to 0.0876) | 0.0374 (0.0171 to 0.0662) |
| E4 laya c10 head-only | 0.0783 | 0.0834 (excluded) | 0.0808 |
| E5 laya c7 | 0.1534 | 0.2420 (excluded) | 0.0008 |
| E5 laya_ml c7 | 0.0921 | 0.0519 | 0.0113 |
| E6 laya c10 n=1000 | 0.1038 | 0.1604 (excluded) | 0.0248 |
| E6 laya c10 n=3000 | 0.1018 | 0.1829 (excluded) | -0.0071 |
| E6 laya c10 n=10000 | 0.1252 | 0.1971 (excluded) | 0.0095 |

## Calibration: T and ECE pre -> post

T: fitted on val by export_check for trained runs; the shipped temperature for zero-shot. ECE: 15 equal-width bins on answer_confidence, before and after the temperature. Baselines: T from calibration.json, fitted on val (B1 majority/prior: T = 1; B5 on its 500 val rows).

| Group | T | val ECE pre -> post | test_id ECE pre -> post |
|---|---|---|---|
| B1 majority c10 | 1.0000 | 0.8970 -> 0.8970 | 0.8857 -> 0.8857 |
| B1 majority c7 | 1.0000 | 0.7737 -> 0.7737 | 0.7913 -> 0.7913 |
| B1 prior c10 | 1.0000 | 0.0019 -> 0.0019 | 0.0133 -> 0.0133 |
| B1 prior c7 | 1.0000 | 0.0261 -> 0.0261 | 0.0084 -> 0.0084 |
| B2 laya c10 (zero-shot, shipped T) | 1.0000 | 0.1237 -> 0.1237 | 0.1212 -> 0.1212 |
| B2 laya_ml c10 (zero-shot, shipped T) | 1.0000 | 0.3513 -> 0.3513 | 0.3449 -> 0.3449 |
| B3 tfidf_lr c10 | 0.9727 | 0.0299 -> 0.0198 | 0.0214 -> 0.0223 |
| B3 tfidf_lr c7 | 0.8612 | 0.0490 -> 0.0211 | 0.0601 -> 0.0193 |
| B4 mmbert_small c10 | 0.8907 (0.8671 to 0.9057) | 0.0385 (0.0332 to 0.0441) -> 0.0289 (0.0258 to 0.0330) | 0.0346 (0.0309 to 0.0401) -> 0.0264 (0.0179 to 0.0322) |
| B4 modernbert_base c10 | 0.9980 (0.9866 to 1.0135) | 0.0405 (0.0374 to 0.0424) -> 0.0385 (0.0375 to 0.0397) | 0.0278 (0.0240 to 0.0338) -> 0.0278 (0.0250 to 0.0318) |
| B5 qwen3_4b c10 (zero-shot reference, eval subsets) | 9.2635 | 0.4883 -> 0.0755 | 0.4483 -> 0.0614 |
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
| B1 majority c10 | 0.1029 | 0.0000 | 1.0000 | n/a | n/a |
| B1 majority c7 | 0.1571 | 0.0000 | 1.0000 | n/a | n/a |
| B1 prior c10 | 0.1029 | 0.0000 | 0.0000 | n/a | n/a |
| B1 prior c7 | 0.1571 | 0.0000 | 0.0000 | n/a | n/a |
| B2 laya c10 (zero-shot, shipped T) | 0.1914 | 0.3380 | 0.0000 | 0.7104 | no |
| B2 laya_ml c10 (zero-shot, shipped T) | 0.1600 | 0.3020 | 0.2480 | 0.7586 | no |
| B3 tfidf_lr c10 | 0.3129 | 0.0940 | 0.0000 | 0.8816 | no |
| B3 tfidf_lr c7 | 0.3500 | 0.2140 | 0.0000 | 0.9072 | no |
| B4 mmbert_small c10 | 0.3695 (0.3657 to 0.3729) | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) | 0.9507 (0.9450 to 0.9568) | no |
| B4 modernbert_base c10 | 0.3657 (0.3643 to 0.3671) | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) | 0.8991 (0.8934 to 0.9026) | no |
| B5 qwen3_4b c10 (zero-shot reference, eval subsets) | 0.2586 | 1.0000 | 0.0000 | n/a | n/a |
| E2 laya c10 | 0.3819 (0.3657 to 0.3929) | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) | 0.8944 (0.8860 to 0.9008) | no |
| E3 laya_ml c10 | 0.3662 (0.3629 to 0.3686) | 1.0000 (1.0000 to 1.0000) | 0.0000 (0.0000 to 0.0000) | 0.9244 (0.9188 to 0.9296) | no |
| E4 laya c10 head-only | 0.1971 | 0.5340 | 0.0000 | 0.7062 | no |
| E5 laya c7 | 0.4143 | 1.0000 | 0.0000 | 0.8964 | no |
| E5 laya_ml c7 | 0.4200 | 1.0000 | 0.0000 | 0.9270 | no |
| E6 laya c10 n=1000 | 0.2671 | 1.0000 | 0.0000 | 0.7954 | no |
| E6 laya c10 n=3000 | 0.2971 | 1.0000 | 0.0000 | 0.8444 | no |
| E6 laya c10 n=10000 | 0.3671 | 1.0000 | 0.0000 | 0.8784 | no |

## Seed variance (design §5.12: Investigate when a macro-F1 range over seeds exceeds 3 points)

Flagged: B4 modernbert_base c10, E2 laya c10, E3 laya_ml c10.

| Group | Metric | Seeds | Range (points) | Flag |
|---|---|---|---|---|
| B4 mmbert_small c10 | val_macro_f1 | 3 | 1.1 | ok |
| B4 mmbert_small c10 | test_id_macro_f1 | 3 | 1.9 | ok |
| B4 mmbert_small c10 | ood_country_macro_f1 | 3 | 0.4 | ok |
| B4 mmbert_small c10 | ood_script_macro_f1 | 3 | 1.3 | ok |
| B4 mmbert_small c10 | ood_brand_macro_f1 | 3 | 2.2 | ok |
| B4 modernbert_base c10 | val_macro_f1 | 3 | 0.9 | ok |
| B4 modernbert_base c10 | test_id_macro_f1 | 3 | 0.6 | ok |
| B4 modernbert_base c10 | ood_country_macro_f1 | 3 | 0.6 | ok |
| B4 modernbert_base c10 | ood_script_macro_f1 | 3 | 2.2 | ok |
| B4 modernbert_base c10 | ood_brand_macro_f1 | 3 | 4.3 | INVESTIGATE |
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

## Decision criterion 1 (early read)

Information only: Phase 5 applies the decision rule (design §5.12). Criterion 1: on OOD, a fine-tuned Laya checkpoint (mean over its seeds) must beat BOTH char TF-IDF+LR (B3) and the fine-tuned small encoder (B4) by >= 3 macro-F1 points, averaged over the OOD pools. `laya` excludes ood_script (it cannot read Thai by design) and every baseline is averaged over the same pools as the arm it is compared with. The best seed-mean B3/B4 group of the same scheme is the bar (a +1.5 to +3 point lead is the §5.12 Investigate band). Zero-shot (B2), head-only (E4) and train-subset (E6) runs are not candidates.

| Laya arm | Seeds | OOD pools | Laya OOD avg | Best trained baseline | Its OOD avg | Lead (points) | >= 3 points | Compared (seed-mean OOD avg) |
|---|---|---|---|---|---|---|---|---|
| E2 laya c10 | 11, 22, 33 | ood_country, ood_brand | 0.5090 (0.4942 to 0.5305) | B4 mmbert_small c10 | 0.5402 | -3.1 | FAIL | B3 tfidf_lr c10 0.3959; B4 mmbert_small c10 0.5402; B4 modernbert_base c10 0.4665 |
| E3 laya_ml c10 | 11, 22, 33 | ood_country, ood_script, ood_brand | 0.5170 (0.5061 to 0.5226) | B4 mmbert_small c10 | 0.5210 | -0.4 | FAIL | B3 tfidf_lr c10 0.3568; B4 mmbert_small c10 0.5210; B4 modernbert_base c10 0.4200 |
| E5 laya c7 | 11 | ood_country, ood_brand | 0.5464 | B3 tfidf_lr c7 | 0.4662 | +8.0 | PASS (incomplete: no B4 c7) | B3 tfidf_lr c7 0.4662 |
| E5 laya_ml c7 | 11 | ood_country, ood_script, ood_brand | 0.5830 | B3 tfidf_lr c7 | 0.4121 | +17.1 | PASS (incomplete: no B4 c7) | B3 tfidf_lr c7 0.4121 |

## Runs

| Run | Card | Epochs | Stop | Best opt | T | val macro-F1 | test_id macro-F1 | Order inv. | Train (min) | Run (min) | export_check |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fsq-c10-B1-majority | G4 | n/a | n/a | n/a | 1.0000 | 0.0187 | 0.0205 | n/a | n/a | 0.0 | - |
| fsq-c10-B1-prior | G4 | n/a | n/a | n/a | 1.0000 | 0.0187 | 0.0205 | n/a | n/a | 0.0 | - |
| fsq-c10-B2-laya-zs | G4 | n/a | n/a | n/a | 1.0000 | 0.3025 | 0.3126 | 0.7104 | n/a | 0.9 | - |
| fsq-c10-B2-laya_ml-zs | G4 | n/a | n/a | n/a | 1.0000 | 0.2813 | 0.2823 | 0.7586 | n/a | 0.5 | - |
| fsq-c10-B3-tfidf_lr | G4 | n/a | n/a | n/a | 0.9727 | 0.4892 | 0.4866 | 0.8816 | n/a | 0.5 | - |
| fsq-c10-B4-mmbert_small-s11 | G4 | 3.00 | epochs | 1672 | 0.8992 | 0.5910 | 0.5858 | 0.9450 | 1.3 | 1.5 | - |
| fsq-c10-B4-mmbert_small-s22 | G4 | 3.00 | epochs | 1672 | 0.9057 | 0.6021 | 0.6047 | 0.9568 | 1.3 | 1.5 | - |
| fsq-c10-B4-mmbert_small-s33 | G4 | 3.00 | epochs | 1672 | 0.8671 | 0.5961 | 0.5971 | 0.9502 | 1.3 | 1.5 | - |
| fsq-c10-B4-modernbert_base-s11 | G4 | 3.00 | epochs | 2508 | 0.9940 | 0.5398 | 0.5596 | 0.8934 | 1.4 | 1.6 | - |
| fsq-c10-B4-modernbert_base-s22 | G4 | 3.00 | epochs | 2508 | 0.9866 | 0.5446 | 0.5627 | 0.9012 | 1.4 | 1.5 | - |
| fsq-c10-B4-modernbert_base-s33 | G4 | 3.00 | epochs | 2508 | 1.0135 | 0.5491 | 0.5570 | 0.9026 | 1.3 | 1.5 | - |
| fsq-c10-B5-qwen3_4b | G4 | n/a | n/a | n/a | 9.2635 CLAMPED | 0.4751 | 0.5022 | n/a | n/a | 13.7 | - |
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
| fsq-c7-B1-majority | G4 | n/a | n/a | n/a | 1.0000 | 0.0527 | 0.0493 | n/a | n/a | 0.0 | - |
| fsq-c7-B1-prior | G4 | n/a | n/a | n/a | 1.0000 | 0.0527 | 0.0493 | n/a | n/a | 0.0 | - |
| fsq-c7-B3-tfidf_lr | G4 | n/a | n/a | n/a | 0.8612 | 0.5319 | 0.5433 | 0.9072 | n/a | 0.5 | - |
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

## Phase 4 exit check

Criterion: every configured baseline run (phase4.arms) has predictions for every eval split in preds/ (B5: its evaluation subsets); an optional run that was not run is not a failure (design §6.2 Phase 4 exit). B2 (zero-shot Laya) ran in Phase 3.

Verdict: **PASS** (13/13 baseline runs complete)

| Run | Kind | done.json | preds (every split) | Result |
|---|---|---|---|---|
| fsq-c10-B1-majority | majority | yes | yes | ok |
| fsq-c10-B1-prior | prior | yes | yes | ok |
| fsq-c7-B1-majority | majority | yes | yes | ok |
| fsq-c7-B1-prior | prior | yes | yes | ok |
| fsq-c10-B3-tfidf_lr | tfidf_lr | yes | yes | ok |
| fsq-c7-B3-tfidf_lr | tfidf_lr | yes | yes | ok |
| fsq-c10-B4-modernbert_base-s11 | small_encoder | yes | yes | ok |
| fsq-c10-B4-modernbert_base-s22 | small_encoder | yes | yes | ok |
| fsq-c10-B4-modernbert_base-s33 | small_encoder | yes | yes | ok |
| fsq-c10-B4-mmbert_small-s11 | small_encoder | yes | yes | ok |
| fsq-c10-B4-mmbert_small-s22 | small_encoder | yes | yes | ok |
| fsq-c10-B4-mmbert_small-s33 | small_encoder | yes | yes | ok |
| fsq-c10-B5-qwen3_4b | llm (optional) | yes | yes | ok |

