# Early CPU benchmark (laptop, 2026-09-26)

Information only. The decision-rule CPU budget (design doc §5.12 criterion 5: p95 <= 500 ms at batch 1 and >= 8 records/s
batched, 4 vCPUs, fp32, faster of ONNX or PyTorch) is judged in Phase 5 on a 4-8 vCPU cloud machine, with ONNX and an
8-thread run. Latency depends on the architecture, not the fine-tuned weights, so the pinned base checkpoints were used.

Setup: `python -m laya_poc.bench_cpu --ckpt hub --model <m> --rows data/test_id.jsonl --n 60 --warmup 10 --threads 4`;
Windows laptop, 8 logical CPUs, PyTorch fp32 (no ONNX), 60 test_id records (string states), batched run
`predict_batch(sort_by_length=True, batch_size=32)`.

| Model | Threads | Cold start (s) | p50 (ms) | p95 (ms) | Batch rec/s | Peak RSS (GB) | p95 <= 500 ms | >= 8 rec/s |
|---|---|---|---|---|---|---|---|---|
| `laya` (ModernBERT-large, 421M) | 4 | 38.8 | 1013 | 1184 | 1.0 | 2.80 | no | no |
| `laya-multilingual` (mmBERT-base, 322M) | 4 | 11.7 | 342 | 403 | 2.7 | 2.30 | yes | no |

For comparison, the E1 fine-tuned `laya` on the Colab G4 host (PyTorch, 200 records): 1 thread p95 1851 ms, 0.6 rec/s;
2 threads p95 1092 ms, 1.2 rec/s (docs/results/phase2_gate_report_2026-09-26.md).

Reading:
- `laya` misses both budgets and is above the doc's stop line (p95 > 1000 ms), which applies only after ONNX on 8 threads,
  so this is a warning, not a verdict.
- `laya-multilingual` meets the latency budget but not the throughput floor.
- Batching gives no throughput gain on CPU here (batched rec/s <= 1 / p50), contrary to the doc's assumption that
  "batching recovers at least a modest multiple". This is the main risk for criterion 5.
