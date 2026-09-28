import sys

import pytest

from laya_poc import fallback_guard as G

# Laya's own wording (laya/agent.py Agent._infer and Agent._restore_runtime at the pinned commit)
LAYA_OOM = "Warning: GPU memory exceeded during inference. Retrying this request on CPU..."
LAYA_STUCK = ("Warning: could not move the model back to cuda:0 after the CPU retry (CUDA out of memory); "
              "staying on CPU.")


def test_guard_flags_the_oom_message_and_still_echoes_it(capsys):
    guard = G.FallbackGuard()
    with guard:
        print("progress line")
        print(LAYA_OOM)
    assert guard.cpu_fallback is True and guard.messages == [LAYA_OOM]
    out = capsys.readouterr().out
    assert "progress line" in out and LAYA_OOM in out  # the tee still shows everything in the cell


def test_guard_stays_false_without_the_message(capsys):
    guard = G.FallbackGuard()
    with guard:
        print("Warning: laya fast path unavailable (no tilelang); using the stock forward.")
    assert guard.cpu_fallback is False and guard.messages == []


def test_guard_flags_a_failed_move_back_to_the_gpu(capsys):
    guard = G.FallbackGuard()
    with guard:
        print(LAYA_STUCK)
    assert guard.cpu_fallback is True


def test_guard_sees_a_message_split_across_writes_without_a_newline(capsys):
    guard = G.FallbackGuard()
    with guard:
        sys.stdout.write("Warning: GPU memory ex")
        sys.stdout.write("ceeded during inference.")
    assert guard.cpu_fallback is True


def test_guard_is_sticky_across_calls_and_restores_stdout(capsys):
    guard, before = G.FallbackGuard(), sys.stdout
    with guard:
        print(LAYA_OOM)
    assert sys.stdout is before
    with guard:
        print("a clean later call")
    assert guard.cpu_fallback is True and sys.stdout is before


def test_guard_scans_and_restores_stdout_when_the_call_raises(capsys):
    guard, before = G.FallbackGuard(), sys.stdout
    with pytest.raises(RuntimeError, match="boom"):
        with guard:
            print(LAYA_OOM)
            raise RuntimeError("boom")
    assert guard.cpu_fallback is True and sys.stdout is before


def test_guard_refuses_nesting(capsys):
    guard = G.FallbackGuard()
    with guard:
        with pytest.raises(RuntimeError, match="already"):
            guard.__enter__()


class _OomOnGpu:
    """Stands in for agent.model: a CUDA OOM for any input on a non-CPU device, the real model on CPU."""

    def __init__(self, model):
        self.model = model

    def __call__(self, *tensors):
        import torch
        if any(t.device.type != "cpu" for t in tensors):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return self.model(*tensors)

    def to(self, *args, **kwargs):  # Laya moves the model to CPU and back; the real weights stay put
        return self


@pytest.mark.torch
def test_guard_catches_the_real_laya_fallback(tiny_ckpt_dir, synthetic_data_dir, capsys):
    """Drive Laya's own Agent._infer OOM branch without a GPU: the meta device stands in for cuda."""
    torch = pytest.importorskip("torch")
    from laya_poc import hub
    from laya_poc import labels as L
    from laya_poc.io_utils import read_jsonl

    agent = hub.load_agent(tiny_ckpt_dir, device="cpu")
    agent.model, agent.device = _OomOnGpu(agent.model), torch.device("meta")
    states = [r["state"] for r in read_jsonl(synthetic_data_dir / "val.jsonl")[:6]]
    guard = G.FallbackGuard()
    with guard:
        out = agent.predict_batch(states, L.question("c10"), batch_size=4)
    assert len(out) == 6 and guard.cpu_fallback is True
    assert agent.device.type == "meta"  # Laya restored the device: agent.device alone cannot tell
    assert "GPU memory exceeded" in capsys.readouterr().out
