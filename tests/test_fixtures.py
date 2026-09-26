import json

import pytest

from laya_poc import labels as L

pytestmark = pytest.mark.torch


def test_tiny_checkpoint_loads_and_predicts(tiny_ckpt_dir):
    import laya
    agent = laya.load(str(tiny_ckpt_dir), device="cpu")
    state = json.dumps({"country": "GB", "name": "Rosa's Trattoria"}, separators=(",", ":"))
    out = agent.predict_batch([state, state], L.question("c10"), batch_size=2)
    ans = out[0]["answers"][L.QUESTION_NAME]
    assert set(ans["probabilities"]) == set(L.option_keys("c10"))
    assert 0.0 <= ans["answer_confidence"] <= 1.0
    assert all(p.dtype.is_floating_point for p in agent.model.parameters())
