import json

import pytest

from laya_poc import labels as L

pytest.importorskip("laya")
from laya_poc import items as I  # noqa: E402

pytestmark = pytest.mark.torch


def _row(state: dict, label: str = "dining", rid: str = "r1") -> dict:
    return {"id": rid, "split": "train",
            "state": json.dumps(state, ensure_ascii=False, separators=(",", ":")),
            "questions": json.dumps(L.question("c10"), ensure_ascii=False),
            "gold": json.dumps(L.gold(label, "c10", 0.1), ensure_ascii=False)}


def test_build_items_matches_runtime_encoding(en_tokenizer):
    from laya.agent import Agent
    row = _row({"country": "GB", "name": "Rosa's Trattoria", "tel": "0113 000 0000"})
    items = I.build_items(en_tokenizer, row, 512, 192)
    assert len(items) == 1
    it = items[0]
    assert len(it["markers"]) == 10 and it["qtype"] == 0
    assert it["target"][L.option_keys("c10").index("dining")] == pytest.approx(0.9)
    assert it["label"] == L.option_keys("c10").index("dining")
    # Same ids as Agent._encode_state on the string state (what predict_batch will see).
    agent = Agent.__new__(Agent)
    agent.tok, agent.cfg = en_tokenizer, {"max_len": 512, "head_max_len": 192}
    q = L.question("c10")
    internal = {k: Agent._to_internal(v) for k, v in q.items()}
    runtime = agent._encode_state(row["state"], list(q), internal)[0]
    assert runtime["ids"] == it["ids"] and runtime["markers"] == it["markers"]


def test_dict_state_would_tokenise_differently(en_tokenizer):
    row = _row({"country": "GB", "name": "Rosa's Trattoria", "tel": "0113 000 0000"})
    compact = I.count_state_tokens(en_tokenizer, row["state"])
    spaced = I.count_state_tokens(en_tokenizer, json.dumps(json.loads(row["state"]), ensure_ascii=False))
    assert spaced > compact


def test_build_items_rejects_dict_state(en_tokenizer):
    row = _row({"name": "x"})
    row["state"] = json.loads(row["state"])
    with pytest.raises(TypeError):
        I.build_items(en_tokenizer, row, 512, 192)


def test_build_items_rejects_gold_key_mismatch(en_tokenizer):
    row = _row({"name": "x"})
    row["gold"] = json.dumps(L.gold("culture", "c7", 0.1))
    with pytest.raises(ValueError, match="gold keys"):
        I.build_items(en_tokenizer, row, 512, 192)


@pytest.mark.parametrize("scheme,hml", [("c10", 192), ("c10", 256), ("c7", 192)])
def test_question_options_are_never_trimmed(en_tokenizer, scheme, hml):
    q = I.internal_question(L.QUESTION_NAME, L.question(scheme)[L.QUESTION_NAME])
    total = I.assert_options_untrimmed(en_tokenizer, q, hml)
    assert total < hml - 16


def test_trimming_detected_for_tiny_head_budget(en_tokenizer):
    q = I.internal_question(L.QUESTION_NAME, L.question("c10")[L.QUESTION_NAME])
    with pytest.raises(ValueError, match="trim"):
        I.assert_options_untrimmed(en_tokenizer, q, 100)


def test_state_room_is_exact(en_tokenizer):
    from laya.common import build_sequence
    q = I.internal_question(L.QUESTION_NAME, L.question("c10")[L.QUESTION_NAME])
    room = I.state_room(en_tokenizer, q, 512, 192)
    assert 300 < room < 512 - 16
    long_state = json.dumps({"name": "word " * 600})
    seq, _ = build_sequence(en_tokenizer, long_state, q, 512, 192)
    assert len(seq) == 512  # a state longer than the room fills the sequence exactly
    n = I.count_state_tokens(en_tokenizer, long_state)
    assert n > room


def test_description_token_counts_are_known(en_tokenizer):
    # The doc says "<=10 tokens"; with the ModernBERT tokenizer several are longer (dining 15,
    # arts 14). What matters is that options are never trimmed, asserted above; this pins the
    # measured counts so a description edit that bloats them is noticed.
    counts = {k: len(en_tokenizer(v, add_special_tokens=False)["input_ids"]) for k, v in L.criteria("c10").items()}
    assert max(counts.values()) <= 15, counts
    assert counts["dining"] == 15 and counts["arts"] == 14
