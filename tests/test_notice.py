"""Attribution texts for FSQ-derived data and fine-tuned Laya checkpoints (design doc Appendix D)."""
import hashlib

from laya_poc import notice as N

# sha256 of NOTICE.txt at the root of the foursquare/fsq-os-places dataset repo (582 bytes, research/fsq-data.md §6)
FSQ_NOTICE_SHA256 = "21454edd7305f3f84444f7cd99644cf0080854eab094720c0f8da35ffe501d1a"


def test_fsq_notice_constant_is_the_verbatim_dataset_notice():
    raw = N.FSQ_NOTICE.encode("utf-8")
    assert len(raw) == 582
    assert hashlib.sha256(raw).hexdigest() == FSQ_NOTICE_SHA256


def test_fsq_notice_appends_the_modification_statement():
    text = N.fsq_notice("2026-09-15")
    assert text.startswith(N.FSQ_NOTICE)
    assert text.endswith("\n\nDerived from Foursquare Open Source Places release 2026-09-15; modified: fields removed, "
                         "records filtered and sampled.\n")


def test_checkpoint_notice_names_base_licence_and_data():
    text = N.checkpoint_notice(model_name="laya-poc-laya-s11", base_repo="convaiinnovations/laya",
                               base_revision="55cf4c4e", laya_commit="4066d5d5")
    assert "laya-poc-laya-s11" in text
    assert "convaiinnovations/laya" in text and "55cf4c4e" in text
    assert "modified: fine-tuned" in text
    assert "Apache-2.0" in text and "https://github.com/NandhaKishorM/laya/blob/4066d5d5/LICENSE" in text
    assert "trained on data derived from FSQ OS Places (see NOTICE_FSQ.txt)" in text


def test_checkpoint_notice_without_provenance_says_so():
    text = N.checkpoint_notice(model_name="m", laya_commit=None)
    assert "not recorded" in text
    assert "https://github.com/NandhaKishorM/laya/blob/main/LICENSE" in text
