import json
from collections import Counter

import pytest

import cafd.target_path_prepare as prepare


def rows(counts):
    return [dict(id=f"{family}-{i:03}",problem_family=family)
            for family,count in counts.items() for i in range(count)]


def test_fixed_cohort_split_5_13_14_has_no_question_leakage():
    available=rows(dict(contains_count=5,contains_ordered=19,contains_substring=21))
    quotas=dict(contains_count=5,contains_ordered=13,contains_substring=14)
    first=prepare.stratified_tasks(available,quotas=quotas)
    assert first==prepare.stratified_tasks(list(reversed(available)),quotas=quotas)
    assert len(first)==32
    assert Counter(r["split"] for r in first)=={"calibration":16,"heldout":16}
    assert Counter(r["family"] for r in first)==quotas
    assert len({r["row_id"] for r in first})==32


@pytest.mark.parametrize("length",[1,2,3,15,16,17,2048])
def test_positions_predeclared_bounded_and_unique(length):
    positions=prepare.prediction_positions([5]*length)
    assert len(positions)==min(length,16)
    assert len(set(positions))==len(positions)
    assert min(positions)>=0 and max(positions)<length


def test_completion_stops_at_first_eos_and_never_scores_padding():
    assert prepare.prediction_positions([9,8,2,0,0,9])==[0,1,2]
    assert prepare.prediction_positions([])==[]


def test_historical_source_records_exact_pre_update_identity(tmp_path):
    source=tmp_path/prepare.OLD_RUN/"raw_rollouts.jsonl"
    source.parent.mkdir(parents=True)
    row=dict(row_id="x",rollout_source="student_on_policy",update=40,prompt_ids=[1,4],completion_ids=[5,2])
    source.write_text(json.dumps(row)+"\n"+json.dumps(dict(row,update=161))+"\n")
    tasks=[dict(row_id="x",family="contains_count",split="calibration")]
    records,missing=prepare.historic_prefixes(tmp_path,tasks)
    assert [r["source"] for r in records]==["early_student","late_student"]
    assert [r["provenance"]["behavior_after_update"] for r in records]==[39,160]
    assert all(r["checkpoint"] is None for r in records)
    assert all(not r["provenance"]["checkpoint_weights_preserved"] for r in records)
    assert records[0]["positions"]==[0,1]
    assert missing[0]["missing_ids"]==["x"]


def test_teacher_rollouts_cannot_be_mislabeled_as_student(tmp_path):
    source=tmp_path/prepare.OLD_RUN/"raw_rollouts.jsonl"
    source.parent.mkdir(parents=True)
    row=dict(row_id="x",rollout_source="R1_next_teacher",update=1,prompt_ids=[1,4],completion_ids=[5,2])
    source.write_text(json.dumps(row)+"\n")
    found,missing=prepare.historic_prefixes(tmp_path,[dict(row_id="x",family="f",split="heldout")])
    assert not found


def test_missing_checkpoint_is_explicit_not_a_global_abort(tmp_path,monkeypatch):
    def missing(*args):
        raise FileNotFoundError("missing weights")
    monkeypatch.setattr(prepare,"checkpoint_metadata",missing)
    result=prepare._checkpoint(tmp_path,"absent",index=2,step="rl40")
    assert result["status"]=="missing_or_invalid"
    assert "missing weights" in result["error"]
    assert not result["actual_model_loaded"]


def test_no_frozen_test_read_in_module():
    # Preparation deliberately does not call mpc_data.prepare, which streams IDs
    # from the official test. Only named fit/selection sources are permitted here.
    from pathlib import Path
    source=Path(prepare.__file__).read_text()
    assert "_stream_ids_only(" not in source
    assert '"data/cafd/test.jsonl"' not in source
    assert 'load_mpc_rows(' not in source
