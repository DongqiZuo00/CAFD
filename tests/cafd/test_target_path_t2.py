import copy
import json
import numpy as np
import pytest
import torch
from cafd.target_path_t2 import (
    validate_records, expand_records, position_weights, exact_kl,
    endpoint_derivative, fit_coefficient, chunked_fit, summarize_values,
    coverage, model_paths, score_checkpoint, checkpoint_stamp, validated_cache, PhysicalAttemptLedger,
)


def manifest():
    return [dict(rowid=str(i), source=s, family="family", checkpoint="/real/S0",
                 prompt_ids=[1,2], completion_ids=[3,4,5], positions=[0,2],
                 split="calibration" if i < 2 else "heldout")
            for s in ("S0", "early_student") for i in range(4)]


def tensors():
    generator = torch.Generator().manual_seed(2027)
    return [torch.randn((8,31), generator=generator) for _ in range(3)]


def test_validate_and_expand_completion_offsets():
    rows = validate_records(manifest())
    assert len(expand_records(rows)) == 16
    assert rows[0]["positions"] == [0,2]


def test_question_split_isolation():
    rows = manifest()
    rows[4]["split"] = "heldout"
    with pytest.raises(ValueError, match="leaks"):
        validate_records(rows)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(positions=list(range(17))),
    lambda r: r.update(positions=[0,0]),
    lambda r: r.update(positions=[3]),
    lambda r: r.update(source="verified_gold"),
    lambda r: r.update(prompt_ids=[]),
    lambda r: r.update(split="test"),
])
def test_invalid_manifest_is_rejected(mutation):
    rows = manifest()
    mutation(rows[0])
    with pytest.raises(ValueError):
        validate_records(rows)


def test_missing_saved_behavior_weights_allowed_only_with_honest_provenance():
    rows = manifest()
    rows[0]["checkpoint"] = None
    with pytest.raises(ValueError, match="provenance"):
        validate_records(rows)
    rows[0]["provenance"] = dict(source_run="old", behavior_after_update=39, source_file="raw.jsonl", line_number=1)
    assert validate_records(rows)[0]["checkpoint"] is None
    assert coverage(rows)[0]["exact_origin_checkpoint_missing"] == 1


def test_equal_source_question_weight_and_zero_heldout():
    records = manifest()
    records[0]["positions"] = [0]
    rows = expand_records(records)
    weights = position_weights(rows, "calibration")
    assert weights.sum() == pytest.approx(1.)
    assert sum(w for w,r in zip(weights,rows) if r["source"] == "S0") == pytest.approx(.5)
    assert sum(w for w,r in zip(weights,rows) if r["source"] == "S0" and r["rowid"] == "0") == pytest.approx(.25)
    assert all(w == 0 for w,r in zip(weights,rows) if r["split"] == "heldout")


def test_exact_kl_matches_reference_full_normalization():
    base, delta, q = tensors()
    actual = exact_kl(q, base+.3*delta)
    reference = (q.double().softmax(-1)*(q.double().log_softmax(-1)-(base+.3*delta).double().log_softmax(-1))).sum(-1)
    torch.testing.assert_close(actual, reference, atol=3e-7, rtol=3e-7)
    assert torch.equal(exact_kl(q,q), torch.zeros(8,dtype=torch.float64))


def test_logit_offset_invariance():
    base, delta, q = tensors()
    shifted = exact_kl(q+3.,base-2.)
    torch.testing.assert_close(shifted,exact_kl(q,base),atol=4e-7,rtol=4e-7)


def test_analytic_derivative_matches_double_autograd():
    base, delta, q = tensors()
    weights = torch.ones(8,dtype=torch.float64)/8
    actual = endpoint_derivative(base,delta,q,.42,weights)
    a = torch.tensor(.42,dtype=torch.float64,requires_grad=True)
    loss = (q.double().softmax(-1)*(q.double().log_softmax(-1)-(base.double()+a*delta.double()).log_softmax(-1))).sum(-1).mean()
    loss.backward()
    assert actual == pytest.approx(float(a.grad),abs=1e-7)


def test_scalar_fit_recovers_endpoint_interpolation():
    base, delta, _ = tensors()
    q = base+.37*delta
    result = fit_coefficient(base,delta,q,torch.ones(8,dtype=torch.float64)/8)
    assert isinstance(result["a"], float)
    assert result["a"] == pytest.approx(.37,abs=5e-7)


@pytest.mark.parametrize("true_a,expected", [(-1.,0.),(2.,1.),(0.,0.),(1.,1.)])
def test_boundary_solution(true_a,expected):
    base,delta,_ = tensors()
    result = fit_coefficient(base,delta,base+true_a*delta,torch.ones(8,dtype=torch.float64)/8)
    assert result["a"] == pytest.approx(expected,abs=1e-6)


def test_one_scalar_not_tokenwise_fit():
    base,delta,_ = tensors()
    q = base+torch.linspace(.1,.9,8)[:,None]*delta
    weights = torch.ones(8,dtype=torch.float64)/8
    result = fit_coefficient(base,delta,q,weights)
    assert .1 < result["a"] < .9
    assert exact_kl(q,base+result["a"]*delta).mean() > 1e-4


def test_heldout_target_changes_do_not_change_fitted_a():
    base,delta,q = tensors()
    weights = torch.tensor([.25,.25,.25,.25,0,0,0,0],dtype=torch.float64)
    before = fit_coefficient(base,delta,q,weights)
    q[4:] = 100*delta[4:]
    after = fit_coefficient(base,delta,q,weights)
    assert before == after


@pytest.mark.parametrize("chunk", [1,3,8])
def test_chunked_fit_matches_full_vocabulary_fit(chunk):
    base,delta,q = tensors()
    weights = torch.ones(8,dtype=torch.float64)/8
    first = fit_coefficient(base,delta,q,weights)
    second = chunked_fit(base,delta,q,weights,chunk=chunk)
    assert first["a"] == pytest.approx(second["a"],abs=2e-6)


def test_signed_roundoff_not_silently_clamped_in_reporting():
    rows = expand_records(manifest())
    values = np.full(len(rows),-1e-8)
    result = summarize_values(values,rows)[("all","heldout")]
    assert result["kl"] == pytest.approx(-1e-8)
    assert result["negative_position_kl_count"] == 8


def test_unavailable_source_cells_disclosed():
    records = manifest()[:-1]
    result = {r["source"]:r for r in coverage(records)}
    assert result["early_student"]["missing_question_ids"] == ["3"]
    assert result["early_student"]["heldout_questions"] == 1


def test_model_map_preserves_integer_stage_labels():
    paths,M = model_paths(dict(s0="/S0",M=5,teacher_route=[dict(stage=0,path="/T0"),dict(stage=5,path="/T5")]))
    assert M == 5
    assert set(paths) == {"S0","T0","T5"}


def test_score_resume_reuses_cache_without_model_load_or_forward(tmp_path,monkeypatch):
    from cafd import mpc_runtime
    def prohibited(*args,**kwargs):
        raise AssertionError("resume cache must not load a Transformer")
    monkeypatch.setattr(mpc_runtime,"load_hidden",prohibited)
    checkpoint=tmp_path/"S0"
    checkpoint.mkdir()
    (checkpoint/"config.json").write_text("{}")
    cache=tmp_path/"S0.npy"
    np.save(cache,np.zeros((2,7),dtype=np.float32))
    cache.with_suffix(".json").write_text(json.dumps(dict(status="complete",manifest_digest="frozen-prefixes",
        checkpoint=str(checkpoint),checkpoint_stamp=checkpoint_stamp(checkpoint),shape=[2,7],elapsed_seconds=12.,transformer_scored_tokens=20)))
    result=score_checkpoint(checkpoint,[],cache,digest="frozen-prefixes",device="cpu",label="S0")
    assert result["reused"] is True
    assert result["elapsed_seconds"] == 12.
    assert result["current_elapsed_seconds"] < 5.


def test_score_resume_refuses_changed_prefix_digest(tmp_path,monkeypatch):
    from cafd import mpc_runtime
    monkeypatch.setattr(mpc_runtime,"load_hidden",lambda *a,**kw: pytest.fail("unexpected Transformer load"))
    checkpoint=tmp_path/"S0"
    checkpoint.mkdir()
    cache=tmp_path/"S0.npy"
    np.save(cache,np.zeros((2,7),dtype=np.float32))
    cache.with_suffix(".json").write_text(json.dumps(dict(status="complete",manifest_digest="old",
        checkpoint=str(checkpoint),shape=[2,7])))
    with pytest.raises(RuntimeError,match="cache provenance"):
        score_checkpoint(checkpoint,[],cache,digest="changed",device="cpu",label="S0")


def cache_fixture(tmp_path,stamp=True):
    checkpoint=tmp_path/"checkpoint"
    checkpoint.mkdir()
    (checkpoint/"config.json").write_text("{}")
    cache=tmp_path/"cache.npy"
    np.save(cache,np.zeros((2,7),dtype=np.float32))
    meta=dict(status="complete",manifest_digest="fixed",checkpoint=str(checkpoint),shape=[2,7])
    if stamp:
        meta["checkpoint_stamp"]=checkpoint_stamp(checkpoint)
    cache.with_suffix(".json").write_text(json.dumps(meta))
    return checkpoint,cache


def test_cache_checkpoint_mutation_refused_without_weight_hashing(tmp_path):
    checkpoint,cache=cache_fixture(tmp_path)
    (checkpoint/"config.json").write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match="size/mtime"):
        validated_cache(checkpoint,cache,"fixed")


def test_legacy_cache_default_refusal(tmp_path):
    checkpoint,cache=cache_fixture(tmp_path,stamp=False)
    with pytest.raises(RuntimeError,match="legacy cache"):
        validated_cache(checkpoint,cache,"fixed")


def test_legacy_cache_migration_requires_exact_original_metadata(tmp_path):
    checkpoint,cache=cache_fixture(tmp_path,stamp=False)
    info=(checkpoint/"config.json").stat()
    proof=tmp_path/"manifest.json"
    proof.write_text(json.dumps(dict(student_base=dict(checkpoint=str(checkpoint),metadata=dict(files=[
        dict(path=str(checkpoint/"config.json"),size_bytes=info.st_size,mtime_ns=info.st_mtime_ns)])))))
    result=validated_cache(checkpoint,cache,"fixed",proof)
    assert result["legacy_stamp_migration"]["transformer_rerun"] is False
    assert result["checkpoint_stamp"]==checkpoint_stamp(checkpoint)
    assert validated_cache(checkpoint,cache,"fixed")["checkpoint_stamp"]==result["checkpoint_stamp"]


def test_wrong_legacy_proof_cannot_bypass_provenance(tmp_path):
    checkpoint,cache=cache_fixture(tmp_path,stamp=False)
    proof=tmp_path/"manifest.json"
    proof.write_text(json.dumps(dict(student_base=dict(checkpoint=str(checkpoint),metadata=dict(files=[])))))
    with pytest.raises(RuntimeError,match="legacy cache"):
        validated_cache(checkpoint,cache,"fixed",proof)


class Clock:
    def __init__(self):
        self.now=0.
    def __call__(self):
        return self.now


def test_physical_resume_never_rebills_cache_and_archives_cost_snapshot(tmp_path):
    clock=Clock()
    first=PhysicalAttemptLedger(tmp_path,clock)
    meta=dict(reused=False,transformer_scored_tokens=100,lm_head_scored_positions=16,
              model_loading_seconds=2.,scoring_seconds=5.)
    first.checkpoint(meta)
    clock.now=10.
    first.finish()
    snapshot=dict(current_total_seconds=10.,transformer_scored_tokens=100,checkpoints=[meta],**first.fields())
    (tmp_path/"t2_costs.json").write_text(json.dumps(snapshot))
    second=PhysicalAttemptLedger(tmp_path,clock)
    second.checkpoint(dict(meta,reused=True))
    clock.now=15.
    second.finish()
    fields=second.fields()
    assert fields["physical_gpu_hours"]==pytest.approx(15/3600)
    assert fields["physical_transformer_scored_tokens"]==100
    assert fields["physical_lm_head_scored_positions"]==16
    assert fields["physical_cost_completeness"]=="complete_for_logged_operations"
    assert len(list((tmp_path/"t2_cost_attempts").glob("*.json")))==1


def test_interrupted_attempt_marks_physical_lower_bound(tmp_path):
    clock=Clock()
    first=PhysicalAttemptLedger(tmp_path,clock)
    clock.now=5.
    first.observe()
    clock.now=10.
    second=PhysicalAttemptLedger(tmp_path,clock)
    clock.now=17.
    second.finish()
    fields=second.fields()
    assert fields["physical_gpu_hours"]==pytest.approx(12/3600)
    assert fields["physical_cost_completeness"]=="lower_bound_after_interruption"
    assert fields["interrupted_or_incomplete_attempts"]==1


def test_legacy_completed_cost_imported_exactly_once(tmp_path):
    (tmp_path/"t2_costs.json").write_text(json.dumps(dict(current_total_seconds=10.,
        transformer_scored_tokens=100,checkpoints=[dict(reused=False,lm_head_scored_positions=16)])))
    clock=Clock()
    first=PhysicalAttemptLedger(tmp_path,clock)
    clock.now=2.
    first.finish()
    second=PhysicalAttemptLedger(tmp_path,clock)
    clock.now=5.
    second.finish()
    assert second.fields()["physical_gpu_hours"]==pytest.approx(15/3600)
    assert second.fields()["physical_transformer_scored_tokens"]==100


def test_orphan_legacy_cache_is_known_cost_lower_bound_only(tmp_path):
    cache=tmp_path/"t2_logits"
    cache.mkdir()
    (cache/"S0.json").write_text(json.dumps(dict(status="complete",elapsed_seconds=10.,
        transformer_scored_tokens=100,lm_head_scored_positions=16)))
    clock=Clock()
    ledger=PhysicalAttemptLedger(tmp_path,clock)
    clock.now=2.
    ledger.finish()
    assert ledger.fields()["physical_gpu_hours"]==pytest.approx(12/3600)
    assert ledger.fields()["physical_cost_completeness"]=="lower_bound_after_interruption"
