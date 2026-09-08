import json
import pytest
from cafd.target_path_resume import archive_stale_claims, root_job_state, numeric_job


def make_claim(root, policy, job="100"):
    claim = root/"claims"/policy
    claim.mkdir(parents=True)
    (claim/"owner.json").write_text(json.dumps(dict(job=job,pid=42,worker=0)))
    (claim/"done.json").write_text('{"complete":false}')
    (claim/"error.json").write_text('{"error":"preserved"}')
    return claim


def test_only_exact_root_state_counts():
    assert root_job_state("100", "100|COMPLETED|\n100.batch|FAILED|\n") == "COMPLETED"
    assert root_job_state("100", "100|CANCELLED by 1234|\n") == "CANCELLED"


@pytest.mark.parametrize("output", ["", "100.batch|COMPLETED|", "100|COMPLETED|\n100|FAILED|"])
def test_missing_ambiguous_accounting_failclosed(output):
    with pytest.raises(RuntimeError):
        root_job_state("100",output)


@pytest.mark.parametrize("value", [None,"100.batch","100_1","0","-1","100;echo","unknown"])
def test_only_numeric_allocation_ids(value):
    with pytest.raises(RuntimeError):
        numeric_job(value)


def test_archive_preserves_claim_details_and_all_outputs(tmp_path):
    old = make_claim(tmp_path,"q5_random")
    (tmp_path/"rollouts.jsonl").write_text("unchanged\n")
    (tmp_path/"worker0_state.json").write_text('{"job_id":"100","status":"partial"}')
    (tmp_path/"worker_exit.json").write_text('{"job_id":"100","worker0_exit":1}')
    result = archive_stale_claims(tmp_path,"200",lambda job:"COMPLETED")
    assert not old.exists()
    preserved = result["archived"][0]["preserved_claim_directory"]
    from pathlib import Path
    assert (Path(preserved)/"error.json").read_text() == '{"error":"preserved"}'
    assert (Path(preserved)/"done.json").read_text() == '{"complete":false}'
    assert (tmp_path/"rollouts.jsonl").read_text() == "unchanged\n"
    assert len(result["worker_provenance_copies"])==2
    for item in result["worker_provenance_copies"]:
        assert Path(item["source"]).exists()
        assert Path(item["source"]).read_bytes()==Path(item["preserved_copy"]).read_bytes()


@pytest.mark.parametrize("state", ["RUNNING","PENDING","COMPLETING","SUSPENDED","REQUEUED","","UNKNOWN"])
def test_active_or_unknown_job_never_reclaimed(tmp_path,state):
    old=make_claim(tmp_path,"q5_random")
    with pytest.raises(RuntimeError,match="not proven terminal"):
        archive_stale_claims(tmp_path,"200",lambda job:state)
    assert old.exists()
    assert not (tmp_path/"claim_archive").exists()


def test_validation_completes_before_any_move(tmp_path):
    first=make_claim(tmp_path,"a","100")
    last=make_claim(tmp_path,"z","101")
    with pytest.raises(RuntimeError):
        archive_stale_claims(tmp_path,"200",lambda job:"COMPLETED" if job=="100" else "RUNNING")
    assert first.exists() and last.exists()
    assert not (tmp_path/"claim_archive").exists()


def test_same_job_claim_never_stolen(tmp_path):
    claim=make_claim(tmp_path,"p5_random","200")
    with pytest.raises(RuntimeError,match="same-job"):
        archive_stale_claims(tmp_path,"200",lambda job:"COMPLETED")
    assert claim.exists()


def test_missing_owner_is_not_inferred(tmp_path):
    claim=make_claim(tmp_path,"p5_random")
    (claim/"owner.json").unlink()
    with pytest.raises(RuntimeError,match="owner"):
        archive_stale_claims(tmp_path,"200",lambda job:"COMPLETED")
    assert claim.exists()


def test_accounting_error_is_not_swallowed(tmp_path):
    claim=make_claim(tmp_path,"p5_random")
    def fail(job):
        raise TimeoutError("sacct unavailable")
    with pytest.raises(TimeoutError):
        archive_stale_claims(tmp_path,"200",fail)
    assert claim.exists()


def test_one_accounting_query_per_prior_job(tmp_path):
    make_claim(tmp_path,"p5_random")
    make_claim(tmp_path,"q5_random")
    calls=[]
    def state(job):
        calls.append(job)
        return "FAILED"
    result=archive_stale_claims(tmp_path,"200",state)
    assert calls==["100"] and len(result["archived"])==2


def test_symlink_claim_refused(tmp_path):
    outside=tmp_path/"outside"
    outside.mkdir()
    (tmp_path/"claims").mkdir()
    (tmp_path/"claims"/"q").symlink_to(outside,target_is_directory=True)
    with pytest.raises(RuntimeError,match="unsafe"):
        archive_stale_claims(tmp_path,"200",lambda job:"COMPLETED")


def test_no_claims_is_noop(tmp_path):
    assert archive_stale_claims(tmp_path,"200",lambda job:"COMPLETED")["status"]=="no_claims"
