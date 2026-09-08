"""No Slurm or GPU required: isolated synthetic ledger/log files only."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cafd.mpc_ledger import CostLedger, allocation_seconds, rollback_journals, snapshot_journals


def test_ledger_replays_all_attempts_and_resume_is_lower_bound(tmp_path):
    path = tmp_path / 'physical.jsonl'
    ledger = CostLedger(path)
    ledger.begin_attempt('123')
    ledger.add('tokens', 10)
    ledger.add('tokens', 4)
    ledger.add('cpu_seconds', 0.5)
    assert ledger['tokens'] == 14
    original = path.read_bytes()
    resumed = CostLedger(path)
    assert resumed['tokens'] == 14 and resumed['cpu_seconds'] == 0.5
    resumed.begin_attempt('124', resume=True)
    resumed.add('tokens', 3)
    assert path.read_bytes().startswith(original)
    third = CostLedger(path)
    assert third['tokens'] == 17 and third.job_ids == ['123', '124']
    assert third['counter_completeness'] == 'lower_bound_after_interruption'
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[-1]['event'] == 'resource' and events[-1]['resource'] == 'tokens'
    assert events[-1]['delta'] == 3 and events[-1]['job_id'] == '124'
    assert isinstance(events[-1]['time'], float)


def test_derived_gauge_assignment_is_not_a_physical_delta(tmp_path):
    path = tmp_path / 'physical.jsonl'
    ledger = CostLedger(path)
    ledger.add('tokens', 3)
    ledger['peak_bytes'] = 1234
    assert json.loads(json.dumps(ledger))['peak_bytes'] == 1234
    replay = CostLedger(path)
    assert replay['tokens'] == 3 and 'peak_bytes' not in replay


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True, '3'])
def test_invalid_resource_deltas_never_write(tmp_path, value):
    ledger = CostLedger(tmp_path/'physical.jsonl')
    with pytest.raises(ValueError):
        ledger.add('tokens', value)
    assert not ledger.path.exists()


def test_torn_tail_keeps_original_bytes_and_replays_after_recovery_marker(tmp_path):
    path = tmp_path/'physical.jsonl'
    ledger = CostLedger(path)
    ledger.begin_attempt('100')
    ledger.add('tokens', 7)
    with path.open('ab') as handle:
        handle.write(b'{"event":"resource","name":"tokens","delta":')
    before = path.read_bytes()
    resumed = CostLedger(path)
    assert resumed['tokens'] == 7
    resumed.begin_attempt('101', resume=True)
    resumed.add('tokens', 2)
    assert path.read_bytes().startswith(before)
    replay = CostLedger(path)
    assert replay['tokens'] == 9 and replay.job_ids == ['100', '101']
    assert replay['counter_completeness'] == 'lower_bound_after_interruption'
    assert b'"event": "interrupted_tail"' in path.read_bytes()


def test_valid_final_event_without_newline_is_not_lost(tmp_path):
    path = tmp_path/'physical.jsonl'
    path.write_text(json.dumps(dict(event='resource', name='tokens', delta=7)))
    ledger = CostLedger(path)
    assert ledger['tokens'] == 7
    ledger.add('tokens', 2)
    assert CostLedger(path)['tokens'] == 9


def test_corrupt_interior_event_rejected_instead_of_silently_ignored(tmp_path):
    path = tmp_path/'physical.jsonl'
    path.write_bytes(b'not-json\n'+json.dumps(dict(event='resource',name='tokens',delta=2)).encode()+b'\n')
    with pytest.raises(ValueError, match='interior'):
        CostLedger(path)


def make_logs(tmp_path):
    out = tmp_path/'new_run'
    artifacts = tmp_path/'new_artifacts'
    out.mkdir()
    artifacts.mkdir()
    first = out/'rollouts.jsonl'
    second = artifacts/'curve.jsonl'
    first.write_bytes(b'committed1\n')
    second.write_bytes(b'committed2\n')
    return out, artifacts, first, second


def test_snapshot_and_rollback_preserve_exact_binary_tails_and_metadata(tmp_path):
    out, artifacts, first, second = make_logs(tmp_path)
    missing = out/'not_created.jsonl'
    offsets = snapshot_journals([first, second, missing])
    assert offsets[str(missing.resolve())] == 0
    tail1, tail2 = b'partial\xff\x00\n', '未提交\n'.encode()
    with first.open('ab') as handle:
        handle.write(tail1)
    with second.open('ab') as handle:
        handle.write(tail2)
    result = rollback_journals(offsets, [out, artifacts], out/'discarded_attempts')
    assert first.read_bytes() == b'committed1\n'
    assert second.read_bytes() == b'committed2\n'
    assert not missing.exists()
    tails = {r['source']: Path(r['archived_tail']).read_bytes() for r in result['files']}
    assert tails == {str(first): tail1, str(second): tail2}
    assert result['archived_bytes'] == len(tail1)+len(tail2)
    for item in result['files']:
        metadata = Path(item['archived_tail']).with_suffix('.metadata.json')
        assert json.loads(metadata.read_text()) == item
    assert rollback_journals(offsets,[out,artifacts],out/'discarded_attempts')['archived_bytes'] == 0


@pytest.mark.parametrize('bad_offset', [-1, True, 1000000, 1.5])
def test_all_bounds_validated_before_any_archive_or_truncate(tmp_path, bad_offset):
    out, artifacts, first, second = make_logs(tmp_path)
    before = first.read_bytes()
    with pytest.raises(ValueError):
        rollback_journals({str(first): 0, str(second): bad_offset},[out,artifacts],out/'discarded_attempts')
    assert first.read_bytes() == before
    assert not (out/'discarded_attempts').exists()


def test_path_escape_symlink_and_archive_escape_rejected(tmp_path):
    out, artifacts, first, second = make_logs(tmp_path)
    old = tmp_path/'old_run.log'
    old.write_bytes(b'old evidence')
    link = out/'escape.log'
    link.symlink_to(old)
    for bad in (str(old), str(link), 'relative.log'):
        with pytest.raises(ValueError):
            rollback_journals({bad:0},[out,artifacts],out/'discarded_attempts')
    with pytest.raises(ValueError):
        rollback_journals({str(first):0},[out,artifacts],tmp_path/'external_archive')
    assert old.read_bytes() == b'old evidence' and first.read_bytes() == b'committed1\n'


def test_hardlinked_log_refused_and_missing_committed_log_refused(tmp_path):
    out, artifacts, first, second = make_logs(tmp_path)
    (out/'other_name').hardlink_to(first)
    with pytest.raises(ValueError, match='hard-linked'):
        rollback_journals({str(first):0},[out,artifacts],out/'discarded_attempts')
    with pytest.raises(ValueError, match='missing committed'):
        rollback_journals({str(out/'missing'):1},[out,artifacts],out/'discarded_attempts')


def test_physical_ledger_is_preserved_when_logical_logs_rollback(tmp_path):
    out, artifacts, first, second = make_logs(tmp_path)
    ledger = CostLedger(out/'physical_cost.jsonl')
    ledger.add('tokens', 11)
    before = ledger.path.read_bytes()
    rollback_journals({str(first):0},[out,artifacts],out/'discarded_attempts')
    assert ledger.path.read_bytes() == before
    assert CostLedger(ledger.path)['tokens'] == 11


def test_allocation_seconds_queries_each_distinct_job_read_only(monkeypatch):
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        job = command[command.index('-j')+1]
        elapsed = {'123': 30, '124': 70}[job]
        return SimpleNamespace(returncode=0, stdout=f'{job}|{elapsed}\n{job}.batch|999\n')
    monkeypatch.setattr('cafd.mpc_ledger.subprocess.run',fake_run)
    assert allocation_seconds(['123','124','123']) == 100
    assert len(calls) == 2
    assert all(c[:4] == ['sacct','-X','-n','-P'] for c in calls)


@pytest.mark.parametrize('result', [
    SimpleNamespace(returncode=1,stdout='123|20\n'),
    SimpleNamespace(returncode=0,stdout=''),
    SimpleNamespace(returncode=0,stdout='123|Unknown\n'),
    SimpleNamespace(returncode=0,stdout='123|20\n123|21\n'),
])
def test_allocation_failure_or_ambiguity_is_none_not_zero(monkeypatch,result):
    monkeypatch.setattr('cafd.mpc_ledger.subprocess.run', lambda *args, **kwargs: result)
    assert allocation_seconds(['123']) is None
    assert allocation_seconds([]) is None
    assert allocation_seconds(['--help']) is None
