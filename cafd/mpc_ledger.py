"""Physical-cost journal and recoverable logical-log rollback for NEW MPC runs.

The physical ledger is append-only and must NEVER be listed among rollback logs.
Resources use ``ledger.add``; ordinary dict assignment is for derived gauges only.
Resume starts from replayed physical totals, not from checkpoint cost snapshots.
"""
from __future__ import annotations

import json
import math
import numbers
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import time
import uuid


def _now():
    return time.time()


def _dump(event):
    return (json.dumps(event, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def _number(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError('resource delta must be a real number, not bool')
    if not math.isfinite(value) or value < 0:
        raise ValueError('physical resource deltas must be finite and nonnegative')
    return int(value) if isinstance(value, numbers.Integral) else float(value)


class CostLedger(dict):
    """Append-before-aggregate physical counters, plus nonjournaled derived gauges.

    A killed write can leave a malformed last line. It remains byte-for-byte in
    place, followed on the next append by an explicit interrupted-tail marker.
    Unknown partial resources are not guessed; completeness becomes lower-bound.
    Malformed *interior* lines without their matching recovery marker are fatal.
    This class has one writer (the experiment runner); it is not a shared logger.
    """

    def __init__(self, path):
        super().__init__()
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.attempts = []
        self.current_job_id = None
        self.resource_names = set()
        self._tail = None
        self._needs_newline = False
        self['counter_completeness'] = 'complete_for_logged_operations'
        if self.path.exists():
            if not self.path.is_file():
                raise ValueError('ledger path must be a regular file')
            self._replay()

    @property
    def job_ids(self):
        return list(dict.fromkeys(a['job_id'] for a in self.attempts if a.get('job_id')))

    def _apply(self, event):
        kind = event.get('event')
        if kind == 'resource':
            name = event.get('name')
            if not isinstance(name, str) or not name or name == 'counter_completeness':
                raise ValueError('invalid resource name in ledger')
            delta = _number(event['delta'])
            self.resource_names.add(name)
            dict.__setitem__(self, name, self.get(name, 0) + delta)
        elif kind == 'attempt':
            self.attempts.append(event)
            self.current_job_id = event.get('job_id')
            if event.get('resume'):
                self['counter_completeness'] = 'lower_bound_after_interruption'
        elif kind == 'interrupted_tail':
            self['counter_completeness'] = 'lower_bound_after_interruption'
        else:
            raise ValueError(f'unknown physical-ledger event: {kind!r}')

    def _replay(self):
        with self.path.open('rb') as handle:
            lines = []
            while True:
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                lines.append((offset, raw))
        for index, (offset, raw) in enumerate(lines):
            if index == len(lines) - 1:
                self._needs_newline = not raw.endswith(b'\n')
            try:
                event = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                tail = raw[:-1] if raw.endswith(b'\n') else raw
                if index + 1 == len(lines):
                    self._tail = dict(offset=offset, byte_length=len(tail), tail_hex=tail.hex())
                    self['counter_completeness'] = 'lower_bound_after_interruption'
                    continue
                try:
                    marker = json.loads(lines[index + 1][1])
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(f'corrupt ledger interior at byte {offset}') from error
                expected = dict(offset=offset, byte_length=len(tail), tail_hex=tail.hex())
                if (not isinstance(marker, dict) or marker.get('event') != 'interrupted_tail'
                        or any(marker.get(k) != v for k, v in expected.items())):
                    raise ValueError(f'corrupt ledger interior at byte {offset}')
                self['counter_completeness'] = 'lower_bound_after_interruption'
                continue
            if not isinstance(event, dict):
                raise ValueError(f'ledger event must be an object at byte {offset}')
            self._apply(event)

    def _append(self, event):
        prefix = b'\n' if self._needs_newline else b''
        if self._tail is not None:
            prefix += _dump(dict(event='interrupted_tail', time=_now(), **self._tail))
        payload = prefix + _dump(event)
        with self.path.open('ab') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self._tail = None
        self._needs_newline = False
        self._apply(event)

    def add(self, name, value):
        if not isinstance(name, str) or not name or name == 'counter_completeness':
            raise ValueError('resource name must be a nonempty nonreserved string')
        delta = _number(value)
        self._append(dict(event='resource', resource=name, name=name, delta=delta,
                          time=_now(), job_id=self.current_job_id))
        return self[name]

    def begin_attempt(self, job_id, resume=False):
        if not isinstance(resume, bool):
            raise ValueError('resume must be bool')
        job_id = None if job_id is None else str(job_id)
        event = dict(event='attempt', attempt_id=uuid.uuid4().hex, job_id=job_id,
                     resume=resume, time=_now(), pid=os.getpid())
        self._append(event)
        return dict(event)


def snapshot_journals(paths):
    """Byte offsets for logical run logs. Missing logs are explicitly offset zero."""
    result = {}
    for value in paths:
        path = Path(value).resolve()
        if path.exists():
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f'not a regular journal: {path}')
            result[str(path)] = info.st_size
        else:
            result[str(path)] = 0
    return result


def _contained(path, roots):
    return any(path != root and path.is_relative_to(root) for root in roots)


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def rollback_journals(offsets, allowed_roots, archive_dir):
    """Archive uncommitted logical-log tails BEFORE truncating any source log.

    Every target must resolve strictly below one of the caller's NEW run output
    or artifact roots. All bounds/paths are validated before creating archives;
    hard-linked logs are rejected. Archived tails and origin metadata remain in
    ``archive_dir``. This never deletes files, checkpoints, or physical ledgers.
    Return ``{'files': [...], 'archived_bytes': int}`` for the resume audit.
    The caller must exclude the physical CostLedger and stop concurrent writers.
    """
    roots = [Path(value).resolve() for value in allowed_roots]
    if not roots or any(root == Path(root.anchor) or not root.is_dir() for root in roots):
        raise ValueError('allowed_roots must be explicit existing run directories')
    archive = Path(archive_dir).resolve()
    if not _contained(archive, roots):
        raise ValueError('archive directory escapes the allowed new-run roots')
    if archive.exists() and not archive.is_dir():
        raise ValueError('archive path is not a directory')
    plans = []
    seen = set()
    for value, offset in offsets.items():
        raw_path = Path(value)
        if not raw_path.is_absolute():
            raise ValueError('journal offsets must use absolute paths')
        path = raw_path.resolve()
        if (not _contained(path, roots) or path == archive or path.is_relative_to(archive)):
            raise ValueError(f'journal path escapes run roots or targets archives: {path}')
        if path in seen:
            raise ValueError('multiple paths resolve to the same journal')
        seen.add(path)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError('journal offsets must be nonnegative integer byte counts')
        if not path.exists():
            if offset:
                raise ValueError(f'missing committed journal: {path}')
            plans.append(dict(path=path, offset=0, info=None))
            continue
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f'journal must be a regular, non-hard-linked file: {path}')
        if offset > info.st_size:
            raise ValueError(f'commit offset exceeds current journal size: {path}')
        plans.append(dict(path=path, offset=offset, info=info))
    active = [plan for plan in plans if plan['info'] is not None and plan['info'].st_size > plan['offset']]
    reports = []
    if not active:
        return dict(files=reports, archived_bytes=0)
    archive.mkdir(parents=True, exist_ok=True)
    # Phase 1: preserve every tail and metadata before changing any logical log.
    for plan in active:
        path, offset, info = plan['path'], plan['offset'], plan['info']
        destination = archive / f'{path.name}.{uuid.uuid4().hex}.tail'
        with path.open('rb') as source, destination.open('xb') as target:
            if _identity(os.fstat(source.fileno())) != _identity(info):
                raise RuntimeError(f'journal changed during rollback validation: {path}')
            source.seek(offset)
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
            if _identity(os.fstat(source.fileno())) != _identity(info):
                raise RuntimeError(f'journal changed while archiving: {path}')
        size = info.st_size - offset
        if destination.stat().st_size != size:
            raise RuntimeError('archived byte count differs from journal tail')
        record = dict(source=str(path), committed_offset=offset, original_size=info.st_size,
                      archived_tail=str(destination), archived_bytes=size, time=_now())
        with destination.with_suffix('.metadata.json').open('x', encoding='utf-8') as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        reports.append(record)
    # Phase 2: reopen/revalidate all targets before the first truncate. Opening
    # O_NOFOLLOW and checking inode identity avoids following a swapped symlink.
    opened = []
    try:
        for plan in active:
            fd = os.open(plan['path'], os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0))
            opened.append((fd, plan))
            if _identity(os.fstat(fd)) != _identity(plan['info']):
                raise RuntimeError(f'journal changed before commit rollback: {plan["path"]}')
        for fd, plan in opened:
            os.ftruncate(fd, plan['offset'])
            os.fsync(fd)
    finally:
        for fd, _ in opened:
            os.close(fd)
    return dict(files=reports, archived_bytes=sum(row['archived_bytes'] for row in reports))


def allocation_seconds(job_ids):
    """Read each Slurm allocation's actual elapsed seconds; unknown => None.

    Arrays must be named by an explicit task ID, not a job range. Duplicate IDs
    count once. No shell, submission, cancellation, or fabricated zero fallback.
    """
    jobs = list(dict.fromkeys(str(value) for value in job_ids))
    if not jobs or any(not re.fullmatch(r'[0-9]+(?:_[0-9]+)?', job) for job in jobs):
        return None
    total = 0
    for job in jobs:
        try:
            result = subprocess.run(
                ['sacct', '-X', '-n', '-P', '-j', job, '-o', 'JobIDRaw,ElapsedRaw'],
                capture_output=True, text=True, check=False, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        values = []
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split('|')]
            if len(fields) >= 2 and fields[0] == job:
                if not fields[1].isdigit():
                    return None
                values.append(int(fields[1]))
        if len(values) != 1:
            return None
        total += values[0]
    return total
