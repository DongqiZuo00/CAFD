"""Synthetic-file tests: no real test labels, GPU, model load or Slurm submit."""
from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path

import pytest

from cafd import mpc_finalize as final

RUN_ID = "mistral_cafd_mpc_v1_formal"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def checkpoint(path):
    write_json(path / "config.json", {"model_type": "ministral3", "vocab_size": 131072})
    write_json(path / "tokenizer_config.json", {"eos_token": "</s>"})
    write_json(path / "tokenizer.json", {"model": {"type": "BPE"}})
    with (path / "model.safetensors").open("wb") as handle:
        handle.truncate(1_000_001)


@pytest.fixture
def fixture(tmp_path):
    run = tmp_path / "runs/cafd/experiments" / RUN_ID
    cp = run / "round40"
    checkpoint(cp)
    checkpoint(tmp_path / final.S0_REL)
    cfg = {
        "profile": "formal", "protocol": "CAFD-MPC-2026-09-06", "total_rounds": 200,
        "max_new_tokens": 2048, "max_prompt_tokens": 4096, "seed": 2027,
        "prompts_per_round": 4, "rollouts_per_prompt": 8, "block_rounds": 10,
        "horizon": 2, "milestones": [0, 10, 40, 80, 120, 160, 200],
        "output_stop": "external_closing_fence_or_generated_EOS_no_synthetic_EOS",
        "reward": "hierarchical_0_.05_.10_.10+.90*pass_rate_1",
    }
    best = {"round": 40, "correct": 40, "total": 64, "accuracy": 40 / 64, "checkpoint": str(cp)}
    selected = dict(best, status="frozen", selection_rule="development_full_pass_earliest_tie")
    complete = {"status": "STUDENT_TRAINING_COMPLETE", "formal_result": True,
                "rounds": 200, "actual_updates": 190, "selected": best, "job_id": "41240000",
                "costs": {"training_generated_tokens": 123456, "training_teacher_scored_tokens": 234567,
                          "gpu_hours": 1.25}}
    write_json(run / "config.json", cfg)
    write_json(run / "selected.json", selected)
    write_json(run / "complete.json", complete)
    rows = [{"id": f"task-{i}", "problem_family": final.FAMILIES[i % 3],
             "messages": [{"role": "user", "content": "SYNTHETIC_TEST_QUESTION"}],
             "ground_truth": "SYNTHETIC_TEST_LABEL"} for i in range(132)]
    test = tmp_path / final.TEST_REL
    test.parent.mkdir(parents=True, exist_ok=True)
    test.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    ids = [row["id"] for row in rows]
    write_json(tmp_path / final.IDS_REL, {"ids": ids, "count": 132, "source": str(final.TEST_REL)})
    manifest = {
        "version": "cafd_mpc_v1", "status": "frozen",
        "partitions": {name: {"ids": [f"{name}-0"]} for name in ("optimization", "control", "development")},
        "official_test": {"count": 132, "ids_metadata_path": str(final.IDS_REL),
            "source_file_metadata": final._file_metadata(tmp_path, test), "historically_evaluated": True},
        "limitations": ["historically evaluated test; development seen by Teacher"],
    }
    write_json(run / "data_manifest.json", manifest)
    return tmp_path, run, rows


def mutate(path, **changes):
    value = json.loads(path.read_text())
    value.update(changes)
    write_json(path, value)


def fake_generate(model, tokenizer, row, **kwargs):
    assert kwargs["count"] == 1 and kwargs["sample"] is False
    assert kwargs["max_new_tokens"] == 2048 and kwargs["max_prompt_tokens"] == 4096
    index = int(row["id"].split("-")[-1])
    success = index % 2 == 0
    return [{
        "row_id": row["id"], "family": row["problem_family"],
        "prompt_ids": [1, 7], "completion_ids": [3, 2],
        "old_log_probs": [-.2, -.3], "reward": 1. if success else .1,
        "full_pass": success, "tier": "full_pass" if success else "parse_only",
        "length_cap": False, "source": "current_student", "seed": kwargs["seed"],
    }]


def test_admission_does_not_read_test_payload(fixture, monkeypatch):
    root, run, _ = fixture
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        if path.resolve() == (root / final.TEST_REL).resolve():
            raise AssertionError("metadata admission read test payload")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    context = final.validate_finalization(root, RUN_ID)
    assert context["identity"]["training_rounds"] == 200
    assert context["identity"]["actual_optimizer_updates"] == 190
    assert len(context["identity"]["test_ids"]) == 132


@pytest.mark.parametrize("changes", [
    {"status": "RUNNING"}, {"status": "SMOKE_PASSED"}, {"formal_result": False},
    {"rounds": 199}, {"rounds": 201}, {"rounds": 200.0}, {"actual_updates": 201},
])
def test_incomplete_or_nonformal_run_rejected_before_test_read(fixture, monkeypatch, changes):
    root, run, _ = fixture
    mutate(run / "complete.json", **changes)
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        if path.resolve() == (root / final.TEST_REL).resolve():
            raise AssertionError("premature test read")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(RuntimeError):
        final.validate_finalization(root, RUN_ID)


def test_mismatched_selected_endpoint_rejected(fixture):
    root, run, _ = fixture
    mutate(run / "selected.json", correct=41)
    with pytest.raises(RuntimeError, match="differs"):
        final.validate_finalization(root, RUN_ID)


def test_foreign_checkpoint_rejected_even_with_matching_completion(fixture):
    root, run, _ = fixture
    path = root / "runs/cafd/experiments/another_run/round40"
    checkpoint(path)
    selected = json.loads((run / "selected.json").read_text())
    selected["checkpoint"] = str(path)
    write_json(run / "selected.json", selected)
    complete = json.loads((run / "complete.json").read_text())
    complete["selected"]["checkpoint"] = str(path)
    write_json(run / "complete.json", complete)
    with pytest.raises(RuntimeError, match="does not belong"):
        final.validate_finalization(root, RUN_ID)


def test_unchanged_s0_can_win_development_selection(fixture):
    root, run, _ = fixture
    selected = json.loads((run / "selected.json").read_text())
    selected.update(round=0, checkpoint=str(root / final.S0_REL))
    write_json(run / "selected.json", selected)
    complete = json.loads((run / "complete.json").read_text())
    complete["selected"] = {k: v for k, v in selected.items() if k not in {"status", "selection_rule"}}
    write_json(run / "complete.json", complete)
    assert final.validate_finalization(root, RUN_ID)["identity"]["selection"]["round"] == 0


def test_changed_test_source_metadata_rejected(fixture):
    root, run, _ = fixture
    with (root / final.TEST_REL).open("a") as handle:
        handle.write("\n")
    with pytest.raises(RuntimeError, match="metadata changed"):
        final.validate_finalization(root, RUN_ID)


def test_frozen_id_overlap_rejected(fixture):
    root, run, _ = fixture
    manifest = json.loads((run / "data_manifest.json").read_text())
    manifest["partitions"]["control"]["ids"] = ["task-0"]
    write_json(run / "data_manifest.json", manifest)
    with pytest.raises(RuntimeError, match="overlap control"):
        final.validate_finalization(root, RUN_ID)


def test_sequential_resume_and_final_artifacts_are_unique_and_idempotent(fixture):
    root, run, _ = fixture
    context = final.validate_finalization(root, RUN_ID)
    rows = final.load_test_rows(context)
    costs = {}
    attempts = []
    def interrupt_once(model, tokenizer, row, **kwargs):
        attempts.append(row["id"])
        if row["id"] == "task-5":
            raise OSError("synthetic interruption")
        return fake_generate(model, tokenizer, row, **kwargs)
    with pytest.raises(OSError):
        final.evaluate_remaining(context, rows, None, None, costs, generate_fn=interrupt_once)
    artifact = Path(context["artifact"])
    assert len(final.read_outputs(artifact / "final_test_outputs.jsonl", context)) == 5
    resumed_ids = []
    def resumed(model, tokenizer, row, **kwargs):
        resumed_ids.append(row["id"])
        return fake_generate(model, tokenizer, row, **kwargs)
    records = final.evaluate_remaining(context, rows, None, None, costs, generate_fn=resumed)
    assert len(records) == 132 and resumed_ids[0] == "task-5"
    assert costs["evaluation_verifier_calls"] == 132
    assert costs["evaluation_generated_tokens"] == 264
    result = final.seal_result(context, records, costs)
    assert result["scores"]["correct"] == 66 and result["scores"]["accuracy"] == .5
    assert all(item["correct"] == 22 and item["total"] == 44 for item in result["by_family"])
    assert result["frozen_test_evaluated"] and not result["checkpoint_selection_changed"]
    assert not result["historically_untouched_test"]
    csv_rows = list(csv.DictReader(io.StringIO((artifact / "final_scores.csv").read_text())))
    assert len(csv_rows) == 1 and csv_rows[0]["correct"] == "66"
    files = ["final_scores.json", "final_scores.csv", "final_scores_by_family.csv", "final_test_costs.json"]
    before = {name: ((artifact / name).stat().st_mtime_ns, (artifact / name).read_bytes()) for name in files}
    costs["evaluation_operation_seconds"] += 99.
    assert final.seal_result(context, records, costs) == result
    for name, old in before.items():
        assert ((artifact / name).stat().st_mtime_ns, (artifact / name).read_bytes()) == old


def test_incomplete_last_journal_line_archived_then_resumed(fixture):
    root, run, _ = fixture
    context = final.validate_finalization(root, RUN_ID)
    rows = final.load_test_rows(context)
    final.evaluate_remaining(context, rows, None, None, {}, generate_fn=fake_generate)
    artifact = Path(context["artifact"])
    journal = artifact / "final_test_outputs.jsonl"
    lines = journal.read_bytes().splitlines(keepends=True)
    journal.write_bytes(b"".join(lines[:3]) + b'{"partial":')
    records = final.read_outputs(journal, context)
    assert len(records) == 3 and journal.read_bytes() == b"".join(lines[:3])
    assert len(list((artifact / "final_test_recovery").glob("tail-*.hex"))) == 1


def test_valid_last_json_without_newline_does_not_lose_record(fixture):
    root, run, _ = fixture
    context = final.validate_finalization(root, RUN_ID)
    rows = final.load_test_rows(context)
    final.evaluate_remaining(context, rows, None, None, {}, generate_fn=fake_generate)
    journal = Path(context["artifact"]) / "final_test_outputs.jsonl"
    lines = journal.read_bytes().splitlines()
    journal.write_bytes(b"\n".join(lines[:3]))
    assert len(final.read_outputs(journal, context)) == 3
    assert journal.read_bytes().endswith(b"\n")
    assert len(final.read_outputs(journal, context)) == 3


def test_corrupt_interior_journal_is_never_silently_skipped(fixture):
    root, run, _ = fixture
    context = final.validate_finalization(root, RUN_ID)
    artifact = Path(context["artifact"])
    artifact.mkdir(parents=True)
    journal = artifact / "final_test_outputs.jsonl"
    journal.write_bytes(b"{broken\n{}\n")
    before = journal.read_bytes()
    with pytest.raises(RuntimeError, match="corrupt interior"):
        final.read_outputs(journal, context)
    assert journal.read_bytes() == before


def test_finalization_refuses_changed_existing_final_csv(fixture):
    root, run, _ = fixture
    context = final.validate_finalization(root, RUN_ID)
    records = final.evaluate_remaining(context, final.load_test_rows(context), None, None, {},
                                       generate_fn=fake_generate)
    artifact = Path(context["artifact"])
    (artifact / "final_scores.csv").write_text("WRONG EXISTING RESULT\n")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        final.seal_result(context, records, {})
    assert (artifact / "final_scores.csv").read_text() == "WRONG EXISTING RESULT\n"


@pytest.mark.parametrize("record,category", [
    ({"full_pass": True, "tier": "full_pass"}, "passed"),
    ({"full_pass": False, "length_cap": True, "tier": "invalid_format"}, "truncation"),
    ({"full_pass": False, "length_cap": False, "tier": "invalid_format"}, "format"),
    ({"full_pass": False, "length_cap": False, "tier": "format_only"}, "parse"),
    ({"full_pass": False, "length_cap": False, "tier": "partial_pass"}, "semantic"),
])
def test_failure_categories(record, category):
    assert final.failure_category(record) == category
