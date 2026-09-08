"""Synthetic-file tests: no real test labels, GPU, model load or Slurm submit."""
from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path

import pytest

from cafd import minimal_baseline_finalize as final

RUN_ID = "mistral_cafd_minimal_grpo_s2027"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def checkpoint(path):
    write_json(path / "config.json", {"model_type": "ministral3", "vocab_size": 131072})
    write_json(path / "tokenizer_config.json", {"eos_token": "</s>"})
    write_json(path / "tokenizer.json", {"model": {"type": "BPE"}})
    with (path / "model.safetensors").open("wb") as handle:
        handle.truncate(1_000_001)


def _training_round(round_number, *, active, condition="grpo"):
    groups = []
    for _ in range(4):
        rewards = ([.1, .2] * 4) if active else [1.] * 8
        flags = [False] * 8 if active else [True] * 8
        groups.append(dict(rewards=rewards, full_pass=flags, completion_tokens=[2] * 8,
                           valid_tokens=16, use_rl=active,
                           use_kd=condition == "absolute_kd" and active))
    return dict(round=round_number, optimizer_step=active, group_routing=groups,
                normalization_tokens=64)


def _write_curve(root, run_id, curve):
    path = root / "artifacts/cafd/experiments" / run_id / "training_curve.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in curve))


def _read_curve(root, run_id=RUN_ID):
    path = root / "artifacts/cafd/experiments" / run_id / "training_curve.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _convert_absolute(fixture):
    root, old, rows = fixture
    run_id = "mistral_cafd_minimal_absolute_kd_s2027"
    run = old.with_name(run_id)
    old.rename(run)
    old_art = root / "artifacts/cafd/experiments" / RUN_ID
    old_art.rename(old_art.with_name(run_id))
    mutate(run / "config.json", condition="absolute_kd", method="Absolute Teacher KD + RL")
    selected = json.loads((run / "selected.json").read_text())
    selected["checkpoint"] = str(run / "round40")
    write_json(run / "selected.json", selected)
    complete = json.loads((run / "complete.json").read_text())
    complete["selected"]["checkpoint"] = str(run / "round40")
    write_json(run / "complete.json", complete)
    _write_curve(root, run_id, [_training_round(i, active=i <= 190, condition="absolute_kd")
                               for i in range(1, 201)])
    return root, run, run_id


@pytest.fixture
def fixture(tmp_path):
    run = tmp_path / "runs/cafd/experiments" / RUN_ID
    cp = run / "round40"
    checkpoint(cp)
    checkpoint(tmp_path / final.S0_REL)
    cfg = {
        "condition": "grpo", "method": "GRPO",
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
    write_json(tmp_path / "data/cafd/mpc_v1/manifest.json", manifest)
    curve = [_training_round(i, active=i <= 190) for i in range(1, 201)]
    _write_curve(tmp_path, RUN_ID, curve)
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
    artifact.mkdir(parents=True, exist_ok=True)
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


def test_s0_prepare_is_idempotent_and_zero_training(fixture):
    root, _, _ = fixture
    first = final.prepare_s0(root)
    run = Path(first["run"])
    names = ["config.json", "data_manifest.json", "evaluation_plan.json"]
    before = {name: ((run / name).read_bytes(), (run / name).stat().st_mtime_ns) for name in names}
    assert final.prepare_s0(root) == first
    assert all(((run / name).read_bytes(), (run / name).stat().st_mtime_ns) == old
               for name, old in before.items())
    assert first["identity"]["training_rounds"] == 0
    assert first["identity"]["actual_optimizer_updates"] == 0
    assert first["identity"]["selection"]["selection_rule"] == "predeclared_original_S0"
    assert first["training_costs"]["gpu_hours"] == 0
    assert first["training_costs"]["training_generated_tokens"] == 0
    assert not (run / "complete.json").exists()


def test_s0_metadata_admission_does_not_read_test_payload(fixture, monkeypatch):
    root, _, _ = fixture
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.resolve() == (root / final.TEST_REL).resolve():
            raise AssertionError("S0 preparation read test payload")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    final.prepare_s0(root)
    assert final.validate_finalization(root, final.S0_RUN)["identity"]["training_rounds"] == 0


@pytest.mark.parametrize("field,value", [("checkpoint", "foreign"), ("test_ids", []),
    ("training_rounds", 200), ("actual_optimizer_updates", 1)])
def test_s0_predeclared_plan_cannot_change(fixture, field, value):
    root, _, _ = fixture
    context = final.prepare_s0(root)
    mutate(Path(context["run"]) / "evaluation_plan.json", **{field: value})
    with pytest.raises(RuntimeError, match="changed"):
        final.validate_finalization(root, final.S0_RUN)


def test_s0_full_132_test_aggregate_and_seal(fixture):
    root, _, _ = fixture
    context = final.prepare_s0(root)
    costs = {}
    calls = []
    def generate(model, tokenizer, row, **kwargs):
        calls.append(row["id"])
        assert kwargs["seed"] == 2027 + 3000000 + len(calls) - 1
        return fake_generate(model, tokenizer, row, **kwargs)
    records = final.evaluate_remaining(context, final.load_test_rows(context), None, None, costs,
                                       generate_fn=generate)
    result = final.seal_result(context, records, costs)
    score = result["scores"]
    assert score["total"] == len(calls) == 132
    assert score["correct"] == 66
    assert score["training_rounds"] == score["actual_optimizer_updates"] == score["selected_round"] == 0
    assert score["condition"] == "Original S0 (no training)"
    assert score["training_gpu_hours"] == 0
    assert score["training_generated_tokens"] == score["training_teacher_scored_tokens"] == 0
    assert sum(r["correct"] for r in result["by_family"]) == 66
    assert final.evaluate_remaining(context, final.load_test_rows(context), None, None, costs,
                                   generate_fn=generate) == records
    assert len(calls) == 132
    assert final.seal_result(context, records, costs) == result


def test_completed_s0_cli_uses_no_torch_or_test_payload(fixture, monkeypatch):
    import builtins
    import sys
    root, _, _ = fixture
    context = final.prepare_s0(root)
    records = final.evaluate_remaining(context, final.load_test_rows(context), None, None, {},
                                       generate_fn=fake_generate)
    final.seal_result(context, records, {})
    original_open = Path.open
    original_import = builtins.__import__
    def guarded_open(path, *args, **kwargs):
        if path.resolve() == (root / final.TEST_REL).resolve():
            raise AssertionError("complete S0 reread test labels")
        return original_open(path, *args, **kwargs)
    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("complete S0 imported torch")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(final, "ROOT", root)
    monkeypatch.chdir(root)
    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "argv", ["minimal_baseline_finalize", "--run-id", final.S0_RUN])
    final.main()
    progress = json.loads((Path(context["artifact"]) / "final_test_progress.json").read_text())
    assert progress["status"] == "complete" and progress["completed"] == 132


@pytest.mark.parametrize("changes", [{"condition": "absolute_kd"}, {"condition": "s0"},
    {"condition": None}, {"max_new_tokens": 4096}, {"prompts_per_round": 8}])
def test_trained_condition_and_protocol_changes_rejected(fixture, changes):
    root, run, _ = fixture
    mutate(run / "config.json", **changes)
    with pytest.raises(RuntimeError):
        final.validate_finalization(root, RUN_ID)


@pytest.mark.parametrize("actual", [0, 189, 191, True, 190.0, -1])
def test_recorded_update_count_must_equal_actual_curve(fixture, actual):
    root, run, _ = fixture
    mutate(run / "complete.json", actual_updates=actual)
    with pytest.raises(RuntimeError):
        final.validate_finalization(root, RUN_ID)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "extra", "out_of_order",
    "not_boolean", "pseudo_skip", "wrong_denominator", "wrong_group_count",
    "wrong_answer_count", "wrong_routing", "wrong_token_sum"])
def test_bad_training_curve_is_rejected(fixture, mode):
    root, _, _ = fixture
    curve = _read_curve(root)
    if mode == "missing": curve.pop()
    elif mode == "duplicate": curve[-1]["round"] = 199
    elif mode == "extra": curve.append(copy.deepcopy(curve[-1]))
    elif mode == "out_of_order": curve[0], curve[1] = curve[1], curve[0]
    elif mode == "not_boolean": curve[0]["optimizer_step"] = 1
    elif mode == "pseudo_skip":
        curve[0]["optimizer_step"] = False
        curve[-1]["optimizer_step"] = True
    elif mode == "wrong_denominator": curve[0]["normalization_tokens"] += 1
    elif mode == "wrong_group_count": curve[0]["group_routing"].pop()
    elif mode == "wrong_answer_count": curve[0]["group_routing"][0]["rewards"].pop()
    elif mode == "wrong_routing": curve[0]["group_routing"][0]["use_kd"] = True
    elif mode == "wrong_token_sum": curve[0]["group_routing"][0]["valid_tokens"] += 1
    _write_curve(root, RUN_ID, curve)
    with pytest.raises(RuntimeError):
        final.validate_finalization(root, RUN_ID)


def test_grpo_zero_variance_all_failed_is_legal_skip(fixture):
    root, _, _ = fixture
    curve = _read_curve(root)
    for group in curve[-1]["group_routing"]:
        group.update(rewards=[.05] * 8, full_pass=[False] * 8, use_kd=False, use_rl=False)
    _write_curve(root, RUN_ID, curve)
    assert final.validate_finalization(root, RUN_ID)["identity"]["actual_optimizer_updates"] == 190


def test_absolute_all_failed_equal_reward_needs_kd(fixture):
    root, run, run_id = _convert_absolute(fixture)
    curve = _read_curve(root, run_id)
    for group in curve[0]["group_routing"]:
        group.update(rewards=[.05] * 8, full_pass=[False] * 8, use_kd=True, use_rl=False)
    _write_curve(root, run_id, curve)
    assert final.validate_finalization(root, run_id)["identity"]["actual_optimizer_updates"] == 190
    curve[0]["optimizer_step"] = False
    curve[-1]["optimizer_step"] = True
    _write_curve(root, run_id, curve)
    with pytest.raises(RuntimeError, match="eligible"):
        final.validate_finalization(root, run_id)


def test_s0_cfg_and_model_metadata_change_rejected(fixture):
    root, _, _ = fixture
    context = final.prepare_s0(root)
    mutate(root / final.S0_REL / "config.json", altered=True)
    with pytest.raises(RuntimeError, match="changed"):
        final.validate_finalization(root, final.S0_RUN)


@pytest.mark.parametrize("bad", [-1, 2.0, True, 2049])
def test_completion_lengths_must_be_valid_bounded_integers(fixture, bad):
    root, _, _ = fixture
    curve = _read_curve(root)
    group = curve[0]["group_routing"][0]
    old_total = group["valid_tokens"]
    group["completion_tokens"][0] = bad
    group["valid_tokens"] = sum(group["completion_tokens"])
    curve[0]["normalization_tokens"] += group["valid_tokens"] - old_total
    _write_curve(root, RUN_ID, curve)
    with pytest.raises(RuntimeError):
        final.validate_finalization(root, RUN_ID)


@pytest.mark.parametrize("bad", [-.1, 1.1, float("nan"), True])
def test_rewards_must_be_finite_nonboolean_probabilities(fixture, bad):
    root, _, _ = fixture
    curve = _read_curve(root)
    curve[0]["group_routing"][0]["rewards"][3] = bad
    _write_curve(root, RUN_ID, curve)
    with pytest.raises(RuntimeError):
        final.validate_finalization(root, RUN_ID)


def test_full_success_flags_cannot_disagree_with_rewards(fixture):
    root, _, _ = fixture
    curve = _read_curve(root)
    # This would otherwise look like a legitimate no-step all-success group.
    curve[-1]["group_routing"][0]["rewards"] = [.1] * 8
    _write_curve(root, RUN_ID, curve)
    with pytest.raises(RuntimeError):
        final.validate_finalization(root, RUN_ID)


@pytest.mark.parametrize("changes", [
    {"output_stop": "different_stopping_rule"},
    {"reward": "full_pass_only"},
])
def test_frozen_stopping_and_verifier_protocol_cannot_change(fixture, changes):
    root, run, _ = fixture
    mutate(run / "config.json", **changes)
    with pytest.raises(RuntimeError):
        final.validate_finalization(root, RUN_ID)
