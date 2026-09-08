"""CPU-only synthetic filesystem tests for CAFD-MPC data/preflight."""
import json
from pathlib import Path

import pytest

from cafd.mpc_data import (
    FAMILIES, OUTPUT, ROUTE, SOURCE, STUDENT_BASE, TEACHER_SPLIT,
    checkpoint_metadata, load_mpc_rows, prepare,
)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_checkpoint(path):
    write_json(path / "config.json", {"model_type": "ministral3", "vocab_size": 131072})
    write_json(path / "tokenizer_config.json", {"eos_token": "</s>"})
    write_json(path / "tokenizer.json", {"model": {"type": "BPE"}})
    write_json(path / "model.safetensors.index.json",
               {"weight_map": {"model.weight": "model-00001-of-00001.safetensors"}})
    with (path / "model-00001-of-00001.safetensors").open("wb") as handle:
        handle.truncate(1_000_001)


@pytest.fixture
def root(tmp_path):
    fit = [{"id": f"fit-{i}", "problem_family": FAMILIES[i % 3],
            "messages": [{"role": "user", "content": f"optimization {i}"}],
            "ground_truth": [{"input": "", "expected_accepted": True}]}
           for i in range(614)]
    dev = [{"id": f"dev-{i}", "problem_family": FAMILIES[i % 3],
            "messages": [{"role": "user", "content": f"development {i}"}],
            "ground_truth": []} for i in range(64)]
    write_rows(tmp_path / SOURCE / "fit.jsonl", fit)
    write_rows(tmp_path / SOURCE / "selection.jsonl", dev)
    write_json(tmp_path / SOURCE / "manifest.json", {
        "status": "frozen", "fit_ids": [row["id"] for row in fit],
        "selection_ids": [row["id"] for row in dev]})
    write_rows(tmp_path / "data/cafd/test.jsonl", [
        {"id": f"test-{i}", "messages": "TEST_CONTENT_SENTINEL",
         "ground_truth": "TEST_ANSWER_SENTINEL"} for i in range(132)])
    write_rows(tmp_path / "data/cafd/development.jsonl", [
        {"id": f"confirmation-{i}", "ground_truth": "UNUSED_CONFIRMATION"} for i in range(64)])
    write_json(tmp_path / TEACHER_SPLIT, {
        "status": "frozen",
        "teacher_sft_ids": [row["id"] for row in fit[:468] + dev[:44]],
        "teacher_rlvr_ids": [row["id"] for row in fit[468:] + dev[44:]],
    })
    checkpoints = []
    for index, step in enumerate(["raw_instruct", "sft125", "rl40", "rl60", "rl80", "rl100"]):
        path = tmp_path / f"fixtures/checkpoint{index}"
        make_checkpoint(path)
        checkpoints.append({"checkpoint": str(path), "step": step})
    make_checkpoint(tmp_path / STUDENT_BASE)
    write_json(tmp_path / ROUTE, {
        "status": "frozen",
        "kind": "mistral_full_acquisition_raw_instruct_to_sft_to_rl",
        "checkpoints": checkpoints})
    return tmp_path


def test_prepare_freezes_602_12_64_and_preserves_sources(root):
    source_paths = [root / SOURCE / "fit.jsonl", root / SOURCE / "selection.jsonl",
                    root / "data/cafd/test.jsonl", root / TEACHER_SPLIT]
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in source_paths}
    manifest = prepare(root)
    assert {name: entry["count"] for name, entry in manifest["partitions"].items()} == {
        "optimization": 602, "control": 12, "development": 64}
    assert manifest["partitions"]["control"]["family_counts"] == dict.fromkeys(FAMILIES, 4)
    assert prepare(root) == manifest
    for path, previous in before.items():
        assert (path.stat().st_mtime_ns, path.read_bytes()) == previous
    groups = [set(entry["ids"]) for entry in manifest["partitions"].values()]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
    assert len(load_mpc_rows(root, "optimization")) == 602
    assert len(load_mpc_rows(root, "control")) == 12
    assert len(load_mpc_rows(root, "development")) == 64


def test_test_ids_only_and_historical_limitations_are_honest(root):
    manifest = prepare(root)
    test_meta = json.loads((root / OUTPUT / "test_ids.json").read_text())
    assert test_meta["ids"] == [f"test-{i}" for i in range(132)]
    assert test_meta["labels_used"] is False and test_meta["evaluated"] is False
    assert test_meta["historically_evaluated"] is True
    assert manifest["official_test"]["frozen_test_access_scope"] == "problem_ids_only"
    assert manifest["official_test"]["untouched_test_claim"] is False
    assert manifest["historical_exposure"]["teacher_sft_seen_development_count"] == 44
    assert manifest["historical_exposure"]["teacher_rl_seen_development_count"] == 20
    assert manifest["historical_exposure"]["whole_pipeline_held_out_claim"] is False
    assert manifest["unused_confirmation"]["used_for_mpc"] is False
    for path in (root / OUTPUT).glob("*"):
        if path.is_file():
            contents = path.read_text()
            assert "TEST_CONTENT_SENTINEL" not in contents
            assert "TEST_ANSWER_SENTINEL" not in contents
            assert "UNUSED_CONFIRMATION" not in contents


@pytest.mark.parametrize("split", ["test", "official_test", "../test", "confirmation", "fit", ""])
def test_loader_refuses_test_and_undeclared_splits_before_io(root, split):
    with pytest.raises(ValueError, match="cannot load"):
        load_mpc_rows(root, split)


def test_seed_is_immutable_once_frozen(root):
    prepare(root)
    with pytest.raises(RuntimeError, match="already frozen differently"):
        prepare(root, seed=2028)


def test_duplicate_fit_id_rejected(root):
    path = root / SOURCE / "fit.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["id"] = rows[0]["id"]
    write_rows(path, rows)
    with pytest.raises(ValueError, match="duplicate"):
        prepare(root)


@pytest.mark.parametrize("source", ["test.jsonl", "development.jsonl"])
def test_test_or_confirmation_id_overlap_rejected(root, source):
    path = root / "data/cafd" / source
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["id"] = "fit-0"
    write_rows(path, rows)
    with pytest.raises(ValueError, match="overlap"):
        prepare(root)


def test_source_fit_order_must_match_frozen_manifest(root):
    path = root / SOURCE / "fit.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.reverse()
    write_rows(path, rows)
    with pytest.raises(ValueError, match="frozen ID order"):
        prepare(root)


def test_does_not_republish_changed_existing_output(root):
    prepare(root)
    path = root / OUTPUT / "control.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["messages"] = [{"role": "user", "content": "tampered"}]
    write_rows(path, rows)
    with pytest.raises(RuntimeError, match="refusing to replace"):
        prepare(root)


def test_loader_detects_changed_ids(root):
    prepare(root)
    path = root / OUTPUT / "control.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["id"] = "not-a-control-task"
    write_rows(path, rows)
    with pytest.raises(ValueError, match="IDs changed"):
        load_mpc_rows(root, "control")


def test_missing_route_shard_rejected(root):
    path = root / "fixtures/checkpoint2/model-00001-of-00001.safetensors"
    path.unlink()
    with pytest.raises(FileNotFoundError):
        prepare(root)


def test_preflight_never_reads_weight_payload(root, monkeypatch):
    real_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path.name.endswith((".safetensors", ".bin")):
            raise AssertionError("attempted weight payload read")
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    manifest = prepare(root)
    assert all(not item["weight_hash_computed"] for item in manifest["preflight"]["checkpoints"])


def test_non_mistral_checkpoint_rejected(root):
    path = root / "fixtures/checkpoint0/config.json"
    write_json(path, {"model_type": "qwen3", "vocab_size": 131072})
    with pytest.raises(ValueError, match="non-Mistral"):
        prepare(root)


def test_checkpoint_shard_escape_rejected(root):
    path = root / "fixtures/checkpoint0/model.safetensors.index.json"
    write_json(path, {"weight_map": {"weight": "../checkpoint1/model.safetensors"}})
    with pytest.raises(ValueError, match="unsafe shard"):
        prepare(root)


def test_checkpoint_path_outside_root_rejected(root):
    with pytest.raises(ValueError, match="escapes"):
        checkpoint_metadata(root, root.parent / "outside-checkpoint")


def test_acquisition_order_cannot_be_replaced_by_repetition(root):
    path = root / ROUTE
    route = json.loads(path.read_text())
    route["checkpoints"][3] = dict(route["checkpoints"][2])
    write_json(path, route)
    with pytest.raises(ValueError, match="acquisition order"):
        prepare(root)


def test_loader_detects_changed_training_payload_without_hashing(root):
    prepare(root)
    path = root / OUTPUT / "control.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["ground_truth"] = [{"wrong": "label"}]
    write_rows(path, rows)
    with pytest.raises(ValueError, match="payload changed"):
        load_mpc_rows(root, "control")


def test_malformed_shard_mapping_rejected(root):
    path = root / "fixtures/checkpoint0/model.safetensors.index.json"
    write_json(path, {"weight_map": {"weight": ["not-a-string"]}})
    with pytest.raises(ValueError, match="unsafe shard"):
        prepare(root)
