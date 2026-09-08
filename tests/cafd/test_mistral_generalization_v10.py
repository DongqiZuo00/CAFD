import json
from collections import Counter
from pathlib import Path

import yaml

from cafd import mistral_cafd_generalization_v10 as v10
from cafd.generalization_split import load_nested_rows, prepare


CONFIGS = (
    "configs/cafd/manufactoria_mistral_cafd_gen_v10_teacher.yaml",
    "configs/cafd/manufactoria_mistral_cafd_gen_v10_mixed.yaml",
)


def test_v10_configs_are_matched_and_forbid_frozen_test() -> None:
    configs = [yaml.safe_load(Path(path).read_text(encoding="utf-8")) for path in CONFIGS]
    for config in configs:
        v10.validate_config(config)
        assert config["student"]["updates"] == 200
        assert config["student"]["prompts_per_update"] == 4
        assert config["student"]["rollouts_per_prompt"] == 8
        assert config["final"]["official_held_out_evaluations"] == 0
        assert "qwen" not in json.dumps(config).lower()
    assert configs[0]["student"] == configs[1]["student"]
    assert configs[0]["models"] == configs[1]["models"]
    assert configs[0]["cafd"]["exact_target"] == configs[1]["cafd"]["exact_target"]
    assert v10._teacher_counts(configs[0]) == [8, 8, 8, 8, 8]
    assert v10._teacher_counts(configs[1]) == [6, 4, 4, 2, 2]


def test_nested_split_is_exact_stratified_and_does_not_need_test(tmp_path) -> None:
    counts = {
        "contains_count": 146,
        "contains_ordered": 268,
        "contains_substring": 264,
    }
    rows = []
    for family, count in counts.items():
        rows.extend(
            {"id": f"{family}-{index}", "problem_family": family}
            for index in range(count)
        )
    source = tmp_path / "data" / "cafd" / "train.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = prepare(tmp_path)
    assert manifest["fit_count"] == 614
    assert manifest["selection_count"] == 64
    assert manifest["frozen_test_accessed"] is False
    assert not (tmp_path / "data" / "cafd" / "test.jsonl").exists()
    fit = load_nested_rows(tmp_path, "fit")
    selection = load_nested_rows(tmp_path, "selection")
    assert not ({row["id"] for row in fit} & {row["id"] for row in selection})
    assert Counter(row["problem_family"] for row in selection) == {
        "contains_count": 14,
        "contains_ordered": 25,
        "contains_substring": 25,
    }
    assert prepare(tmp_path) == manifest


def test_mixed_support_keeps_exactly_eight_slots_per_prompt(monkeypatch) -> None:
    config = yaml.safe_load(Path(CONFIGS[1]).read_text(encoding="utf-8"))
    monkeypatch.setattr(v10, "_active_config", config)
    student = object()
    teachers = {f"R{index}": object() for index in range(1, 6)}
    monkeypatch.setattr(v10, "_support_models", teachers)
    calls = []

    def fake_generate(model, tokenizer, rows, indices, update, settings, *rest):
        calls.append((model, int(settings["rollouts_per_prompt"])))
        result = []
        for prompt_slot, _ in enumerate(indices):
            for sample_slot in range(int(settings["rollouts_per_prompt"])):
                result.append({
                    "row_id": str(prompt_slot),
                    "prompt_ids": [1],
                    "completion_ids": [2],
                    "prompt_slot": prompt_slot,
                    "sample_slot": sample_slot,
                })
        return result

    monkeypatch.setattr(v10, "_base_generate_rollouts", fake_generate)
    for phase, expected_teacher in enumerate([6, 4, 4, 2, 2]):
        calls.clear()
        rollouts = v10._support_rollouts(
            student, object(), [{}, {}], [0, 1], phase * 40,
            {"rollouts_per_prompt": 8}, 2048, object(), "renderer",
        )
        assert calls == [
            (teachers[f"R{phase + 1}"], expected_teacher),
            (student, 8 - expected_teacher),
        ]
        assert len(rollouts) == 16
        assert Counter(item["prompt_slot"] for item in rollouts) == {0: 8, 1: 8}
        assert {item["sample_slot"] for item in rollouts if item["prompt_slot"] == 0} == set(range(8))
