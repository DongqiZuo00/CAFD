from gap_validation.merge_teacher_responses import select_candidate_prefix
from gap_validation.teacher_responses import prompt_shard, stable_prompt_seed, worker_output_dir


def test_prompt_shards_are_disjoint_and_complete(tmp_path):
    assignments = [prompt_shard(position, 4) for position in range(20)]
    assert assignments == [0, 1, 2, 3] * 5
    assert len({worker_output_dir(tmp_path, index, 4) for index in range(4)}) == 4


def test_prompt_seed_is_stable_and_problem_specific():
    assert stable_prompt_seed("problem-a", 42) == stable_prompt_seed("problem-a", 42)
    assert stable_prompt_seed("problem-a", 42) != stable_prompt_seed("problem-b", 42)


def test_merge_stops_only_after_complete_prompt_batch():
    rows = []
    for position, problem_id in enumerate(("a", "b", "c")):
        for slot in range(4):
            rows.append(
                {
                    "problem_id": problem_id,
                    "stream_position": position,
                    "rollout_slot": slot,
                    "correct": slot < 2,
                }
            )
    selected = select_candidate_prefix(list(reversed(rows)), required_successes=3)
    assert len(selected) == 8
    assert {row["problem_id"] for row in selected} == {"a", "b"}
    assert sum(bool(row["correct"]) for row in selected) == 4
