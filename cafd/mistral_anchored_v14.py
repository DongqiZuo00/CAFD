"""Isolated 80-update Residual-Anchored CAFD experiment; selection split only."""
from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
import yaml
from . import anchored_target, train_student
from . import mistral_cafd_generalization_v10 as original

ROOT = Path("/blue/du.j/jinjiaguo/CAFD")
RUN_ID = "mistral_cafd_anchored_v14"
BASE_CONFIG = "configs/cafd/manufactoria_mistral_cafd_gen_v10_mixed.yaml"
LABEL = "Residual-Anchored CAFD"
_original_update = train_student._distillation_update
_current_update_index = None


def generated_rollouts(*args, **kwargs):
    global _current_update_index
    _current_update_index = int(kwargs["update"] if "update" in kwargs else args[4])
    return original._support_rollouts(*args, **kwargs)



def resolved_config(root):
    config = yaml.safe_load((root / BASE_CONFIG).read_text())
    original.validate_config(config)
    assert config["seed"] == 2027
    assert float(config["student"]["learning_rate"]) == 1e-6
    assert config["cafd"]["gamma"] == config["cafd"]["temperature"] == 1.0
    config["experiment"]["run_id"] = RUN_ID
    config["protocol"] = "mistral_residual_anchored_v14"
    config["student"]["updates"] = 80
    config["student"]["milestones"] = [0, 10, 40, 80]
    config["cafd"]["exact_target"] = "(1-alpha)*softmax(z_next)+alpha*softmax(z_ref+z_next-z_prev)"
    config["cafd"]["alpha"] = "per-position centered-vocabulary RMS(delta)/(RMS(delta)+RMS(residual)); zero/zero=0"
    config["pilot_gate"] = {
        "step40_minimum": 20, "step80_minimum": 33,
        "requires_step80_greater_than_step40": True, "step80_aspirational": 40,
        "confirmation_evaluations": 0, "frozen_test_evaluations": 0,
    }
    return config


def load_config(root):
    config = resolved_config(root)
    original._active_config = config
    return config


def measured_update(*args, **kwargs):
    stats = anchored_target.AlphaStats()
    anchored_target.ACTIVE_STATS = stats
    try:
        result = _original_update(*args, **kwargs)
    finally:
        anchored_target.ACTIVE_STATS = None
    rollouts = kwargs.get("rollouts", args[4] if len(args) > 4 else None)
    output = kwargs.get("output", args[9] if len(args) > 9 else None)
    phase = kwargs.get("phase_index", args[7] if len(args) > 7 else None)
    metric = stats.result()
    if _current_update_index is None:
        raise RuntimeError("rollout update was not recorded")
    metric.update(update=_current_update_index + 1, phase=int(phase),
                  loss=result[0], gradient_norm=result[2])
    with (output / "alpha_metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(metric, sort_keys=True) + "\n")
    return result


def prepare(root):
    if root.resolve() != ROOT or Path.cwd().resolve() != ROOT:
        raise RuntimeError("must work in /blue/du.j/jinjiaguo/CAFD")
    config = resolved_config(root)
    run = root / "runs/cafd/experiments" / RUN_ID
    source = root / "runs/cafd/experiments/mistral_cafd_disjoint_v7"
    run.mkdir(parents=True, exist_ok=True)
    (run / "student_base").mkdir(exist_ok=True)
    for link, target in ((run / "student_base/S0", source / "student_base/S0"),
                         (run / "teacher", source / "teacher")):
        if not target.exists():
            raise FileNotFoundError(target)
        if link.exists() or link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"input link mismatch: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
    route = json.loads((source / "teacher/route.json").read_text())
    if route.get("status") != "frozen" or len(route["checkpoints"]) != 6:
        raise RuntimeError("Teacher route is not frozen")
    for item in route["checkpoints"]:
        if not Path(item["checkpoint"]).exists():
            raise FileNotFoundError(item["checkpoint"])
    for name in ("fit", "selection"):
        if not (root / "data/cafd/generalization_v10" / f"{name}.jsonl").is_file():
            raise FileNotFoundError(name)
    frozen = run / "resolved_config.json"
    if frozen.exists() and json.loads(frozen.read_text()) != config:
        raise RuntimeError("existing run config differs")
    train_student.write_json(frozen, config)
    train_student.write_json(run / "provenance.json", {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "job_id": os.environ.get("SLURM_JOB_ID"), "teacher_route": route,
        "base_config": BASE_CONFIG, "method": LABEL,
        "confirmation_accessed": False, "frozen_test_accessed": False,
    })
    return run


def summarize(root):
    artifact = root / "artifacts/cafd/experiments" / RUN_ID
    rows = list(csv.DictReader((artifact / "student_curves.csv").open()))
    scores = {int(r["step"]): int(r["correct"]) for r in rows if r["condition"] == LABEL}
    if set(scores) != {0, 10, 40, 80}:
        raise RuntimeError(f"pilot milestones incomplete: {scores}")
    passed = scores[40] >= 20 and scores[80] >= 33 and scores[80] > scores[40]
    selected = json.loads((root / "runs/cafd/experiments" / RUN_ID / "cafd/selected.json").read_text())
    train_student.write_json(artifact / "pilot_gate.json", {
        "status": "PILOT_PASSED" if passed else "PILOT_NOT_PASSED",
        "actual_updates": 80, "scores": scores, "selection": selected,
        "step80_aspirational_reached": scores[80] >= 40,
        "continuation_eligible": passed, "confirmation_accessed": False,
        "frozen_test_accessed": False,
        "interpretation": "Single-seed selection evidence, not a causal or generalization proof",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run = prepare(args.root)
    original.install_v10()
    train_student.load_experiment_config = load_config
    train_student._generate_rollouts = generated_rollouts
    train_student.iter_relative_kl_blocks = anchored_target.iter_anchored_kl_blocks
    train_student._distillation_update = measured_update
    train_student.METHOD_LABELS["cafd"] = LABEL
    sys.argv = [sys.argv[0], "--root", str(args.root), "--method", "cafd"]
    train_student.main()
    summarize(args.root)


if __name__ == "__main__":
    main()
