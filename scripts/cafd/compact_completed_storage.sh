#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_ROOT=/blue/du.j/jinjiaguo/CAFD
readonly ROOT=${1:-$EXPECTED_ROOT}
cd "$ROOT"
if [[ "$(pwd -P)" != "$EXPECTED_ROOT" ]]; then
  echo "refusing cleanup outside $EXPECTED_ROOT" >&2
  exit 64
fi
if squeue -u "$USER" -h -o '%j|%T' | grep -qi cafd; then
  echo "refusing cleanup while a CAFD Slurm job is active" >&2
  exit 65
fi

readonly retained=(
  "runs/cafd/experiments/mistral_cafd_only_v6/student_base/S0"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher_capacity/TBase"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher_capacity/SFT125"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher/candidates/T0"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher/candidates/T20"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher/candidates/T40"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher/candidates/T60"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher/candidates/T80"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher/candidates/T100"
  "runs/cafd/experiments/mistral_cafd_support_full_v9/cafd/step80"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher/route.json"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher/route_rl_only.json"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher/gate.json"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher_capacity/gate.json"
  "runs/cafd/experiments/mistral_cafd_support_full_v9/cafd/selected.json"
  "runs/cafd/experiments/mistral_cafd_support_full_v9/frozen_selection.json"
  "runs/cafd/experiments/mistral_cafd_support_full_v9/final/cafd.json"
  "artifacts/cafd/experiments/mistral_cafd_support_full_v9/final_scores.csv"
  "artifacts/cafd/experiments/mistral_cafd_support_full_v9/costs.csv"
  "artifacts/cafd/experiments/mistral_cafd_support_full_v9/student_curves.csv"
  "reports/cafd_mistral_support_v9.md"
  "reports/cafd_retention_manifest.json"
)
for relative in "${retained[@]}"; do
  if [[ ! -e "$ROOT/$relative" && ! -L "$ROOT/$relative" ]]; then
    echo "missing retained path: $relative" >&2
    exit 66
  fi
done

readonly targets=(
  "runs/cafd/experiments/mistral_cafd_support_full_v9/cafd/step10"
  "runs/cafd/experiments/mistral_cafd_support_full_v9/cafd/step40"
  "runs/cafd/experiments/mistral_cafd_support_full_v9/cafd/step120"
  "runs/cafd/experiments/mistral_cafd_support_full_v9/cafd/step160"
  "runs/cafd/experiments/mistral_cafd_support_full_v9/cafd/step200"
  "runs/cafd/experiments/mistral_cafd_support_full_v9/cafd/phase_references"
  "runs/cafd/experiments/mistral_cafd_support_full_v9/cafd/resume.pt"
  "runs/cafd/experiments/mistral_cafd_support_full_v9/cafd/raw_rollouts.jsonl"
  "runs/cafd/experiments/mistral_cafd_support_probe_v8"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/cafd"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher/raw_rollouts"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher/resume.rank0.pt"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher/resume.rank1.pt"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher/resume.rank2.pt"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher/resume.rank3.pt"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher_capacity/SFT25"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher_capacity/SFT50"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher_capacity/SFT75"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher_capacity/SFT100"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher_capacity/resume.rank0.pt"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher_capacity/resume.rank1.pt"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher_capacity/resume.rank2.pt"
  "runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher_capacity/resume.rank3.pt"
  "runs/cafd/experiments/mistral_cafd_only_v6/capacity"
  "runs/cafd/experiments/mistral_cafd_only_v6/teacher_capacity"
  "runs/cafd/experiments/cafd_only_v5_full_route"
  "runs/cafd/experiments/chat_hier_v3"
  "runs/cafd/experiments/chat_hier_v4_sft25"
  "runs/cafd/experiments/chat_hier_v4b_sft25"
  "runs/cafd/experiments/teacher_capacity_v1"
  "runs/cafd/teacher"
  "runs/cafd/capacity"
  "runs/cafd/capacity_runs"
  "runs/cafd/student_base"
  "runs/cafd/student_base_runs"
)

planned_kib=0
for relative in "${targets[@]}"; do
  target="$ROOT/$relative"
  case "$target" in
    "$ROOT/runs/cafd/"*) ;;
    *)
      echo "unsafe cleanup target: $target" >&2
      exit 67
      ;;
  esac
  if [[ "$target" == "$ROOT/runs/cafd" || "$target" == "$ROOT/runs/cafd/experiments" ]]; then
    echo "refusing broad cleanup target: $target" >&2
    exit 68
  fi
  if [[ -e "$target" || -L "$target" ]]; then
    resolved="$(realpath -m -- "$target")"
    case "$resolved" in
      "$ROOT/runs/cafd/"*) ;;
      *)
        echo "target resolves outside CAFD runs: $target -> $resolved" >&2
        exit 69
        ;;
    esac
    size_kib="$(du -sk -- "$target" | cut -f1)"
    planned_kib=$((planned_kib + size_kib))
  fi
done

echo "validated ${#retained[@]} retained paths"
echo "planned reclaim: $((planned_kib / 1024 / 1024)) GiB"

if [[ "${CAFD_CLEANUP_DRY_RUN:-0}" == "1" ]]; then
  echo "dry run complete; no files removed"
  exit 0
fi

for relative in "${targets[@]}"; do
  target="$ROOT/$relative"
  if [[ -e "$target" || -L "$target" ]]; then
    rm -rf -- "$target"
  fi
done

for relative in "${retained[@]}"; do
  if [[ ! -e "$ROOT/$relative" && ! -L "$ROOT/$relative" ]]; then
    echo "retained path disappeared: $relative" >&2
    exit 70
  fi
done
if find -L   "$ROOT/runs/cafd/experiments/mistral_cafd_support_full_v9"   "$ROOT/runs/cafd/experiments/mistral_cafd_disjoint_v7"   -type l -print -quit | grep -q .; then
  echo "broken retained symlink detected" >&2
  exit 71
fi

echo "storage compaction complete"
du -sh "$ROOT/runs/cafd"
df -h "$ROOT"
