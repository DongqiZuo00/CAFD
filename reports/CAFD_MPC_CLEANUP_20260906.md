# CAFD-MPC preparation: recoverable-cache cleanup

2026-09-06. Requested cleanup completed before implementing the new method.

Only four Qwen download-cache directories under
/blue/du.j/jinjiaguo/CAFD/.cache/huggingface/hub were removed.
Their resolved paths were verified to remain inside CAFD and not be directory
symlinks. There were no active CAFD jobs at the time of removal.

| Cache | Allocated bytes before removal |
|---|---:|
| models--Qwen--Qwen3.5-2B | 22,949,888 |
| models--Qwen--Qwen3.5-9B | 19,329,409,024 |
| models--Qwen--Qwen3-1.7B | 4,079,538,176 |
| models--Qwen--Qwen3-4B-Instruct-2507 | 8,061,014,016 |

Total removed: 31,492,911,104 bytes = approximately 29.33 GiB.
The files were permanently unlinked, but these are public downloadable caches,
not irreplaceable experimental checkpoints. Restoring them would require another
download. The user has banned Qwen from subsequent CAFD work.

Preserved: all experimental runs/resume files, reports, raw rollouts, Mistral
caches, S0 and every retained Teacher route checkpoint. No files in other projects
were changed. Shared /blue/du.j filesystem availability was about 4.1 TiB after
cleanup; that shared free-space reading is not a per-user quota guarantee.

New MPC data/code/results use separate paths. Previous CAFD results must not be
reported as evaluations of this new method.
