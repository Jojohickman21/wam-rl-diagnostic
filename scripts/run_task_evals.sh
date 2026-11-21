#!/bin/zsh
# Evaluations behind RESULTS.md §9.2: held-out T2, in-distribution tasks, fixture yaw ±20° and ±40°.
# Needs runs/ from snoopy1 (rsync snoopy1:/scr/jhickman/wam-rl-diagnostic/runs/ runs/). Writes one
# line per evaluation to $OUT (default /tmp/task_evals). Run from the repo root.
OUT=${OUT:-/tmp/task_evals}; mkdir -p $OUT
job() { # train_env seed task yaw
  .venv/bin/python scripts/eval_policy.py $1 $2 $3 200 $4 2>/dev/null | tail -1 > $OUT/$1__$2__$3__$4.txt
}
for s in 1 2 3; do
  job WamMultiTask-v0 $s T2 20 & job WamTaskT2-v0 $s T2 20 & job WamTaskT1-v0 $s T2 20 & job WamTaskT4x-v0 $s T2 20 & wait
done
for s in 1 2 3; do
  for t in T0 T1 T4x T4xneg; do job WamMultiTask-v0 $s $t 20 & job WamTask$t-v0 $s $t 20 & done; wait
  for t in T0 T1 T4x T4xneg; do job WamMultiTask-v0 $s $t 40 & done; wait
  job WamMultiTask-v0 $s T2 40 & job WamTaskT2-v0 $s T2 40 & job WamTaskT1-v0 $s T2 40 & job WamTaskT4x-v0 $s T2 40 & wait
done
