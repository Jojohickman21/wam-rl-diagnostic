#!/bin/zsh
# Evaluations behind RESULTS.md §10: held-out T2 for every complete seed of every policy, and the
# descriptor and label policies on their four training tasks. Only seeds with a saved model are
# listed (the rest were preempted on snoopy1 and are being rerun). 200 episodes, yaw ±20°.
cd "$(dirname "$0")/.."
OUT=runs/diagnostics/evals; mkdir -p $OUT
list=()
for s in 1 2 3 4; do list+=("WamMultiTask-v0 $s T2"); done
for s in 2 3 4 5; do list+=("WamMultiTaskLabel-v0 $s T2"); done
for s in 1 2 3 4 5 6 7; do list+=("WamTaskT1-v0 $s T2"); done
for s in 1 2 3 4 5 6; do list+=("WamTaskT4x-v0 $s T2" "WamTaskT2-v0 $s T2"); done
for t in T0 T1 T4x T4xneg; do
  for s in 1 2 3 4; do list+=("WamMultiTask-v0 $s $t"); done
  for s in 2 3 4 5; do list+=("WamMultiTaskLabel-v0 $s $t"); done
done
n=0
for job in "${list[@]}"; do
  set -- ${=job}
  [ -s $OUT/$1__$2__$3__20.txt ] && continue
  .venv/bin/python scripts/eval_policy.py $1 $2 $3 200 20 2>/dev/null | tail -1 > $OUT/$1__$2__$3__20.txt &
  n=$((n + 1)); (( n % 8 == 0 )) && wait
done
wait
