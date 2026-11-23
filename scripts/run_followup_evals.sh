#!/bin/zsh
# Evaluations behind RESULTS.md §10: held-out T2 for every seed (1-8) of every policy that has a
# saved model, and the descriptor and label policies on their four training tasks. 200 episodes,
# fixture yaw ±20°. Results already present are kept, so rerunning only fills in new seeds.
cd "$(dirname "$0")/.."
setopt +o nomatch  # unfinished seeds have no model yet; skip them quietly
OUT=runs/diagnostics/evals; mkdir -p $OUT
has_model() { ls runs/$1__train_ppo__$2__*/train_ppo.cleanrl_model >/dev/null 2>&1; }
list=()
for s in 1 2 3 4 5 6 7 8; do
  for env in WamMultiTask-v0 WamMultiTaskLabel-v0 WamTaskT1-v0 WamTaskT4x-v0 WamTaskT2-v0; do
    has_model $env $s && list+=("$env $s T2")
  done
  for env in WamMultiTask-v0 WamMultiTaskLabel-v0; do
    has_model $env $s && for t in T0 T1 T4x T4xneg; do list+=("$env $s $t"); done
  done
done
n=0
for job in "${list[@]}"; do
  set -- ${=job}
  [ -s $OUT/$1__$2__$3__20.txt ] && continue
  .venv/bin/python scripts/eval_policy.py $1 $2 $3 200 20 2>/dev/null | tail -1 > $OUT/$1__$2__$3__20.txt &
  n=$((n + 1)); (( n % 8 == 0 )) && wait
done
wait
echo "$n evaluations run, $(ls $OUT/*.txt | wc -l | tr -d ' ') on disk"
