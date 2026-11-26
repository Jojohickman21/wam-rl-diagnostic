#!/bin/zsh
# Evaluations behind the G1-G3 batch (TASK_SET.md): every family policy with a saved model,
# zero-shot on T1, T2, T4x and T4xneg. 200 episodes, fixture yaw ±20°. Existing results are kept, so
# rerunning only adds seeds that have finished since.
cd "$(dirname "$0")/.."
setopt +o nomatch
OUT=runs/diagnostics/gen_evals; mkdir -p $OUT
list=()
for s in 1 2 3 4 5 6 7 8; do
  for env in WamFamilyDesc-v0 WamFamilyContact-v0 WamFamilyContactFK-v0; do
    ls runs/${env}__train_ppo__${s}__*/train_ppo.cleanrl_model >/dev/null 2>&1 || continue
    for t in T2 T1 T4x T4xneg; do list+=("$env $s $t"); done
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
