#!/bin/zsh
# Failure-profile replays behind RESULTS.md §10.5: (a) every policy on held-out T2, 8 seeds, 30k
# steps each, into runs/diagnostics/profiles_t2/; (b) single-task seeds 4-8 on their own tasks,
# 60k steps each, into runs/diagnostics/profiles/ (seeds 1-3 are there from §10.3). 6 at a time.
cd "$(dirname "$0")/../.."
mkdir -p runs/diagnostics/profiles_t2 runs/diagnostics/profiles
rm -f runs/diagnostics/profiles_t2/DONE
jobs_list=()
for s in 1 2 3 4 5 6 7 8; do
  for env in WamMultiTask-v0 WamMultiTaskLabel-v0 WamTaskT1-v0 WamTaskT4x-v0 WamTaskT2-v0; do
    jobs_list+=("$env $s 30000 runs/diagnostics/profiles_t2/${env}_s$s.npy T2")
  done
done
for s in 4 5 6 7 8; do
  for env in WamTaskT1-v0 WamTaskT2-v0 WamTaskT4x-v0; do
    jobs_list+=("$env $s 60000 runs/diagnostics/profiles/${env}_s$s.npy")
  done
done
n=0
for job in "${jobs_list[@]}"; do
  set -- ${=job}
  [ -s "$4" ] && continue
  .venv/bin/python scripts/diagnose_timeouts/task_profile.py "$@" > /dev/null 2>&1 &
  n=$((n + 1)); (( n % 6 == 0 )) && wait
done
wait
echo "$n replays" > runs/diagnostics/profiles_t2/DONE
