#!/bin/zsh
# Failure-profile replays behind RESULTS.md §10: every task-set env, both rewards, seeds 1-3, 60k
# steps each, 6 at a time. Writes runs/diagnostics/profiles/<env>_s<seed>.npy, then DONE.
cd "$(dirname "$0")/../.."
mkdir -p runs/diagnostics/profiles
rm -f runs/diagnostics/profiles/DONE
n=0
for t in T0 T1 T2 T4x T4xneg T4xy; do
  for suf in "" Outcome; do
    for s in 1 2 3; do
      .venv/bin/python scripts/diagnose_timeouts/task_profile.py WamTask${t}${suf}-v0 $s 60000 \
        runs/diagnostics/profiles/WamTask${t}${suf}-v0_s$s.npy > /dev/null 2>&1 &
      n=$((n + 1)); (( n % 6 == 0 )) && wait
    done
  done
done
wait
echo done > runs/diagnostics/profiles/DONE
