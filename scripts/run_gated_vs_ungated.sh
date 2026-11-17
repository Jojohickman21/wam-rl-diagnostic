#!/bin/sh
# Three seeds of each named env, 2M steps, two runs at a time. Run from the repo root inside the
# uv environment.
#   sh scripts/run_gated_vs_ungated.sh WamCloseCabinet-v0 WamCloseCabinetUngated-v0
set -e
for seed in 1 2 3; do
  for env in "$@"; do
    python scripts/train_ppo.py --env-id "$env" --seed "$seed" \
      --num-envs 8 --num-steps 256 --total-timesteps 2000000 --save-model \
      > "runs/$env-seed$seed.log" 2>&1 &
  done
  wait
done
