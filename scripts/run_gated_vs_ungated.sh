#!/bin/sh
# Gated reward against ablation A0 (ungated), 3 seeds each, 2M steps, two runs at a time.
# Usage: sh scripts/run_gated_vs_ungated.sh   (from the repo root, inside the uv environment)
set -e
for seed in 1 2 3; do
  for env in WamCloseCabinet-v0 WamCloseCabinetUngated-v0; do
    python scripts/train_ppo.py --env-id "$env" --seed "$seed" \
      --num-envs 8 --num-steps 256 --total-timesteps 2000000 --save-model \
      > "runs/$env-seed$seed.log" 2>&1 &
  done
  wait
done
