#!/bin/sh
# Gated reward against ablation A0 (ungated), 3 seeds each, 2M steps, two runs at a time.
# Usage: sh scripts/run_gated_vs_ungated.sh [suffix]   e.g. "RandomStart" for the randomised start
# Run from the repo root inside the uv environment.
set -e
suffix=${1:-}
for seed in 1 2 3; do
  for env in "WamCloseCabinet$suffix-v0" "WamCloseCabinet${suffix}Ungated-v0"; do
    python scripts/train_ppo.py --env-id "$env" --seed "$seed" \
      --num-envs 8 --num-steps 256 --total-timesteps 2000000 --save-model \
      > "runs/$env-seed$seed.log" 2>&1 &
  done
  wait
done
