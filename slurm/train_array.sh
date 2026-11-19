#!/bin/bash
# One PPO run per array index; line N of $MANIFEST is "<env id> <seed>". Same command as every
# laptop run (8 envs, 256 steps, 2M steps, --save-model). At most 6 at once (--array=...%6), so
# 60 CPUs, under the lab's half-server rule (64 of 128).
#SBATCH --cpus-per-task=10
#SBATCH --mem=16gb
#SBATCH --time=03:00:00
#SBATCH --output=slurm/%x-%A_%a.out
source /scr/jhickman/miniconda3/etc/profile.d/conda.sh && conda activate wam
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/scr/jhickman/wam-rl-diagnostic
cd /scr/jhickman/wam-rl-diagnostic
read -r ENV SEED < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
echo "$(date +%H:%M) start $ENV seed $SEED job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID" >> "$PROGRESS"
srun python scripts/train_ppo.py --env-id "$ENV" --seed "$SEED" --num-envs 8 --num-steps 256 \
  --total-timesteps "${STEPS:-2000000}" --save-model > "runs/$ENV-seed$SEED.log" 2>&1
echo "$(date +%H:%M) done $ENV seed $SEED exit $?" >> "$PROGRESS"
