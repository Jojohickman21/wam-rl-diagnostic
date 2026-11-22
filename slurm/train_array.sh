#!/bin/bash
# One PPO run per array index; line N of $MANIFEST is "<env id> <seed>". Same command as every
# laptop run (8 envs, 256 steps, --save-model), $STEPS steps (default 2M). Throttle arrays so the
# total stays at 6 runs (60 CPUs), under the lab's half-server rule (64 of 128).
# Preemption: snoopy1 preempts jobs; --requeue puts this one back in the queue, and a restart
# clears this run's unfinished folders first, so a run never leaves duplicates or partial models.
# The progress line records the real exit status and whether a model was saved: a run counts only
# with "model yes".
#SBATCH --cpus-per-task=10
#SBATCH --mem=16gb
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=slurm/%x-%A_%a.out
source /scr/jhickman/miniconda3/etc/profile.d/conda.sh && conda activate wam
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/scr/jhickman/wam-rl-diagnostic
cd /scr/jhickman/wam-rl-diagnostic
read -r ENV SEED < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
mkdir -p runs/aborted
for dir in runs/${ENV}__train_ppo__${SEED}__*; do
  [ -d "$dir" ] && [ ! -f "$dir/train_ppo.cleanrl_model" ] && mv "$dir" runs/aborted/
done
echo "$(date +%H:%M) start $ENV seed $SEED job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID restart ${SLURM_RESTART_COUNT:-0}" >> "$PROGRESS"
srun python scripts/train_ppo.py --env-id "$ENV" --seed "$SEED" --num-envs 8 --num-steps 256 \
  --total-timesteps "${STEPS:-2000000}" --save-model > "runs/$ENV-seed$SEED.log" 2>&1
status=$?
model=no
ls runs/${ENV}__train_ppo__${SEED}__*/train_ppo.cleanrl_model >/dev/null 2>&1 && model=yes
echo "$(date +%H:%M) end $ENV seed $SEED exit $status model $model" >> "$PROGRESS"
