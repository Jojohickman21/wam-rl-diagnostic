#!/bin/bash
# One Warp PPO run per array index on snoopy1: seed = line N of $MANIFEST (one seed per line), name $EXP, extra
# trainer flags $WARP_ARGS. Uses a GPU shard (all 8 A6000s are shared) and 4 CPUs (the start pool is drawn by the C
# env on CPU, once per seed, then cached). --requeue plus train_ppo_warp.py's checkpoints make a preempted run resume
# from its last checkpoint instead of restarting. Submit with sbatch so a dropped ssh/VPN session cannot kill it.
#SBATCH --gres=shard:16
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=slurm/%x-%A_%a.out
source /scr/jhickman/miniconda3/etc/profile.d/conda.sh && conda activate /scr/jhickman/conda_envs/wamwarp
export OMP_NUM_THREADS=1 PYTHONPATH=/scr/jhickman/wam-rl-diagnostic
cd /scr/jhickman/wam-rl-diagnostic
SEED=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
echo "$(date +%H:%M) start warp $EXP seed $SEED job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID restart ${SLURM_RESTART_COUNT:-0}" >> "$PROGRESS"
python scripts/train_ppo_warp.py --seed "$SEED" --exp "$EXP" $WARP_ARGS >> "runs/warp-$EXP-seed$SEED.log" 2>&1
status=$?
model=no
[ -f "runs/WamFamilyContactFK-v0__${EXP}__${SEED}__w/${EXP}.cleanrl_model" ] && model=yes
echo "$(date +%H:%M) end warp $EXP seed $SEED exit $status model $model" >> "$PROGRESS"
