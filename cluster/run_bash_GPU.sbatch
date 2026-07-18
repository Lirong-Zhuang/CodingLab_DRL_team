#!/bin/bash
#SBATCH -N 1
#SBATCH -p lrz-dgx-1-v100x8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -o slurm_logs/slurm-%x-%j.out
#SBATCH -e slurm_logs/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=lirong.zhuang@tum.de
#SBATCH --time=24:00:00

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$PROJECT_DIR/tf_container_dgx-a100-80.sqsh}"

MAIN_ARGS=(
  --network_version="${NETWORK_VERSION}"
  --env_version="${ENV_VERSION}"
  --model_version="${MODEL_VERSION}"
  --num_episodes="${NUM_EPISODES}"
  --variant="${VARIANT}"
  --seed="${SEED}"
  --data_dir=/mnt/code/data
)

if [[ -n "${ENCODER_PATH:-}" ]]; then
  MAIN_ARGS+=(--encoder_path="${ENCODER_PATH}")
fi

if [[ -n "${ENCODER_TYPE:-}" ]]; then
  MAIN_ARGS+=(--encoder_type="${ENCODER_TYPE}")
fi

if [[ "${FREEZE_ENCODER:-0}" == "1" || "${FREEZE_ENCODER:-}" == "true" || "${FREEZE_ENCODER:-}" == "TRUE" ]]; then
  MAIN_ARGS+=(--freeze_encoder)
fi

srun --mpi=pmi2 \
  --container-mounts="${PROJECT_DIR}:/mnt/code" \
  --container-image="${CONTAINER_IMAGE}" \
  --container-workdir=/mnt/code \
  python main.py \
  "${MAIN_ARGS[@]}"
