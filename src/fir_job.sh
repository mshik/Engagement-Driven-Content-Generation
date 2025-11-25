#!/bin/bash
#SBATCH --job-name=socllm
#SBATCH --account=${ACCOUNT:-def-youraccount}
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=FAIL

set -euo pipefail

module load StdEnv/2020
module load python/3.10

# Try to source a conda installation if available; fall back to module-provided conda.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
  source "$HOME/.conda/etc/profile.d/conda.sh"
elif [ -n "${EBROOTANACONDA3:-}" ] && [ -f "$EBROOTANACONDA3/etc/profile.d/conda.sh" ]; then
  source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
else
  echo "Conda installation not found. Please load/initialize conda on fir." >&2
  exit 1
fi

ENV_NAME=${ENV_NAME:-socLLM}
conda activate "$ENV_NAME"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}
REPO_DIR=${REPO_DIR:-$SLURM_SUBMIT_DIR}
cd "$REPO_DIR/src"

RUN_TARGET=${RUN_TARGET:-synthetic}
case "$RUN_TARGET" in
  synthetic)
    bash synthetic_config_pipeline.sh
    ;;
  real-brexit)
    bash real_brexit_config_pipeline.sh
    ;;
  *)
    python3 sentiment-propagation.py "$@"
    ;;

esac
