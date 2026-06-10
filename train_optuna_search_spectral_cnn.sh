#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --array=0-3
#SBATCH -o /scratch/phys/sin/sethih1/Multimodal-ters/slurm_logs_optuna_spectral_cnn/optuna_%a_%A.out
#SBATCH -e /scratch/phys/sin/sethih1/Multimodal-ters/slurm_logs_optuna_spectral_cnn/optuna_%a_%A.err
#SBATCH --job-name=optuna_spectral_cnn

# =============================================================================
# Multimodal TERS — Optuna Search with SpectralCNNEncoder
#
# Same structure as train_optuna_search.sh but uses the 1D CNN frequency
# encoder (SpectralCNNEncoder, 100 bins) via optuna_search_spectral_cnn.yaml.
#
# Fusion types (array index -> fusion):
#   0 -> none
#   1 -> early
#   2 -> late
#   3 -> attention
#
# Usage:
#   sbatch train_optuna_search_spectral_cnn.sh
# =============================================================================

FUSION_TYPES=("none" "early" "late" "attention")
FUSION_TYPE=${FUSION_TYPES[$SLURM_ARRAY_TASK_ID]}

echo "=============================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Array Task ID:   $SLURM_ARRAY_TASK_ID"
echo "Fusion Type:     $FUSION_TYPE"
echo "Freq Encoder:    spectral_cnn (100 bins)"
echo "Node:            $SLURMD_NODENAME"
echo "Start Time:      $(date)"
echo "=============================================="

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# --- Paths ---
script_dir=/scratch/phys/sin/sethih1/Multimodal-ters
log_dir="$script_dir/slurm_logs_optuna_spectral_cnn"
optuna_dir="$script_dir/optuna_studies_spectral_cnn"
checkpoint_dir="$script_dir/model_checkpoints/optuna_spectral_cnn/optuna_${FUSION_TYPE}"

mkdir -p "$log_dir"
mkdir -p "$optuna_dir"
mkdir -p "$checkpoint_dir"

# --- Environment ---
source /scratch/phys/sin/sethih1/venv/multimodal-ters/bin/activate

# --- GPU / resource monitoring ---
nvidia-smi \
    --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
    --format=csv -l 30 \
    > "$log_dir/gpu_usage_${FUSION_TYPE}_${SLURM_JOB_ID}.log" &
GPU_MONITOR_PID=$!

vmstat -n 30 \
    > "$log_dir/resource_usage_${FUSION_TYPE}_${SLURM_JOB_ID}.log" &
RESOURCE_MONITOR_PID=$!

# --- WandB ---
WANDB_PROJECT=${WANDB_PROJECT:-multimodal-ters-optuna-spectral-cnn}
WANDB_API_KEY=8e4e0db2307a46c329b7d30d5f7ab11a176ba158

export WANDB_PROJECT
if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY
fi

# --- Reproducibility ---
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo ""
echo "Starting Optuna search for fusion_type=$FUSION_TYPE ..."
echo "WandB project:   $WANDB_PROJECT"
echo "WandB group:     optuna_${FUSION_TYPE}"
echo "Optuna storage:  $optuna_dir/study_${FUSION_TYPE}.db"
echo "Checkpoints:     $checkpoint_dir"
echo ""

python "$script_dir/train_multimodal_optuna.py" \
    --fusion_type "$FUSION_TYPE" \
    --config "$script_dir/src/configs/optuna_search_spectral_cnn.yaml"

EXIT_CODE=$?

# --- Stop monitoring ---
kill $GPU_MONITOR_PID 2>/dev/null
kill $RESOURCE_MONITOR_PID 2>/dev/null

echo ""
echo "=============================================="
echo "Fusion Type:  $FUSION_TYPE"
echo "Exit Code:    $EXIT_CODE"
echo "End Time:     $(date)"
echo "=============================================="

exit $EXIT_CODE
