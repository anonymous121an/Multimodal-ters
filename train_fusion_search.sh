#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --gpus=1
#SBATCH -c 16
# SBATCH --partition=gpu-h100-80g-short
# SBATCH --partition=gpu-a100-80g
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --array=0-5
#SBATCH -o /scratch/work/sethih1/Multimodal_TERS/slurm_logs_multimodal/fusion_%a_%A.out
#SBATCH -e /scratch/work/sethih1/Multimodal_TERS/slurm_logs_multimodal/fusion_%a_%A.err
#SBATCH --job-name=multimodal_fusion

# =============================================================================
# Multimodal TERS Fusion Strategy Training
# Runs all fusion configurations as a SLURM array job
# Usage: sbatch train_fusion_search.sh
# =============================================================================

# Fusion types array (matches SLURM_ARRAY_TASK_ID 0-5)
FUSION_TYPES=("none" "early" "late" "attention" "film" "hybrid") # "none"
FUSION_TYPE=${FUSION_TYPES[$SLURM_ARRAY_TASK_ID]}

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Fusion Type: $FUSION_TYPE"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=============================================="

# Print GPU info
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Create log directories
mkdir -p /scratch/work/sethih1/slurm_logs_multimodal
mkdir -p /scratch/work/sethih1/checkpoints_multimodal_again

# Load environment
source /scratch/phys/sin/sethih1/venv/multimodal-ters/bin/activate

# Start GPU monitoring
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
         --format=csv -l 30 > /scratch/work/sethih1/slurm_logs_multimodal/gpu_usage_${FUSION_TYPE}.log &
GPU_MONITOR_PID=$!

# Start resource monitoring
vmstat -n 30 > /scratch/work/sethih1/slurm_logs_multimodal/resource_usage_${FUSION_TYPE}.log &
RESOURCE_MONITOR_PID=$!


SUFFIX="freq-bin-100-b16"
# Training configuration
TRAIN_PATH="/scratch/phys/sin/sethih1/Multimodal_TERS/planar_hdf5_0.05/train.h5"
VAL_PATH="/scratch/phys/sin/sethih1/Multimodal_TERS/planar_hdf5_0.05/val.h5"
SAVE_DIR="/scratch/work/sethih1/checkpoints_multimodal_ters_${SUFFIX}"
EPOCHS=100
BATCH_SIZE=16
LR=1e-4
NUM_CHANNELS=100
MAX_FREQS=${MAX_FREQS:-100}
LOSS_FN="dice_loss"
SEED=${SEED:-42}
FREQ_ENCODING="binning"
#FREQ_ENCODING=${FREQ_ENCODING:-normalize}

# WANDB config: set these as needed
WANDB_PROJECT=${WANDB_PROJECT:-multimodal-ters-${SUFFIX}}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-fusion_${FUSION_TYPE}} # _${SLURM_JOB_ID}}
WANDB_API_KEY=8e4e0db2307a46c329b7d30d5f7ab11a176ba158
# Run training
echo ""
echo "Starting training with fusion_type=$FUSION_TYPE..."
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "WANDB_RUN_NAME: $WANDB_RUN_NAME"
echo "SEED: $SEED"
echo "MAX_FREQS: $MAX_FREQS"
echo "FREQ_ENCODING: $FREQ_ENCODING"
echo ""

# Export WANDB environment variables for python
export WANDB_PROJECT
export WANDB_RUN_NAME


# Export WANDB_API_KEY if it is set
if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY
fi

# Reproducibility settings
export PYTHONHASHSEED=$SEED
export CUBLAS_WORKSPACE_CONFIG=:4096:8


python /scratch/work/sethih1/Multimodal_TERS/train_multimodal.py \
    --fusion_type $FUSION_TYPE \
    --train_path $TRAIN_PATH \
    --val_path $VAL_PATH \
    --save_dir $SAVE_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --num_channels $NUM_CHANNELS \
    --max_freqs $MAX_FREQS \
    --loss_fn $LOSS_FN \
    --num_workers 8 \
    --seed $SEED \
    --freq_encoding $FREQ_ENCODING

# Capture exit code
EXIT_CODE=$?

# Stop monitoring
kill $GPU_MONITOR_PID 2>/dev/null
kill $RESOURCE_MONITOR_PID 2>/dev/null

echo ""
echo "=============================================="
echo "Fusion Type: $FUSION_TYPE"
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_CODE
