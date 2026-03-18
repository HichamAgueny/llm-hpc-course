#!/bin/bash
#SBATCH -A nn9970k
#SBATCH -p accel
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1      
#SBATCH --gpus=1              
#SBATCH --mem-per-cpu=80G
#SBATCH -J vllm-lora
#SBATCH -o ./out/%x-%j.out
#SBATCH -e ./out/%x-%j.err

# Exit on error
set -e
module load NRIS/GPU
module load vLLM/0.11.0

# -------- User configuration --------
PROJECT_DIR="/cluster/work/projects/nn9970k"
MyWD="$PROJECT_DIR/$USER/llm-hpc-course"
CURRENT_DIR="${MyWD}/day2_multi_gpu/inference/task_XSum"

# Python path for inference
PYTHON_FILE="${MyWD}/recipes/inference/vllm_inference.py"

# Set paths
MODEL_PATH=${MODEL_PATH:-"${MyWD}/shared/models/Llama-3.2-1B-Instruct"}
LORA_PATH=${LORA_PATH:-"${MyWD}/results/checkpoints_out/llama3_2_1B_lora_single_device/epoch_0"}
PROMPT_FILE=${PROMPT_FILE:-"$CURRENT_DIR/prompt_XSum.json"}

QUANTIZATION=${QUANTIZATION:-"None"}  # Set to "bitsandbytes" for QLoRA

echo "----------------------------------------"
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  LoRA: $LORA_PATH"
echo "  Prompt file: $PROMPT_FILE"
echo "  Quantization: $QUANTIZATION"
echo "----------------------------------------"

# Verify prompt file exists
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "ERROR: Prompt file not found: $PROMPT_FILE"
    exit 1
fi

export VLLM_CACHE_ROOT=$MyWD/.cache/vllm
if [ ! -d "$VLLM_CACHE_ROOT" ]; then
  mkdir -p "$VLLM_CACHE_ROOT"
fi

# Suppress Logs (Set to ERROR to hide Warnings/Info)
export VLLM_LOGGING_LEVEL=ERROR

# Monitoring logs
LOG_DIR="${MyWD}/day2_multi_gpu/inference/logs"
if [ ! -d "$LOG_DIR" ]; then
   mkdir -p ${MyWD}/day2_multi_gpu/inference/logs	
fi
# --- Start GPU Monitoring in the background ---
export MONITOR_LOG="$LOG_DIR/gpu_lora_utilization_xsum_${SLURM_JOB_ID}.csv"
python $MyWD/utils/gpu_monitor.py --interval 2 --output "$MONITOR_LOG" &
MONITOR_PID=$!
echo "Started GPU monitor (PID: $MONITOR_PID) logging to $MONITOR_LOG"

# -------- Launch --------
srun python3 "$PYTHON_FILE" \
          --model "$MODEL_PATH" \
          --lora-path "$LORA_PATH" \
          --prompt-file "$PROMPT_FILE" \
          --quantization "$QUANTIZATION"
	
echo "Job finished at: $(date)"

# --- Stop GPU Monitoring ---
echo "Stopping GPU monitor..."
kill $MONITOR_PID
