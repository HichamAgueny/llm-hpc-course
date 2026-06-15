#!/bin/bash
#SBATCH -A nn9970k
#SBATCH --reservation=LLM_in_person_course
#SBATCH -p accel
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4             
#SBATCH --mem-per-gpu=97G
#SBATCH -J vllm-distributed
#SBATCH -o ./out/%x-%j.out
#SBATCH -e ./out/%x-%j.err

# Exit on error
set -e
module load NRIS/GPU
module load vLLM/0.11.0

export PYTHONNOUSERSITE=1

# -------- User configuration --------
PROJECT_DIR="/cluster/work/projects/nn9970k"
MyWD="$PROJECT_DIR/$USER/llm-hpc-course"
CURRENT_DIR="${MyWD}/day2_multi_gpu/inference/task_QA"
# Python path for inference
export PYTHON_FILE="${MyWD}/recipes/inference/vllm_distributed.py"

# Set paths
export MODEL_PATH=${MODEL_PATH:-"${MyWD}/shared/models/Llama-3.2-1B-Instruct"}
export LORA_PATH=${LORA_PATH:-"${MyWD}/results/checkpoints_out/llama3_2_1B_lora_multi_device/epoch_0"}
export PROMPT_FILE=${PROMPT_FILE:-"$CURRENT_DIR/prompt_QA.json"}
export QUANTIZATION=${QUANTIZATION:-"None"}  # Set to "bitsandbytes" for QLoRA

# Parallelism settings
export TP_SIZE=${TP_SIZE:-4}   # Tensor parallel = number of GPUs per node (must match --ntasks-per-node)
export PP_SIZE=${PP_SIZE:-1}   # Pipeline parallel (use >1 for multi-node)

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
export MONITOR_LOG="$LOG_DIR/multi_gpu_utilization_${SLURM_JOB_ID}.csv"
python $MyWD/utils/gpu_monitor.py --interval 3 --output "$MONITOR_LOG" &
MONITOR_PID=$!
echo "Started GPU monitor (PID: $MONITOR_PID) logging to $MONITOR_LOG"

# Set up variables to control distributed PyTorch training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25900
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

# --- Create the Inner Script (runs INSIDE container) ---
INNER_SCRIPT_TEMP="./.my_script_temp_${SLURM_JOB_ID}"
cat > "${INNER_SCRIPT_TEMP}" << EOF
#!/bin/bash -e

# Flash Attention for efficiency
export USE_FLASH_ATTENTION=1

# Set up variables to control distributed PyTorch training
export RANK=\$SLURM_PROCID
export LOCAL_RANK=\$SLURM_LOCALID

echo "Task \${SLURM_PROCID}: RANK=${SLURM_PROCID}, LOCAL_RANK=${SLURM_LOCALID}, WORLD_SIZE = $WORLD_SIZE, LOCAL_WORLD_SIZE = $LOCAL_WORLD_SIZE"
echo "LOCAL_RANK: \${LOCAL_RANK}, CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES}"

python "${PYTHON_FILE}" \
    --model "$MODEL_PATH" \
    --lora-path "$LORA_PATH" \
    --prompt-file "$PROMPT_FILE" \
    --tensor-parallel-size "$TP_SIZE" \
    --pipeline-parallel-size "$PP_SIZE"
EOF

chmod +x "${INNER_SCRIPT_TEMP}"

# Run the inference script
time srun "${INNER_SCRIPT_TEMP}"

# --- Clean Up Temporary Script ---
rm -f "${INNER_SCRIPT_TEMP}"

echo "Job finished at: $(date)"

# --- Stop GPU Monitoring ---
echo "Stopping GPU monitor..."
kill $MONITOR_PID
