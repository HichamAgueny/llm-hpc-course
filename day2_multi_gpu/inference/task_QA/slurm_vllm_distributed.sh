#!/bin/bash
#SBATCH -A nn9970k
#SBATCH -p accel
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4             
#SBATCH --mem-per-gpu=96G
#SBATCH -J vllm-distributed
#SBATCH -o ./out/%x-%j.out
#SBATCH -e ./out/%x-%j.err

# Exit on error
set -e
module load NRIS/GPU
module load CUDA/12.9.1
module load vLLM/0.11.0

# -------- User configuration --------
PROJECT_DIR="/cluster/work/projects/nn9970k"
MyWD="$PROJECT_DIR/$USER/llm-hpc-course"
CURRENT_DIR="${MyWD}/day2_multi_gpu/inference/task_QA"
# Python path for inference
PYTHON_FILE="${MyWD}/recipes/inference/vllm_distributed.py"

# Set paths
MODEL_PATH=${MODEL_PATH:-"${MyWD}/shared/models/Llama-3.1-8B-Instruct"}
LORA_PATH=${LORA_PATH:-"${MyWD}/results/checkpoints_out/llama3_1_8B_lora_multi_device/epoch_0"}
PROMPT_FILE=${PROMPT_FILE:-"$CURRENT_DIR/prompt_QA.json"}
QUANTIZATION=${QUANTIZATION:-"None"}  # Set to "bitsandbytes" for QLoRA

# Parallelism settings
TP_SIZE=${TP_SIZE:-4}   # Tensor parallel = number of GPUs per node (must match --ntasks-per-node)
PP_SIZE=${PP_SIZE:-1}   # Pipeline parallel (use >1 for multi-node)

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

# Set up variables to control distributed PyTorch training
export RANK=\$SLURM_PROCID
export LOCAL_RANK=\$SLURM_LOCALID

echo "Task \${SLURM_PROCID}: RANK=${SLURM_PROCID}, LOCAL_RANK=${SLURM_LOCALID}, WORLD_SIZE = $WORLD_SIZE, LOCAL_WORLD_SIZE = $LOCAL_WORLD_SIZE"
echo "LOCAL_RANK: \${LOCAL_RANK}, CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES}"

# Run the inference script
python "${PYTHON_FILE}" \
    --model "$MODEL_PATH" \
    --lora-path "$LORA_PATH" \
    --prompt-file "$PROMPT_FILE"
    --tensor-parallel-size "$TP_SIZE" \
    --pipeline-parallel-size "$PP_SIZE"  
#
# Syntax of "tune run" command
#the flag --standalone is Useful when launching single-node, multi-worker job
#If --standalone specified then the options --rdzv-backend, --rdzv-endpoint, --rdzv-id are auto-assigned and any explicitly set values are ignored.

#tune run --nnodes $N --nproc_per_node $nproc_perN --standalone lora_finetune_distributed --config "${CONFIG_FILE}" checkpointer.output_dir="${OUTPUT_DIR}" output_dir="${LOGGING_DIR}" epochs=1
EOF

chmod +x "${INNER_SCRIPT_TEMP}"

# --- Suppress LMOD Debugging ---
export LMOD_SH_DBG_ON=0
srun "${INNER_SCRIPT_TEMP}"

echo "Job finished at: $(date)"

# --- Stop GPU Monitoring ---
echo "Stopping GPU monitor..."
kill $MONITOR_PID
