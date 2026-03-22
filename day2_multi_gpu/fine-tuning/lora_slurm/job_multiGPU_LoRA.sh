#!/bin/bash -e
#SBATCH --job-name=ft-llama3-8B-lora-4gpu
#SBATCH --account=nn9970k
#SBATCH --time=00:10:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-gpu=96G

echo "--Node: $(hostname)"
echo

# --- Variables and Paths (HOST-SIDE) ---
PROJECT_DIR="/cluster/work/projects/nn9970k"
MyWD="$PROJECT_DIR/$USER/llm-hpc-course"
CONTAINER_DIR="${MyWD}/apptainer"
APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.05_cuda12.9_arm_custom.sif"

# Configs and python files for fine-tuning
CONFIG_FILE="${MyWD}/configs/lora/llama3_1_8B_lora_multi_device_alpaca_gpt4.yaml"
PYTHON_FILE="${MyWD}/recipes/distributed/lora_finetune_distributed.py"

# Host-side directories for output/logging
OUTPUT_DIR="${MyWD}/results/checkpoints_out/llama3_1_8B_lora_multi_device"
LOGGING_DIR="${MyWD}/results/logs/lora_finetune_8B_output"

# Create directories on the host filesystem (persisted via bind mount)
if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Creating output directory: $OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
fi

if [ ! -d "$LOGGING_DIR" ]; then
  echo "Creating logging directory: $LOGGING_DIR"
  mkdir -p "$LOGGING_DIR"
fi

echo "--- My Main Directory (host): ${MyWD}"
echo "--- Bind-mounted inside container as: /workspace"
echo

echo "=== Running inside Apptainer ==="
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "PYTHON_FILE: ${PYTHON_FILE}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "LOGGING_DIR: ${LOGGING_DIR}"
echo

# --- Slurm setting
N=$SLURM_JOB_NUM_NODES
nproc_perN=$SLURM_NTASKS_PER_NODE
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "--nbr of nodes: $N"
echo "--nbr of GPUs: $nproc_perN"
echo

# Set up variables to control distributed PyTorch training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25900
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

echo "--SLURM_PROCID: $SLURM_PROCID"
echo

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

# Run the fine-tuning script
# To override output dirs (optional):
#python "${PYTHON_FILE}" --config "${CONFIG_FILE}" checkpointer.checkpoint_dir="${OUTPUT_DIR}"

# Default execution
python "${PYTHON_FILE}" --config "${CONFIG_FILE}"

# Syntax of "tune run" command
#the flag --standalone is Useful when launching single-node, multi-worker job
#If --standalone specified then the options --rdzv-backend, --rdzv-endpoint, --rdzv-id are auto-assigned and any explicitly set values are ignored.

#tune run --nnodes $N --nproc_per_node $nproc_perN --standalone lora_finetune_distributed --config "${CONFIG_FILE}" checkpointer.output_dir="${OUTPUT_DIR}" output_dir="${LOGGING_DIR}" epochs=1
EOF

chmod +x "${INNER_SCRIPT_TEMP}"

# --- Suppress LMOD Debugging ---
export LMOD_SH_DBG_ON=0
# --- Locale Settings ---
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo "--- Launching the application inside Apptainer ---"
# CPU affinity
CPU_BIND="map_cpu:1,73,145,217"

# --- Execute with Apptainer ---
# Bind host project directory to /workspace inside container
# --nv enables NVIDIA GPU support

time srun --cpu-bind=${CPU_BIND} apptainer exec --nv \
      -B "${MyWD}:/workspace" \
      -B $PROJECT_DIR \
      "${APPTAINER_SIF}" \
      "${INNER_SCRIPT_TEMP}"

# --- Clean Up Temporary Script ---
rm -f "${INNER_SCRIPT_TEMP}"

echo
echo "--- Finished :) ---"
