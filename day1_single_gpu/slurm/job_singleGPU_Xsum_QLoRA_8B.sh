#!/bin/bash -e
#SBATCH --job-name=ft-llama3-1B-qlora-1gpu_Xsum
#SBATCH --account=nn9997k
#SBATCH --time=00:29:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-cpu=8G

echo "--Node: $(hostname)"
echo

# --- Variables and Paths (HOST-SIDE) ---
MyWD="/cluster/projects/nn9997k/$USER/llm-hpc-course"
FINETUNE_DIR="${MyWD}/day1_single_gpu"
CONTAINER_DIR="${MyWD}/setup/apptainer"
APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.05_cuda12.9_arm_custom.sif"

# Host-side directories for output/logging (used for mkdir only)
HOST_OUTPUT_DIR="$MyWD/results/checkpoints_out/llama3_2_1B_qlora_single_device"
HOST_LOGGING_DIR="$MyWD/results/logs/qlora_finetune_1B_output"

# Create directories on the host filesystem (persisted via bind mount)
if [ ! -d "$HOST_OUTPUT_DIR" ]; then
  echo "Creating output directory: $HOST_OUTPUT_DIR"
  mkdir -p "$HOST_OUTPUT_DIR"
fi

if [ ! -d "$HOST_LOGGING_DIR" ]; then
  echo "Creating logging directory: $HOST_LOGGING_DIR"
  mkdir -p "$HOST_LOGGING_DIR"
fi

echo "--- My Main Directory (host): ${MyWD}"
echo "--- Bind-mounted inside container as: /workspace"
echo

# === CONTAINER-PATHS: MyWD is bound to /workspace ===
CONFIG_FILE="/workspace/configs/lora/llama3_1_8B_lora_single_device_XSum.yaml"
PYTHON_FILE="/workspace/recipes/single_device/lora_finetune_single_device.py"
OUTPUT_DIR="/workspace/results/checkpoints_out/llama3_1_8B_lora_single_device"
LOGGING_DIR="/workspace/results/lora_finetune_8B_output"

echo "=== Running inside Apptainer ==="
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "PYTHON_FILE: ${PYTHON_FILE}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "LOGGING_DIR: ${LOGGING_DIR}"
echo

# --- Create the Inner Script (runs INSIDE container) ---
INNER_SCRIPT_TEMP="./.my_script_temp_${SLURM_JOB_ID}"

cat > "${INNER_SCRIPT_TEMP}" << EOF
#!/bin/bash -e

# Flash Attention for efficiency
export USE_FLASH_ATTENTION=1

# Avoid CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Verify critical files exist
if [ ! -f "${PYTHON_FILE}" ]; then
  echo "-- ERROR: Python script not found at ${PYTHON_FILE}"
  exit 1
fi

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "--ERROR: Config file not found at ${CONFIG_FILE}"
  exit 1
fi

echo "--All paths resolved. Starting fine-tuning..."
echo

# Run the fine-tuning script
# To override output dirs (optional):
# python "${PYTHON_FILE}" --config "${CONFIG_FILE}" checkpointer.output_dir="${OUTPUT_DIR}" output_dir="${LOGGING_DIR}" epochs=1

# Default execution
python "${PYTHON_FILE}" --config "${CONFIG_FILE}"
EOF

chmod +x "${INNER_SCRIPT_TEMP}"

# --- Suppress LMOD Debugging ---
export LMOD_SH_DBG_ON=0
# --- Locale Settings ---
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo "--- Launching the application inside Apptainer ---"

# --- Execute with Apptainer ---
# Bind host project directory to /workspace inside container
# --nv enables NVIDIA GPU support
time srun apptainer exec --nv -B "${MyWD}:/workspace" \
      "${APPTAINER_SIF}" \
      "${INNER_SCRIPT_TEMP}"

# --- Clean Up Temporary Script ---
rm -f "${INNER_SCRIPT_TEMP}"

echo
echo "--- Finished :) ---"
