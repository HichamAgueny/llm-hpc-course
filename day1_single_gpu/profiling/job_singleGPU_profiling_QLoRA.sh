#!/bin/bash -e
#SBATCH --job-name=profiling_ft-llama3-1B-qlora-1gpu
#SBATCH --account=nn9997k
#SBATCH --time=00:10:00
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
PROJECT_DIR="/cluster/work/projects/nn9997k"
MyWD="$PROJECT_DIR/$USER/llm-hpc-course"
CONTAINER_DIR="${MyWD}/apptainer"
APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.05_cuda12.9_arm_custom.sif"

# Configs and python files for fine-tuning
CONFIG_FILE="${MyWD}/configs/lora/llama3_2_1B_qlora_single_device_profiling.yaml"
PYTHON_FILE="${MyWD}/recipes/single_device/lora_finetune_single_device.py"

# Host-side directories for output/logging
OUTPUT_DIR="${MyWD}/results/profiles/llama3_2_1B_qlora_single_device"
LOGGING_DIR="${MyWD}/results/profiles/logs/qlora_finetune_1B_output"

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

# Run the fine-tuning script
# To override output dirs (optional):
python "${PYTHON_FILE}" --config "${CONFIG_FILE}" checkpointer.output_dir="${OUTPUT_DIR}" output_dir="${LOGGING_DIR}" epochs=1

# Default execution
#python "${PYTHON_FILE}" --config "${CONFIG_FILE}"
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
time srun apptainer exec --nv \
      -B "${MyWD}:/workspace" \
       -B $PROJECT_DIR \
      "${APPTAINER_SIF}" \
      "${INNER_SCRIPT_TEMP}"

# --- Clean Up Temporary Script ---
rm -f "${INNER_SCRIPT_TEMP}"

echo
echo "--- Finished :) ---"
