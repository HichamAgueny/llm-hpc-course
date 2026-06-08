#!/bin/bash -e
#SBATCH --job-name=ft-llama3-1B-qlora-1gpu
#SBATCH --reservation=LLM_in_person_course
#SBATCH --account=nn9970k
#SBATCH --time=00:15:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-gpu=97G

echo "--Node: $(hostname)"
echo

# --- Variables and Paths (HOST-SIDE) ---
PROJECT_DIR="/cluster/work/projects/nn9970k"
MyWD="$PROJECT_DIR/$USER/llm-hpc-course"
CONTAINER_DIR="${MyWD}/apptainer"
APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.08_cuda13.0_arm_custom.sif"

# Configs and python files for fine-tuning
CONFIG_FILE="${MyWD}/configs/qlora/llama3_2_1B_qlora_single_device_QA.yaml"
PYTHON_FILE="${MyWD}/recipes/single_device/lora_finetune_single_device.py"

echo "--- My Main Directory (host): ${MyWD}"
echo "--- Bind-mounted inside container as: /workspace"
echo
echo "=== Running inside Apptainer ==="
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "PYTHON_FILE: ${PYTHON_FILE}"
echo

# --- Create the Inner Script (runs INSIDE container) ---
INNER_SCRIPT_TEMP="./.my_script_temp_${SLURM_JOB_ID}"

cat > "${INNER_SCRIPT_TEMP}" << EOF
#!/bin/bash -e

# Flash Attention for efficiency
export USE_FLASH_ATTENTION=1

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
time srun apptainer exec --nv \
      -B "${MyWD}:/workspace" \
       -B $PROJECT_DIR \
      "${APPTAINER_SIF}" \
      "${INNER_SCRIPT_TEMP}"

# --- Clean Up Temporary Script ---
rm -f "${INNER_SCRIPT_TEMP}"

echo
echo "--- Finished :) ---"
