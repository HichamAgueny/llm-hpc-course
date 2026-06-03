#!/bin/bash -e
#SBATCH --job-name=ft-llama3-1B-lora-1gpu
#SBATCH --account=nn9997k
#SBATCH --time=00:10:00
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
PROJECT_DIR="/cluster/projects/nn9997k"
MyWD="$PROJECT_DIR/$USER/llm-hpc-course"
CONTAINER_DIR="${MyWD}/apptainer"
APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.08_cuda13.0_arm_custom.sif"

# Configs and python files for fine-tuning
CONFIG_FILE="${MyWD}/configs/lora/llama3_2_1B_lora_single_device_QA.yaml"
PYTHON_FILE="${MyWD}/recipes/single_device/lora_finetune_single_device.py"

echo "--- My Main Directory (host): ${MyWD}"
echo "--- Bind-mounted inside container as: /workspace"
echo
echo "=== Running inside Apptainer ==="
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "PYTHON_FILE: ${PYTHON_FILE}"
echo

echo "--- Launching the application inside Apptainer ---"

# --- Execute with Apptainer ---
# Bind host project directory to /workspace inside container
# --nv enables NVIDIA GPU support
time srun apptainer exec --nv \
      -B "${MyWD}:/workspace" \
       -B $PROJECT_DIR \
      "${APPTAINER_SIF}" \
      python "${PYTHON_FILE}" --config "${CONFIG_FILE}"

echo
echo "--- Finished :) ---"
