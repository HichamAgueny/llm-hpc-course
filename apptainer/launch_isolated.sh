#!/bin/bash -e
# --- Configuration ---
export PROJECT_DIR="/cluster/work/projects/nn9970k"
export MyWD="$PROJECT_DIR/$USER/llm-hpc-course"
CONTAINER_DIR="${MyWD}/apptainer"
APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.05_cuda12.9_arm_custom.sif"

# --- Launch (NO -B FLAGS) ---
echo "--Launching Isolated Container (No Project Mounts)..."
echo "--Try running 'ls /workspace' inside. It will fail!"
apptainer shell --nv \
      "${APPTAINER_SIF}"
