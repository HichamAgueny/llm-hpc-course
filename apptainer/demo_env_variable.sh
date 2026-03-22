#!/bin/bash -e
export PROJECT_DIR="/cluster/work/projects/nn9970k"
export MyWD="$PROJECT_DIR/$USER/llm-hpc-course"
export CONTAINER_WD="/workspace"
CONTAINER_DIR="${MyWD}/apptainer"
APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.05_cuda12.9_arm_custom.sif"

echo "Testing: Environment Variable (--env MyWD)"
apptainer shell --nv \
      -B "${MyWD}:${CONTAINER_WD}" \      # ACTIVE: Files are mounted
      -B "${PROJECT_DIR}" \               # ACTIVE: Project is mounted
      # --env MyWD="$MyWD" \              # REMOVED: Variable not passed inside
      "${APPTAINER_SIF}"
