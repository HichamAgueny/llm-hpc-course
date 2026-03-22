#!/bin/bash -e
export PROJECT_DIR="/cluster/work/projects/nn9970k"
export MyWD="$PROJECT_DIR/$USER/llm-hpc-course"
export CONTAINER_WD="/workspace"
CONTAINER_DIR="${MyWD}/apptainer"
APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.05_cuda12.9_arm_custom.sif"

echo " Testing: Project Mount (-B PROJECT_DIR)"
apptainer shell --nv \
      # -B "${MyWD}:${CONTAINER_WD}" \    # REMOVED: Personal files not at /workspace
      -B "${PROJECT_DIR}" \               # ACTIVE: Mounts shared data to original path
      #--env MyWD="$MyWD" \
      "${APPTAINER_SIF}"
