#!/bin/bash -e
export PROJECT_DIR="/cluster/work/projects/nn9970k"
export MyWD="$PROJECT_DIR/$USER/llm-hpc-course"
export CONTAINER_WD="/workspace"
CONTAINER_DIR="${MyWD}/apptainer"
APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.08_cuda13.0_arm_custom.sif"

echo " Testing: Project Mount (-B PROJECT_DIR)"
apptainer shell --nv \
      # -B "${MyWD}:${CONTAINER_WD}" \    
      -B "${PROJECT_DIR}" \               
      #--env MyWD="$MyWD" \
      "${APPTAINER_SIF}"
