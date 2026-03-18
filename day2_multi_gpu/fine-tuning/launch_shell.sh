#!/bin/bash -e
export PROJECT_DIR="/cluster/work/projects/nn9997k"
export MyWD="$PROJECT_DIR/$USER/llm-hpc-course"
export CONTAINER_WD="/workspace"

CONTAINER_DIR="${MyWD}/apptainer"
APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.05_cuda12.9_arm_custom.sif"

apptainer shell --nv \
      -B "${MyWD}:${CONTAINER_WD}" \
      -B $PROJECT_DIR \
      --env MyWD="$PROJECT_DIR/$USER/llm-hpc-course" \
      "${APPTAINER_SIF}"

