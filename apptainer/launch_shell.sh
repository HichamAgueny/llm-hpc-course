#!/bin/bash -e
# Open an interactive shell inside the PyTorch course container
# (run it on a GPU node, e.g. after: bash apptainer/srun_interactive).
# The course folder is mounted at its own path and as /workspace.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/course_env.sh"

apptainer shell --nv ${CONTAINER_BINDS} \
      --env MyWD="${COURSE_ROOT}" \
      "${PYTORCH_SIF}"
