#!/bin/bash -e
# Open the PyTorch course container WITHOUT any folder mounted,
# to show why bind mounts (-B) are needed.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/course_env.sh"

# --- Launch (NO -B FLAGS) ---
echo "--Launching Isolated Container (No Project Mounts)..."
echo "--Try running 'ls /workspace' inside. It will fail!"
apptainer shell --nv \
      "${PYTORCH_SIF}"
