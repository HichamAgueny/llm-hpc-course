Here is your updated script.

I refactored the Apptainer copy section into a loop so it cleanly handles both `.sif` files, and I streamlined the directory creation block to target just your three new base paths.

```bash
#!/bin/bash

# ==============================================================================
# Environment Setup Script for LLM-HPC-Course
# ==============================================================================
# This script initializes the project environment by copying shared resources
# and creating necessary directory structures for results, logs, and profiles.
# ==============================================================================

# 1. Define Base Directories
# Source: Original shared location
SOURCE_BASE="/cluster/work/projects/nn9970k/hicham/llm-hpc-course"

# Target: Current user's project location
TARGET_BASE="/cluster/work/projects/nn9970k/$USER/llm-hpc-course"

echo "----------------------------------------------------------------"
echo "Initializing Environment Setup"
echo "Source: $SOURCE_BASE"
echo "Target: $TARGET_BASE"
echo "----------------------------------------------------------------"

# 2. Create Target Base and Apptainer Directories
echo "[1/4] Creating core directories..."
mkdir -p "$TARGET_BASE/apptainer"

# 3. Copy Shared Folder and Apptainer Images
echo "[2/4] Copying shared resources..."
if [ -d "$SOURCE_BASE/shared" ]; then
    echo " -> Copying shared/ directory..."
    cp -r "$SOURCE_BASE/shared" "$TARGET_BASE/"
else
    echo " !! Warning: Source shared folder not found at $SOURCE_BASE/shared"
fi

echo "[3/4] Copying Apptainer images..."
SIF_FILES=(
    "pytorch_25.08_cuda13.0_arm_custom.sif"
    "vllm0.12_cu131_py3.12_arm_custom.sif"
)

for SIF_FILE in "${SIF_FILES[@]}"; do
    if [ -f "$SOURCE_BASE/apptainer/$SIF_FILE" ]; then
        echo " -> Copying $SIF_FILE..."
        cp "$SOURCE_BASE/apptainer/$SIF_FILE" "$TARGET_BASE/apptainer/"
    else
        echo " !! Warning: Apptainer image not found at $SOURCE_BASE/apptainer/$SIF_FILE"
    fi
done

# 4. Create Results, Logs, and Profiling Paths
echo "[4/4] Creating results, logs, and profiling structures..."
PATHS=(
    "$TARGET_BASE/results/checkpoints_out"
    "$TARGET_BASE/results/logs"
    "$TARGET_BASE/results/profiles"
)

for p in "${PATHS[@]}"; do
    mkdir -p "$p"
done

echo "----------------------------------------------------------------"
echo "Setup Complete!"
echo "You can now navigate to $TARGET_BASE and start the labs."
echo "----------------------------------------------------------------"

```
