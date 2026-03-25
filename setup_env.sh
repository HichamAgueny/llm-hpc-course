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
# (Adapting to the project code specified by the user: nn9970k)
TARGET_BASE="/cluster/work/projects/nn9970k/$USER/llm-hpc-course"

echo "----------------------------------------------------------------"
echo "Initializing Environment Setup"
echo "Source: $SOURCE_BASE"
echo "Target: $TARGET_BASE"
echo "----------------------------------------------------------------"

# 2. Create Target Base and Apptainer Directories
echo "[1/4] Creating core directories..."
mkdir -p "$TARGET_BASE/apptainer"

# 3. Copy Shared Folder and Apptainer Image
echo "[2/4] Copying shared resources..."
if [ -d "$SOURCE_BASE/shared" ]; then
    echo " -> Copying shared/ directory..."
    cp -r "$SOURCE_BASE/shared" "$TARGET_BASE/"
else
    echo " !! Warning: Source shared folder not found at $SOURCE_BASE/shared"
fi

echo "[3/4] Copying Apptainer image..."
SIF_FILE="pytorch_25.05_cuda12.9_arm_custom.sif"
if [ -f "$SOURCE_BASE/apptainer/$SIF_FILE" ]; then
    echo " -> Copying $SIF_FILE..."
    cp "$SOURCE_BASE/apptainer/$SIF_FILE" "$TARGET_BASE/apptainer/"
else
    echo " !! Warning: Apptainer image not found at $SOURCE_BASE/apptainer/$SIF_FILE"
fi

# 4. Create Results, Logs, and Profiling Paths
echo "[4/4] Creating results, logs, and profiling structures..."
PATHS=(
    "$TARGET_BASE/results/checkpoints_out/llama3_1_8B_lora_multi_device"
    "$TARGET_BASE/results/checkpoints_out/llama3_1_8B_qlora_multi_device"
    "$TARGET_BASE/results/checkpoints_out/llama3_2_1B_lora_single_device"
    "$TARGET_BASE/results/checkpoints_out/llama3_2_1B_qlora_single_device"
    "$TARGET_BASE/results/logs/lora_finetune_1B_output"
    "$TARGET_BASE/results/logs/lora_finetune_8B_output"
    "$TARGET_BASE/results/logs/qlora_finetune_1B_output"
    "$TARGET_BASE/results/logs/qlora_finetune_8B_output"
    "$TARGET_BASE/results/profiles/profiling_outputs"
    "$TARGET_BASE/results/profiles/profiling_outputs_multigpu"
    "$TARGET_BASE/results/profiles/logs"
)

for p in "${PATHS[@]}"; do
    mkdir -p "$p"
done

echo "----------------------------------------------------------------"
echo "Setup Complete!"
echo "You can now navigate to $TARGET_BASE and start the labs."
echo "----------------------------------------------------------------"
