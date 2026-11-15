#!/bin/bash

# Exit immediately if any command fails
set -e

# =====================================================================
# 1. HPC Cluster & Apptainer Environment Configuration
# =====================================================================
export PYTHONNOUSERSITE=1

export PROJECT_DIR="/cluster/projects/nn9997k"
export MyWD="$PROJECT_DIR/$USER/llm-hpc-course"

CONTAINER_DIR="${MyWD}/apptainer"
APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.08_cuda13.0_arm_custom.sif"

export CONTAINER_WD="/workspace"
# =====================================================================
# 2. Conversion Asset Paths
# =====================================================================
INPUT_HF_BASE_DIR="$MyWD/shared/models/Llama-3.2-1B-Instruct"
OUTPUT_BASE_GGUF="$MyWD/shared/models/Llama-3.2-1B-Instruct-F16.gguf"
BASE_PRECISION="f16"

INPUT_LORA_DIR="$MyWD/results/checkpoints_out/llama3_2_1B_lora_multi_device/epoch_3"
OUTPUT_LORA_GGUF="$MyWD/results/checkpoints_out/llama3_2_1B-lora-adapter-epoch3_F16.gguf"
LORA_PRECISION="f16"

# Verify the Apptainer image exists before proceeding
if [ ! -f "$APPTAINER_SIF" ]; then
    echo "--Error: Apptainer image not found at $APPTAINER_SIF" >&2
    echo "Please run your setup/download script first." >&2
    exit 1
fi

# Define the shared Apptainer execution runtime wrapper
APPTAINER_RUN="apptainer exec --nv -B ${MyWD}:${CONTAINER_WD} -B $PROJECT_DIR $APPTAINER_SIF"

# =====================================================================
# 3. Interactive Selection Menu
# =====================================================================
echo "----------------------------------------------------------------"
echo " Native Apptainer GGUF Conversion Manager"
echo "----------------------------------------------------------------"
echo "Container Image: $(basename "$APPTAINER_SIF")"
echo "Status: Using pre-installed container packages."
echo "Please select an execution mode:"
echo "  1) Convert BOTH (Base Model & LoRA Adapter)"
echo "  2) Convert BASE MODEL Only  (One-time setup)"
echo "  3) Convert LoRA ADAPTER Only (Frequent/Task-specific)"
echo "----------------------------------------------------------------"
read -r -p "Enter choice [1-3]: " SELECTION

CONVERT_BASE=false
CONVERT_LORA=false

case "$SELECTION" in
    1) CONVERT_BASE=true; CONVERT_LORA=true; echo "-> Selected: Full pipeline conversion.";;
    2) CONVERT_BASE=true; echo "-> Selected: Base model conversion only.";;
    3) CONVERT_LORA=true; echo "-> Selected: LoRA adapter conversion only.";;
    *) echo "-- Error: Invalid selection." >&2; exit 1;;
esac

# =====================================================================
# 3.5. Smart File Existence & Override Verification Layer
# =====================================================================
# Check Base Model file existence
if [ "$CONVERT_BASE" = true ] && [ -f "$OUTPUT_BASE_GGUF" ]; then
    echo -e "\n⚠️ Notice: Base Model conversion is already available at:"
    echo "   $OUTPUT_BASE_GGUF"
    read -r -p "   Do you want to override it? (Yes/No): " OVERRIDE_BASE
    case "$OVERRIDE_BASE" in
        [yY][eE][sS]|[yY])
            echo "-> Confirmed: Existing Base Model GGUF will be overwritten."
            ;;
        *)
            echo "-> Skipping Phase A: Retaining existing Base Model GGUF."
            CONVERT_BASE=false
            ;;
    esac
fi

# Check LoRA Adapter file existence
if [ "$CONVERT_LORA" = true ] && [ -f "$OUTPUT_LORA_GGUF" ]; then
    echo -e "\n⚠️ Notice: LoRA Adapter conversion is already available at:"
    echo "   $OUTPUT_LORA_GGUF"
    read -r -p "   Do you want to override it? (Yes/No): " OVERRIDE_LORA
    case "$OVERRIDE_LORA" in
        [yY][eE][sS]|[yY])
            echo "-> Confirmed: Existing LoRA Adapter GGUF will be overwritten."
            ;;
        *)
            echo "-> Skipping Phase B: Retaining existing LoRA Adapter GGUF."
            CONVERT_LORA=false
            ;;
    esac
fi

# Early Exit Protection: If both actions are skipped/refused, exit cleanly
if [ "$CONVERT_BASE" = false ] && [ "$CONVERT_LORA" = false ]; then
    echo -e "\n================================================================="
    echo "⏩ All conversion tasks skipped or already satisfied. Exiting cleanly."
    echo "================================================================="
    exit 0
fi

# =====================================================================
# 4. Repository Staging
# =====================================================================
echo -e "\n=== Step 1: Checking for llama.cpp conversion blueprints ==="
if [ ! -d "$MyWD/llama.cpp" ]; then
    echo " -> Cloning llama.cpp framework repository..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$MyWD/llama.cpp"
fi

# =====================================================================
# 5. Execution Phase A: Convert Base Model (Conditional)
# =====================================================================
if [ "$CONVERT_BASE" = true ]; then
    echo -e "\n=== Step 2: Running Base Model Conversion Natively inside Container ==="
    echo "----------------------------------------------------------------"
    
    $APPTAINER_RUN python3 "$MyWD/llama.cpp/convert_hf_to_gguf.py" \
        "$INPUT_HF_BASE_DIR" \
        --outfile "$OUTPUT_BASE_GGUF" \
        --outtype "$BASE_PRECISION"

    echo "----------------------------------------------------------------"
    echo "--Base Model Conversion Complete!"
else
    echo -e "\n--> Skipping Phase A: Base Model conversion bypassed."
fi

# =====================================================================
# 6. Execution Phase B: Convert LoRA Adapters (Conditional)
# =====================================================================
if [ "$CONVERT_LORA" = true ]; then
    echo -e "\n=== Step 3: Running LoRA Adapter Conversion Natively inside Container ==="
    echo "----------------------------------------------------------------"
    
    $APPTAINER_RUN python3 "$MyWD/llama.cpp/convert_lora_to_gguf.py" \
        "$INPUT_LORA_DIR" \
        --base "$INPUT_HF_BASE_DIR" \
        --outfile "$OUTPUT_LORA_GGUF" \
        --outtype "$LORA_PRECISION"

    echo "----------------------------------------------------------------"
    echo "--LoRA Adapter Conversion Complete!"
else
    echo -e "\n--> Skipping Phase B: LoRA Adapter conversion bypassed."
fi

# =====================================================================
# 7. Wrap-Up
# =====================================================================
echo "================================================================="
echo "--Native Apptainer workflow execution complete!"
[ "$CONVERT_BASE" = true ] && echo " --Base Model ready at: $OUTPUT_BASE_GGUF"
[ "$CONVERT_LORA" = true ] && echo " --LoRA Adapter ready at: $OUTPUT_LORA_GGUF"
echo "================================================================="
