#!/bin/bash -e
#SBATCH --job-name=mmlu_eval-llama3.2-1B-lora-1gpu_epoch0
#SBATCH --account=nn9971k
#SBATCH --time=00:30:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-gpu=96G
#SBATCH --reservation=nn9971k_213

echo "--Node: $(hostname)"
echo

export PYTHONNOUSERSITE=1
export NUMEXPR_MAX_THREADS=72

# =====================================================================
# --- CONFIGURABLE VARIABLES AND PATHS (USERS ONLY MODIFY THIS BLOCK) ---
# =====================================================================

# 1. Base Project Allocation
export CLUSTER_PROJECT_ROOT="/cluster/work/projects/nn9971k"

# 2. Personal Workspace Setup
export PROJECT_DIR="${CLUSTER_PROJECT_ROOT}/${USER}/llm-hpc-course"
export SHARED_DIR="${PROJECT_DIR}/shared"
export RESULT_DIR="${PROJECT_DIR}/results"

# 3. Model & Adapter Component Paths
export BASE_MODEL_DIR="${SHARED_DIR}/models/Llama-3.2-1B-Instruct"
export LORA_ADAPTER_DIR="${RESULT_DIR}/checkpoint_results/llama3.2-1b_lora_single_device/checkpoints/epoch_1"
export MERGED_MODEL_DIR="${RESULT_DIR}/checkpoint_results/Llama-3.2-1B-Instruct_merged"
export TOKENIZER_DIR="${BASE_MODEL_DIR}/original/tokenizer.model"
export MAX_SHARD_SIZE="5GB"

# 4. Output Component Paths
export OUTPUT_DIR="${RESULT_DIR}/evaluation_results/llama3.2-1b-mmlu_singlegpu"
mkdir -p "$OUTPUT_DIR"

# 5. Script / Config Layout
export CONFIG_FILE="${PROJECT_DIR}/configs/eval_cfg/eval_config.yaml"
export PYTHON_FILE_EVAL="${PROJECT_DIR}/recipes/eval/eleuther_eval.py"
export PYTHON_FILE_MERGE="${PROJECT_DIR}/recipes/merge/merge_and_save.py"

# 6. Apptainer Container Configuration
export CONTAINER_DIR="${PROJECT_DIR}/apptainer"
export APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.08_cuda13.0_arm_custom.sif"

# 7. HuggingFace caches under project share
export HF_HOME="${PROJECT_DIR}/hf_cache"
mkdir -p "${HF_HOME}"
# =====================================================================

echo "--- My Main Directory (host): ${PROJECT_DIR}"
echo "--- Bind-mounted inside container as: /workspace"
echo
echo "=== Running inside Apptainer ==="
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "EVAL_FILE:   ${PYTHON_FILE_EVAL}"
echo "MERGE_FILE:  ${PYTHON_FILE_MERGE}"
echo

# =====================================================================
# STEP 1: Check & Merge Sequence
# =====================================================================
echo "--- Checking for existing merged weights ---"

if [ -f "${MERGED_MODEL_DIR}/config.json" ]; then
    echo "--Found valid merged model at: ${MERGED_MODEL_DIR}"
    echo ">> Skipping merge step to save cluster compute time."
else
    echo "--No merged model found at destination. Initiating merge..."
    echo "--- Launching the merge application inside Apptainer ---"

    time srun apptainer exec --nv \
          -B "${PROJECT_DIR}:/workspace" \
          -B "${CLUSTER_PROJECT_ROOT}" \
          "${APPTAINER_SIF}" \
          python "${PYTHON_FILE_MERGE}" \
                --base_model_path "${BASE_MODEL_DIR}" \
                --lora_adapter_path "${LORA_ADAPTER_DIR}" \
                --output_path "${MERGED_MODEL_DIR}" \
                --max_shard_size "${MAX_SHARD_SIZE}"

    echo "--Weight merge processing complete :)"
fi

echo

# =====================================================================
# STEP 2: Evaluation Sequence
# =====================================================================
echo "--- Launching Evaluation Harness inside Apptainer ---"

time srun apptainer exec --nv \
      -B "${PROJECT_DIR}:/workspace" \
      -B "${CLUSTER_PROJECT_ROOT}" \
      "${APPTAINER_SIF}" \
      python "${PYTHON_FILE_EVAL}" \
      --config "${CONFIG_FILE}" \
      tasks=["mmlu_philosophy","mmlu_sociology","mmlu_business_ethics","mmlu_college_physics"] \
      batch_size=8 \
      save_results=True

echo
echo "--- Finished :) ---"
# tasks=["mmlu"] 
# tasks='["hellaswag", "truthfulqa_mc2", "gsm8k"]'
