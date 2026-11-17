#!/bin/bash -e
#SBATCH --job-name=mmlu_eval-llama3.2-1B-base-1gpu_epoch0_eessi
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
#SBATCH --reservation=LLM_day1

echo "--Node: $(hostname)"
echo

export PYTHONNOUSERSITE=1

ml EESSI/2025.06
module load torchtune/0.6.1-foss-2025b-PyTorch-2.12.0-CUDA-12.9.1
module load Transformers
module load accelerate
module load lm-eval/0.4.12-foss-2025b

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
export TOKENIZER_DIR="${BASE_MODEL_DIR}/original/tokenizer.model"
# Point the YAML's ${checkpoint_dir} at the BASE model — no LoRA involved
export MERGED_MODEL_DIR="${BASE_MODEL_DIR}"

# >>> BASE MODEL EVAL: no LoRA, no merge <<<
export EVAL_CHECKPOINT_DIR="${BASE_MODEL_DIR}"

# 4. Output Component Paths
export OUTPUT_DIR="${RESULT_DIR}/evaluation_results/llama3.2-1b-base_mmlu_singlegpu_eessi"
mkdir -p "$OUTPUT_DIR"

# 5. Script / Config Layout
export CONFIG_FILE="${PROJECT_DIR}/configs/eval_cfg/eval_config.yaml"
export PYTHON_FILE_EVAL="${PROJECT_DIR}/recipes/eval/eleuther_eval.py"

# 6. HuggingFace caches under project share
export HF_HOME="${PROJECT_DIR}/hf_cache"
mkdir -p "${HF_HOME}"
# =====================================================================

echo "--- My Main Directory (host): ${PROJECT_DIR}"
echo
echo "=== Running inside Apptainer ==="
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "EVAL_FILE:   ${PYTHON_FILE_EVAL}"
echo

# =====================================================================
# Evaluation Sequence
# =====================================================================
echo "--- Launching Evaluation Harness ---"

time srun python "${PYTHON_FILE_EVAL}" \
      --config "${CONFIG_FILE}" \
      tasks=["mmlu_philosophy","mmlu_sociology","mmlu_business_ethics","mmlu_college_physics"] \
      batch_size=8 \
      save_results=True

echo
echo "--- Finished :) ---"
# tasks=["mmlu"] 
# tasks='["hellaswag", "truthfulqa_mc2", "gsm8k"]'
