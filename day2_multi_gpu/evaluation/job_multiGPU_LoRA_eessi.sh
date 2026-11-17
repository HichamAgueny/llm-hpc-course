#!/bin/bash -e
#SBATCH --job-name=eval-llama3.2-1B-lora-4gpu_mmlu_eessi
#SBATCH --account=nn9971k
#SBATCH --time=00:30:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-gpu=96G
#SBATCH --reservation=LLM_day2
#SBATCH -o ./out/%x-%j.out

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

export PYTHONNOUSERSITE=1

ml EESSI/2025.06
module load torchtune/0.6.1-foss-2025b-PyTorch-2.12.0-CUDA-12.9.1
module load Transformers
module load accelerate
module load lm-eval/0.4.12-foss-2025b
module load peft

export NUMEXPR_MAX_THREADS=72

# 1. Base Project Allocation (Change this if migrating to a different project share)
export CLUSTER_PROJECT_ROOT="/cluster/work/projects/nn9971k"

# 2. Personal Workspace Setup (Derived automatically from your root above)
export PROJECT_DIR="${CLUSTER_PROJECT_ROOT}/${USER}/llm-hpc-course"
export SHARED_DIR="${PROJECT_DIR}/shared"
export RESULT_DIR="${PROJECT_DIR}/results"

# 3. Model & Adapter Component Paths
export BASE_MODEL_DIR="${SHARED_DIR}/models/Llama-3.2-1B-Instruct"
# Adapter produced by the EESSI fine-tuning job (job_singleGPU_QA_LoRA_eessi.sh)
export LORA_ADAPTER_DIR="${RESULT_DIR}/checkpoint_results/eessi_llama3.2-1b_lora_single_device/checkpoints/epoch_1"
export MERGED_MODEL_DIR="${RESULT_DIR}/checkpoint_results/Llama-3.2-1B-Instruct_merged_eessi"
export TOKENIZER_DIR="${BASE_MODEL_DIR}/original/tokenizer.model"
export MAX_SHARD_SIZE="5GB"

# 4. Output Component Paths
export OUTPUT_DIR="${RESULT_DIR}/evaluation_results/llama3.2-1b-mmlu_eessi"
mkdir -p "$OUTPUT_DIR"

# 5. Script Layout
export PYTHON_FILE_MERGE="${PROJECT_DIR}/recipes/merge/merge_and_save.py"

# 6. HuggingFace caches (datasets + hub) under the project share
export HF_HOME="${PROJECT_DIR}/hf_cache"
mkdir -p "${HF_HOME}"

# ---------------------------------------------------------------------
# GPU information
# ---------------------------------------------------------------------

echo "======================================================"
echo "Starting 4-GPU evaluation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Allocated GPUs: ${SLURM_GPUS}"
echo "Merged model: ${MERGED_MODEL_DIR}"
echo "======================================================"

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

    time srun python "${PYTHON_FILE_MERGE}" \
                --base_model_path "${BASE_MODEL_DIR}" \
                --lora_adapter_path "${LORA_ADAPTER_DIR}" \
                --output_path "${MERGED_MODEL_DIR}" \
                --max_shard_size "${MAX_SHARD_SIZE}"

    echo "--Weight merge processing complete :)"
fi

echo

# ---------------------------------------------------------------------
# STEP 2: Run lm-evaluation-harness with Accelerate
# ---------------------------------------------------------------------
# CPU affinity
CPU_BIND="map_cpu:1,73,145,217"

time srun --cpu-bind=${CPU_BIND} \
    --ntasks=1 \
    --gpus=4 \
    --cpus-per-task=72 \
        bash -c '
            echo "Starting Accelerate evaluation..."
            echo "Output directory: '"${OUTPUT_DIR}"'"
            accelerate launch \
                --multi_gpu \
                --num_processes 4 \
                -m lm_eval \
                --model hf \
                --model_args "pretrained='"${MERGED_MODEL_DIR}"',dtype=bfloat16" \
                --tasks mmlu_philosophy,mmlu_sociology,mmlu_business_ethics,mmlu_college_physics \
                --apply_chat_template \
                --batch_size 8 \
                --output_path "'"${OUTPUT_DIR}"'/results.json"
        '

echo "======================================================"
echo "Evaluation completed."
echo "======================================================"

