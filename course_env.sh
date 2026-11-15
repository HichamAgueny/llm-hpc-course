#!/bin/bash
# ==============================================================================
# course_env.sh — THE single configuration file of the course
# LLM Fine-Tuning & Inference on HPC (Olivia Supercomputer)
# ==============================================================================
#
#  INSTRUCTORS: this is the only file you need to edit (sections 1-8).
#
#  PARTICIPANTS: you do not need to edit anything.
#    - ./run.sh and ./menu.sh load this file for you.
#    - Every Slurm job script loads this file by itself.
#    - Before a manual "sbatch", load it once per login so that the Slurm
#      account and reservation are applied:
#          source course_env.sh
#
#  Every job writes its results under  <course folder>/results/
#
#  Override knobs (export before running, e.g. "LORA_PATH=... ./run.sh day1 eval"):
#    COURSE_SLURM_ACCOUNT, COURSE_SLURM_RESERVATION  (empty = no reservation)
#    LORA_PATH         which LoRA adapter eval / serving / inference should use
#    DEFAULT_BACKEND   "container" or "eessi"
# ==============================================================================

# --- 0. Course folder (auto-detected from the location of this file) ---------
COURSE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COURSE_ROOT
export PROJECT_DIR="${COURSE_ROOT}"          # older name, kept for compatibility

# --- 1. Slurm account & reservation ------------------------------------------
# sbatch reads SBATCH_ACCOUNT / SBATCH_RESERVATION automatically, so these
# values are NOT repeated in the #SBATCH headers of the job scripts.
COURSE_SLURM_ACCOUNT="${COURSE_SLURM_ACCOUNT:-nn9997k}"
COURSE_SLURM_RESERVATION="${COURSE_SLURM_RESERVATION-software}"

export SBATCH_ACCOUNT="${COURSE_SLURM_ACCOUNT}"
if [[ -n "${COURSE_SLURM_RESERVATION}" ]]; then
    export SBATCH_RESERVATION="${COURSE_SLURM_RESERVATION}"
else
    unset SBATCH_RESERVATION
fi

# --- 2. Cluster & shared course assets ---------------------------------------
export CLUSTER_PROJECT_ROOT="/cluster/work/projects/${COURSE_SLURM_ACCOUNT}"

# Participants clone the course into:  /cluster/work/projects/nn9971k/$USER/llm-hpc-course
# (any folder works: COURSE_ROOT above is detected automatically)

# Read-only models, datasets and container images prepared by the instructors.
# Expected layout:  ${COURSE_ASSETS_DIR}/shared/{models,datasets}
#                   ${COURSE_ASSETS_DIR}/apptainer/*.sif
export COURSE_ASSETS_DIR="/cluster/projects/nn9971k/llm_course"

# setup_env.sh copies (or with --link, links) ${COURSE_ASSETS_DIR}/shared to
# ${COURSE_ROOT}/shared. If neither exists, the shared folder is used directly.
if [[ -e "${COURSE_ROOT}/shared" ]]; then
    export SHARED_DIR="${COURSE_ROOT}/shared"
else
    export SHARED_DIR="${COURSE_ASSETS_DIR}/shared"
fi

# --- 3. Base model & dataset (inputs) ----------------------------------------
export BASE_MODEL_NAME="Llama-3.2-1B-Instruct"
export BASE_MODEL_DIR="${SHARED_DIR}/models/${BASE_MODEL_NAME}"
export TOKENIZER_DIR="${BASE_MODEL_DIR}/original/tokenizer.model"
export QA_DATASET_DIR="${SHARED_DIR}/datasets/GPT-4-LLM/alpaca_gpt4_data.json"

# --- 4. Results (outputs, one folder per participant) ------------------------
export RESULTS_ROOT="${COURSE_ROOT}/results"
export CHECKPOINT_RESULTS_DIR="${RESULTS_ROOT}/checkpoint_results"
export PROFILING_RESULTS_DIR="${RESULTS_ROOT}/profiling_results"
export EVAL_RESULTS_DIR="${RESULTS_ROOT}/evaluation_results"
export QUANTIZED_MODELS_DIR="${RESULTS_ROOT}/quantized_models"
export BENCHMARK_RESULTS_DIR="${RESULTS_ROOT}/benchmark_logs"

# --- 5. LoRA adapter handed from fine-tuning to eval / serving / inference ---
# Default: the Day 1 single-GPU LoRA adapter (container run, or the EESSI run
# if only that one exists). Choose another one with LORA_PATH=/path/to/epoch_N
_course_adapter="${CHECKPOINT_RESULTS_DIR}/llama3.2-1b_lora_single_device/checkpoints/epoch_0"
_course_adapter_eessi="${CHECKPOINT_RESULTS_DIR}/eessi_llama3.2-1b_lora_single_device/checkpoints/epoch_0"
if [[ ! -d "${_course_adapter}" && -d "${_course_adapter_eessi}" ]]; then
    _course_adapter="${_course_adapter_eessi}"
fi
export LORA_ADAPTER_DIR="${LORA_PATH:-${_course_adapter}}"
# Base model + adapter merged into one Hugging Face checkpoint (one per adapter)
export MERGED_MODEL_DIR="${LORA_ADAPTER_DIR%/}_merged"
unset _course_adapter _course_adapter_eessi

# --- 6. Apptainer containers --------------------------------------------------
_course_sif() {   # prefer a local copy/link in ./apptainer, else the shared one
    if [[ -e "${COURSE_ROOT}/apptainer/$1" ]]; then
        echo "${COURSE_ROOT}/apptainer/$1"
    else
        echo "${COURSE_ASSETS_DIR}/apptainer/$1"
    fi
}
export PYTORCH_SIF="$(_course_sif pytorch_25.08_cuda13.0_arm_custom.sif)"
export VLLM_SIF="$(_course_sif vllm0.12_cu131_py3.12_arm_custom.sif)"
unset -f _course_sif

# Folders mounted into every container. Each is mounted at its own path so the
# host paths used in the scripts also work inside the container; the course
# folder is additionally available as /workspace.
CONTAINER_BINDS="-B ${COURSE_ROOT}:/workspace"
_course_bound=()
for _dir in "${CLUSTER_PROJECT_ROOT}" "${COURSE_ASSETS_DIR}" "${COURSE_ROOT}"; do
    [[ -d "${_dir}" ]] || continue
    _covered=false
    for _b in "${_course_bound[@]}"; do
        if [[ "${_dir}/" == "${_b}/"* ]]; then _covered=true; fi
    done
    if [[ "${_covered}" == false ]]; then
        _course_bound+=("${_dir}")
        CONTAINER_BINDS+=" -B ${_dir}"
    fi
done
export CONTAINER_BINDS
unset _course_bound _dir _b _covered

# --- 7. EESSI software modules ------------------------------------------------
export EESSI_MODULE="EESSI/2025.06"
export TORCHTUNE_MODULE="torchtune/0.6.1-foss-2025b-PyTorch-2.12.0-CUDA-12.9.1"
export VLLM_MODULE="vllm/0.22.1-foss-2025b-PyTorch-2.12.0-CUDA-12.9.1"
export LM_EVAL_MODULE="lm-eval/0.4.12-foss-2025b"

# --- 8. Multi-node networking (Slingshot) for container jobs -----------------
export NRIS_GPU_MODULE="NRIS/GPU"
export NCCL_MODULE="NCCL/2.30.4-GCCcore-14.3.0-CUDA-13.0.0"
export HOST_NCCL_PATH="/cluster/software/NRIS/neoverse_v2/software/NCCL/2.30.4-GCCcore-14.3.0-CUDA-13.0.0"
export HOST_AWS_OFI_NCCL_PATH="/cluster/software/NRIS/neoverse_v2/software/aws-ofi-nccl/1.19.1-GCCcore-14.3.0-CUDA-13.0.0/lib"
export HOST_LIBFABRIC_LIB_PATH="/opt/cray/libfabric/2.3.1/lib64"
export HOST_LIBFABRIC_INCLUDE_PATH="/opt/cray/libfabric/2.3.1/include"
export HOST_CXI_LIB_PATH="/usr/lib64"

# HTTP proxy for compute-node jobs that download data (e.g. GPTQ calibration)
export COURSE_HTTP_PROXY="http://10.63.2.48:3128/"

# --- 9. Caches -----------------------------------------------------------------
export HF_HOME="${COURSE_ROOT}/hf_cache"
export VLLM_CACHE_ROOT="${COURSE_ROOT}/.cache/vllm"

# --- 10. Defaults & helpers ----------------------------------------------------
export DEFAULT_BACKEND="${DEFAULT_BACKEND:-container}"

# Make "gpu.monitor.sh <JobID>" and "run_nvitop.sh <JobID>" work from any folder
case ":${PATH}:" in
    *":${COURSE_ROOT}/utils:"*) ;;
    *) export PATH="${COURSE_ROOT}/utils:${PATH}" ;;
esac

# Used inside job scripts: stop early with a clear message if an input is missing.
#   course_require <path> "<what it is>" ["<how to fix it>"]
course_require() {
    if [[ ! -e "$1" ]]; then
        echo "ERROR: $2 not found: $1" >&2
        if [[ -n "${3:-}" ]]; then echo "       -> $3" >&2; fi
        exit 1
    fi
}

# Path of a quantized model, e.g. "course_quantized_model bnb_8bit": your own copy
# from the quantization lab (results/quantized_models/) if it was completed, else the shared one.
course_quantized_model() {
    local name="${BASE_MODEL_NAME}-$1"
    if [[ -f "${QUANTIZED_MODELS_DIR}/${name}/config.json" ]]; then
        echo "${QUANTIZED_MODELS_DIR}/${name}"
    else
        echo "${SHARED_DIR}/models/${name}"
    fi
}

true
