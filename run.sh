#!/bin/bash
# ==============================================================================
# Unified Job Runner & Dispatcher
# LLM Fine-Tuning & Inference on HPC
# ==============================================================================

set -e

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_ROOT}/course_env.sh" ]; then
    source "${SCRIPT_ROOT}/course_env.sh"
fi

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

print_usage() {
    echo -e "${BOLD}Usage:${NC} ./run.sh <category> <action> [options]"
    echo ""
    echo -e "${BOLD}Day 1 Commands (Single GPU):${NC}"
    echo -e "  ./run.sh day1 lora       [--env=container|eessi] [--with-eval]  Single-GPU LoRA fine-tuning"
    echo -e "  ./run.sh day1 qlora      [--env=container|eessi]               Single-GPU 4-bit QLoRA fine-tuning"
    echo -e "  ./run.sh day1 eval       [--env=container|eessi] [--base]      Evaluate LoRA adapter (or base model)"
    echo -e "  ./run.sh day1 profile    [--method=lora|qlora]                 Profile GPU memory & kernel execution"
    echo ""
    echo -e "${BOLD}Day 2 Commands (Multi-GPU & Distributed):${NC}"
    echo -e "  ./run.sh day2 ft-single  [--method=lora|qlora|qat-lora] [--env=container|eessi]  Single-node 4-GPU FT"
    echo -e "  ./run.sh day2 ft-multi   [--method=lora|qlora|qat-lora] [--env=container|eessi]  Multi-node distributed FT"
    echo -e "  ./run.sh day2 eval       [--env=container|eessi] [--base]                         Distributed MMLU evaluation"
    echo -e "  ./run.sh day2 profile    [--method=lora|qlora|qat-lora]                          Multi-GPU distributed profiling"
    echo -e "  ./run.sh day2 quantize   <bnb4|bnb8|torchao|gptq>                                Post-training quantization"
    echo -e "  ./run.sh day2 serve      [--type=single|multi] [--env=container|eessi]            Launch vLLM server"
    echo -e "  ./run.sh day2 chat       [--type=single|multi]                                    Open terminal chat client"
    echo -e "  ./run.sh day2 inference  [--env=container|eessi]                                  Distributed batch inference"
    echo ""
    echo -e "${BOLD}Cluster Utilities:${NC}"
    echo -e "  ./run.sh monitor <JobID>   Attach real-time GPU monitor to a running job"
    echo -e "  ./run.sh status            Show your active jobs (squeue --me)"
    echo -e "  ./run.sh check             Run pre-flight environment & directory check"
    echo ""
}

# Helper to run sbatch safely from the script's directory
submit_job() {
    local target_dir="$1"
    local script_name="$2"
    local extra_flags="$3"
    local with_eval="$4"
    local eval_dir="$5"
    local eval_script="$6"

    mkdir -p "${target_dir}/out"

    # Build sbatch flags
    local sbatch_args=()
    if [ -n "${COURSE_SLURM_ACCOUNT:-}" ]; then
        sbatch_args+=("--account=${COURSE_SLURM_ACCOUNT}")
    fi
    if [ -n "${COURSE_SLURM_RESERVATION:-}" ]; then
        sbatch_args+=("--reservation=${COURSE_SLURM_RESERVATION}")
    fi
    if [ -n "${extra_flags}" ]; then
        sbatch_args+=("${extra_flags}")
    fi

    echo -e "${BLUE}${BOLD}Submitting:${NC} ${script_name}"
    echo -e "${BLUE}Directory:${NC}  ${target_dir}"

    cd "${target_dir}"
    local output
    output=$(sbatch "${sbatch_args[@]}" "${script_name}")
    echo -e "${GREEN}${BOLD}✓ ${output}${NC}"

    local job_id
    job_id=$(echo "${output}" | awk '{print $NF}')

    echo ""
    echo -e "Monitor job:  ${BOLD}./run.sh monitor ${job_id}${NC}"
    echo -e "Check status: ${BOLD}./run.sh status${NC}"

    # Auto-eval chaining
    if [ "${with_eval}" == "true" ] && [ -n "${job_id}" ] && [ -n "${eval_script}" ]; then
        echo ""
        echo -e "${YELLOW}${BOLD}Chaining Evaluation Job (will run automatically when Job ${job_id} succeeds)...${NC}"
        mkdir -p "${eval_dir}/out"
        cd "${eval_dir}"
        local eval_output
        eval_output=$(sbatch --dependency=afterok:"${job_id}" "${sbatch_args[@]}" "${eval_script}")
        echo -e "${GREEN}${BOLD}✓ Scheduled Evaluation: ${eval_output}${NC}"
    fi

    cd "${SCRIPT_ROOT}"
}

# Parse options
ENV_CHOICE="${DEFAULT_BACKEND:-container}"
METHOD_CHOICE="lora"
TYPE_CHOICE="multi"
WITH_EVAL=false
EVAL_BASE=false

# Read subcommands
CATEGORY="${1:-}"
ACTION="${2:-}"
shift 2 2>/dev/null || true

for arg in "$@"; do
    case "$arg" in
        --env=*)
            ENV_CHOICE="${arg#*=}"
            ;;
        --method=*)
            METHOD_CHOICE="${arg#*=}"
            ;;
        --type=*)
            TYPE_CHOICE="${arg#*=}"
            ;;
        --with-eval)
            WITH_EVAL=true
            ;;
        --base)
            EVAL_BASE=true
            ;;
        *)
            ;;
    esac
done

case "${CATEGORY}" in
    day1)
        case "${ACTION}" in
            lora)
                if [ "${ENV_CHOICE}" == "eessi" ]; then
                    submit_job \
                        "${SCRIPT_ROOT}/day1_single_gpu/finetuning/lora_slurm/job_eessi" \
                        "job_singleGPU_QA_LoRA_eessi.sh" \
                        "" \
                        "${WITH_EVAL}" \
                        "${SCRIPT_ROOT}/day1_single_gpu/evaluation" \
                        "job_singleGPU_LoRA_eessi.sh"
                else
                    submit_job \
                        "${SCRIPT_ROOT}/day1_single_gpu/finetuning/lora_slurm/job_container" \
                        "job_singleGPU_QA_LoRA.sh" \
                        "" \
                        "${WITH_EVAL}" \
                        "${SCRIPT_ROOT}/day1_single_gpu/evaluation" \
                        "job_singleGPU_LoRA.sh"
                fi
                ;;
            qlora)
                if [ "${ENV_CHOICE}" == "eessi" ]; then
                    submit_job \
                        "${SCRIPT_ROOT}/day1_single_gpu/finetuning/qlora_slurm/job_eessi" \
                        "job_singleGPU_QA_QLoRA_eessi.sh"
                else
                    submit_job \
                        "${SCRIPT_ROOT}/day1_single_gpu/finetuning/qlora_slurm/job_container" \
                        "job_singleGPU_QA_QLoRA.sh"
                fi
                ;;
            eval)
                local eval_script="job_singleGPU_LoRA.sh"
                if [ "${EVAL_BASE}" == "true" ]; then
                    eval_script="job_singleGPU_No_LoRA.sh"
                fi
                if [ "${ENV_CHOICE}" == "eessi" ]; then
                    if [ "${EVAL_BASE}" == "true" ]; then
                        eval_script="job_singleGPU_No_LoRA_eessi.sh"
                    else
                        eval_script="job_singleGPU_LoRA_eessi.sh"
                    fi
                fi
                submit_job "${SCRIPT_ROOT}/day1_single_gpu/evaluation" "${eval_script}"
                ;;
            profile)
                local prof_script="job_singleGPU_QA_LoRA_profiling.sh"
                if [ "${METHOD_CHOICE}" == "qlora" ]; then
                    prof_script="job_singleGPU_QA_QLoRA_profiling.sh"
                fi
                submit_job "${SCRIPT_ROOT}/day1_single_gpu/profiling" "${prof_script}"
                ;;
            *)
                echo -e "${YELLOW}Unknown Day 1 action: ${ACTION}${NC}"
                print_usage
                exit 1
                ;;
        esac
        ;;

    day2)
        case "${ACTION}" in
            ft-single|ft-single-node)
                case "${METHOD_CHOICE}" in
                    lora)
                        if [ "${ENV_CHOICE}" == "eessi" ]; then
                            submit_job "${SCRIPT_ROOT}/day2_multi_gpu/fine-tuning/ft_single_node/lora_slurm/job_eessi" "job_multiGPU_QA_LoRA_eessi.sh"
                        else
                            submit_job "${SCRIPT_ROOT}/day2_multi_gpu/fine-tuning/ft_single_node/lora_slurm/job_container" "job_multiGPU_QA_LoRA.sh"
                        fi
                        ;;
                    qlora)
                        submit_job "${SCRIPT_ROOT}/day2_multi_gpu/fine-tuning/ft_single_node/qlora_slurm" "job_multiGPU_QA_QLoRA.sh"
                        ;;
                    qat|qat-lora)
                        submit_job "${SCRIPT_ROOT}/day2_multi_gpu/fine-tuning/ft_single_node/qat-lora_slurm" "job_multiGPU_QA_qat-LoRA.sh"
                        ;;
                esac
                ;;
            ft-multi|ft-multi-node)
                case "${METHOD_CHOICE}" in
                    lora)
                        if [ "${ENV_CHOICE}" == "eessi" ]; then
                            submit_job "${SCRIPT_ROOT}/day2_multi_gpu/fine-tuning/ft_multi_node/lora_slurm/job_eessi" "job_multiNode_QA_LoRA_eessi.sh"
                        else
                            submit_job "${SCRIPT_ROOT}/day2_multi_gpu/fine-tuning/ft_multi_node/lora_slurm/job_container" "job_multiNode_QA_LoRA.sh"
                        fi
                        ;;
                    qlora)
                        submit_job "${SCRIPT_ROOT}/day2_multi_gpu/fine-tuning/ft_multi_node/qlora_slurm" "job_multiNode_QA_QLoRA.sh"
                        ;;
                    qat|qat-lora)
                        submit_job "${SCRIPT_ROOT}/day2_multi_gpu/fine-tuning/ft_multi_node/qat-lora_slurm" "job_multiNode_QA_qat-LoRA.sh"
                        ;;
                esac
                ;;
            eval)
                local eval_script="job_multiGPU_LoRA.sh"
                if [ "${EVAL_BASE}" == "true" ]; then
                    eval_script="job_multiGPU_No_LoRA.sh"
                fi
                if [ "${ENV_CHOICE}" == "eessi" ]; then
                    if [ "${EVAL_BASE}" == "true" ]; then
                        eval_script="job_multiGPU_No_LoRA_eessi.sh"
                    else
                        eval_script="job_multiGPU_LoRA_eessi.sh"
                    fi
                fi
                submit_job "${SCRIPT_ROOT}/day2_multi_gpu/evaluation" "${eval_script}"
                ;;
            profile)
                local prof_script="job_multiGPU_QA_LoRA.sh"
                if [ "${METHOD_CHOICE}" == "qlora" ]; then
                    prof_script="job_multiGPU_QA_QLoRA.sh"
                elif [ "${METHOD_CHOICE}" == "qat" ] || [ "${METHOD_CHOICE}" == "qat-lora" ]; then
                    prof_script="job_multiGPU_QA_qat-LoRA.sh"
                fi
                submit_job "${SCRIPT_ROOT}/day2_multi_gpu/profiling" "${prof_script}"
                ;;
            quantize)
                local quant_type="${1:-bnb4}"
                case "${quant_type}" in
                    bnb4)
                        submit_job "${SCRIPT_ROOT}/day2_multi_gpu/quantization/quat_bitsandbytes" "job_bnb_4bit.sh"
                        ;;
                    bnb8)
                        submit_job "${SCRIPT_ROOT}/day2_multi_gpu/quantization/quat_bitsandbytes" "job_bnb_8bit.sh"
                        ;;
                    torchao)
                        submit_job "${SCRIPT_ROOT}/day2_multi_gpu/quantization/quat_torchao" "job_torchao.sh"
                        ;;
                    gptq)
                        submit_job "${SCRIPT_ROOT}/day2_multi_gpu/quantization/quat_gptqmodel" "job_gptq.sh"
                        ;;
                    *)
                        echo -e "${YELLOW}Unknown quantization type: ${quant_type}. Choose bnb4, bnb8, torchao, or gptq.${NC}"
                        exit 1
                        ;;
                esac
                ;;
            serve)
                local serve_dir="${SCRIPT_ROOT}/day2_multi_gpu/serving/${TYPE_CHOICE}_device/vllm_${ENV_CHOICE}"
                submit_job "${serve_dir}" "serve_vllm.sh"
                ;;
            chat)
                local chat_dir="${SCRIPT_ROOT}/day2_multi_gpu/serving/${TYPE_CHOICE}_device/vllm_${ENV_CHOICE}"
                cd "${chat_dir}"
                bash chat.sh
                ;;
            inference)
                local inf_script="slurm_vllm_distributed_container.sh"
                if [ "${ENV_CHOICE}" == "eessi" ]; then
                    inf_script="slurm_vllm_distributed_eessi.sh"
                fi
                submit_job "${SCRIPT_ROOT}/day2_multi_gpu/inference" "${inf_script}"
                ;;
            *)
                echo -e "${YELLOW}Unknown Day 2 action: ${ACTION}${NC}"
                print_usage
                exit 1
                ;;
        esac
        ;;

    monitor)
        local job_id="${ACTION}"
        if [ -z "${job_id}" ]; then
            echo -e "${RED}Error: Please provide a Slurm JobID: ./run.sh monitor <JobID>${NC}"
            exit 1
        fi
        exec "${SCRIPT_ROOT}/utils/gpu.monitor.sh" "${job_id}"
        ;;

    status)
        exec squeue --me
        ;;

    check)
        exec "${SCRIPT_ROOT}/check_env.sh"
        ;;

    help|--help|-h|"")
        print_usage
        ;;

    *)
        echo -e "${RED}Error: Unknown command '${CATEGORY}'${NC}"
        print_usage
        exit 1
        ;;
esac
