#!/bin/bash
# ==============================================================================
# Unified Job Runner & Dispatcher
# LLM Fine-Tuning & Inference on HPC
#
# Submits any lab from the course folder - no need to cd into lab folders.
# Account, reservation and paths come from course_env.sh.
# ==============================================================================

set -e

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_ROOT}/course_env.sh"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

D1="${SCRIPT_ROOT}/day1_single_gpu"
D2="${SCRIPT_ROOT}/day2_multi_gpu"

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
    echo -e "  ./run.sh day2 eval       [--env=container|eessi] [--base]                         Multi-GPU MMLU evaluation"
    echo -e "  ./run.sh day2 profile    [--method=lora|qlora|qat-lora]                          Multi-GPU distributed profiling"
    echo -e "  ./run.sh day2 quantize   <bnb4|bnb8|torchao|gptq>                                Post-training quantization"
    echo -e "  ./run.sh day2 serve      [--type=single|multi] [--env=container|eessi]"
    echo -e "                           [--quant=none|bnb|gptq|torchao]                          Launch vLLM server"
    echo -e "  ./run.sh day2 chat       [--type=single|multi] [--env=container|eessi]            Open terminal chat client"
    echo -e "  ./run.sh day2 inference  [--env=container|eessi]                                  Distributed batch inference"
    echo ""
    echo -e "${BOLD}Cluster Utilities:${NC}"
    echo -e "  ./run.sh monitor <JobID>   Attach real-time GPU monitor to a running job"
    echo -e "  ./run.sh status            Show your active jobs (squeue --me)"
    echo -e "  ./run.sh check             Run pre-flight environment & directory check"
    echo ""
    echo -e "${BOLD}Useful overrides:${NC}"
    echo -e "  LORA_PATH=/path/to/checkpoints/epoch_N ./run.sh day1 eval   Evaluate / serve another adapter"
    echo -e "  PORT=8001 ./run.sh day2 serve                               Use another port for the server"
    echo ""
}

die() {
    echo -e "${RED}Error: $*${NC}" >&2
    exit 1
}

# Submit a job script from its own folder (so its ./out/ log folder is used)
submit_job() {
    local target_dir="$1"
    local script_name="$2"
    local with_eval="${3:-false}"
    local eval_dir="${4:-}"
    local eval_script="${5:-}"

    [ -f "${target_dir}/${script_name}" ] || die "Job script not found: ${target_dir}/${script_name}"
    mkdir -p "${target_dir}/out"

    local sbatch_args=("--account=${COURSE_SLURM_ACCOUNT}")
    if [ -n "${COURSE_SLURM_RESERVATION}" ]; then
        sbatch_args+=("--reservation=${COURSE_SLURM_RESERVATION}")
    fi

    echo -e "${BLUE}${BOLD}Submitting:${NC} ${script_name}"
    echo -e "${BLUE}Directory:${NC}  ${target_dir#${SCRIPT_ROOT}/}"
    echo -e "${BLUE}Command:${NC}    sbatch ${sbatch_args[*]} ${script_name}"

    local output job_id
    output=$(cd "${target_dir}" && sbatch "${sbatch_args[@]}" "${script_name}")
    echo -e "${GREEN}${BOLD}✓ ${output}${NC}"
    job_id=$(echo "${output}" | awk '{print $NF}')

    echo ""
    echo -e "Job log:      ${BOLD}${target_dir#${SCRIPT_ROOT}/}/out/${NC}"
    echo -e "Monitor job:  ${BOLD}./run.sh monitor ${job_id}${NC}"
    echo -e "Check status: ${BOLD}./run.sh status${NC}"

    # Auto-eval chaining
    if [ "${with_eval}" == "true" ] && [ -n "${job_id}" ] && [ -n "${eval_script}" ]; then
        echo ""
        echo -e "${YELLOW}${BOLD}Chaining Evaluation Job (will run automatically when Job ${job_id} succeeds)...${NC}"
        mkdir -p "${eval_dir}/out"
        local eval_output
        eval_output=$(cd "${eval_dir}" && sbatch --dependency=afterok:"${job_id}" "${sbatch_args[@]}" "${eval_script}")
        echo -e "${GREEN}${BOLD}✓ Scheduled Evaluation: ${eval_output}${NC}"
    fi
}

# --- Parse arguments -----------------------------------------------------------
ENV_CHOICE="${DEFAULT_BACKEND:-container}"
METHOD_CHOICE="lora"
TYPE_CHOICE="multi"
QUANT_CHOICE="none"
WITH_EVAL=false
EVAL_BASE=false
POSITIONAL=()

CATEGORY="${1:-}"
ACTION="${2:-}"
shift 2 2>/dev/null || shift $# 2>/dev/null || true

for arg in "$@"; do
    case "$arg" in
        --env=*)    ENV_CHOICE="${arg#*=}" ;;
        --method=*) METHOD_CHOICE="${arg#*=}" ;;
        --type=*)   TYPE_CHOICE="${arg#*=}" ;;
        --quant=*)  QUANT_CHOICE="${arg#*=}" ;;
        --with-eval) WITH_EVAL=true ;;
        --base)     EVAL_BASE=true ;;
        --*)        die "Unknown option '${arg}' (see ./run.sh help)" ;;
        *)          POSITIONAL+=("$arg") ;;
    esac
done

case "${ENV_CHOICE}" in container|eessi) ;; *) die "--env must be 'container' or 'eessi'" ;; esac
case "${TYPE_CHOICE}" in single|multi) ;; *) die "--type must be 'single' or 'multi'" ;; esac
[ "${METHOD_CHOICE}" == "qat" ] && METHOD_CHOICE="qat-lora"

# Only LoRA has an EESSI variant for the Day 2 fine-tuning labs
container_only_note() {
    if [ "${ENV_CHOICE}" == "eessi" ]; then
        echo -e "${YELLOW}Note: '${METHOD_CHOICE}' is only available as a container job; submitting the container version.${NC}"
    fi
}

case "${CATEGORY}" in
    day1)
        case "${ACTION}" in
            lora)
                if [ "${ENV_CHOICE}" == "eessi" ]; then
                    submit_job "${D1}/finetuning/lora_slurm/job_eessi" "job_singleGPU_QA_LoRA_eessi.sh" \
                        "${WITH_EVAL}" "${D1}/evaluation" "job_singleGPU_LoRA_eessi.sh"
                else
                    submit_job "${D1}/finetuning/lora_slurm/job_container" "job_singleGPU_QA_LoRA.sh" \
                        "${WITH_EVAL}" "${D1}/evaluation" "job_singleGPU_LoRA.sh"
                fi
                ;;
            qlora)
                if [ "${ENV_CHOICE}" == "eessi" ]; then
                    submit_job "${D1}/finetuning/qlora_slurm/job_eessi" "job_singleGPU_QA_QLoRA_eessi.sh"
                else
                    submit_job "${D1}/finetuning/qlora_slurm/job_container" "job_singleGPU_QA_QLoRA.sh"
                fi
                ;;
            eval)
                eval_script="job_singleGPU_LoRA"
                [ "${EVAL_BASE}" == "true" ] && eval_script="job_singleGPU_No_LoRA"
                [ "${ENV_CHOICE}" == "eessi" ] && eval_script="${eval_script}_eessi"
                submit_job "${D1}/evaluation" "${eval_script}.sh"
                ;;
            profile)
                case "${METHOD_CHOICE}" in
                    lora)  submit_job "${D1}/profiling" "job_singleGPU_QA_LoRA_profiling.sh" ;;
                    qlora) submit_job "${D1}/profiling" "job_singleGPU_QA_QLoRA_profiling.sh" ;;
                    *)     die "--method must be lora or qlora for day1 profile" ;;
                esac
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
                            submit_job "${D2}/fine-tuning/ft_single_node/lora_slurm/job_eessi" "job_multiGPU_QA_LoRA_eessi.sh"
                        else
                            submit_job "${D2}/fine-tuning/ft_single_node/lora_slurm/job_container" "job_multiGPU_QA_LoRA.sh"
                        fi
                        ;;
                    qlora)
                        container_only_note
                        submit_job "${D2}/fine-tuning/ft_single_node/qlora_slurm" "job_multiGPU_QA_QLoRA.sh"
                        ;;
                    qat-lora)
                        container_only_note
                        submit_job "${D2}/fine-tuning/ft_single_node/qat-lora_slurm" "job_multiGPU_QA_qat-LoRA.sh"
                        ;;
                    *) die "--method must be lora, qlora or qat-lora" ;;
                esac
                ;;
            ft-multi|ft-multi-node)
                case "${METHOD_CHOICE}" in
                    lora)
                        if [ "${ENV_CHOICE}" == "eessi" ]; then
                            submit_job "${D2}/fine-tuning/ft_multi_node/lora_slurm/job_eessi" "job_multiNode_QA_LoRA_eessi.sh"
                        else
                            submit_job "${D2}/fine-tuning/ft_multi_node/lora_slurm/job_container" "job_multiNode_QA_LoRA.sh"
                        fi
                        ;;
                    qlora)
                        container_only_note
                        submit_job "${D2}/fine-tuning/ft_multi_node/qlora_slurm" "job_multiNode_QA_QLoRA.sh"
                        ;;
                    qat-lora)
                        container_only_note
                        submit_job "${D2}/fine-tuning/ft_multi_node/qat-lora_slurm" "job_multiNode_QA_qat-LoRA.sh"
                        ;;
                    *) die "--method must be lora, qlora or qat-lora" ;;
                esac
                ;;
            eval)
                eval_script="job_multiGPU_LoRA"
                [ "${EVAL_BASE}" == "true" ] && eval_script="job_multiGPU_No_LoRA"
                [ "${ENV_CHOICE}" == "eessi" ] && eval_script="${eval_script}_eessi"
                submit_job "${D2}/evaluation" "${eval_script}.sh"
                ;;
            profile)
                case "${METHOD_CHOICE}" in
                    lora)     submit_job "${D2}/profiling" "job_multiGPU_QA_LoRA.sh" ;;
                    qlora)    submit_job "${D2}/profiling" "job_multiGPU_QA_QLoRA.sh" ;;
                    qat-lora) submit_job "${D2}/profiling" "job_multiGPU_QA_qat-LoRA.sh" ;;
                    *) die "--method must be lora, qlora or qat-lora" ;;
                esac
                ;;
            quantize)
                quant_type="${POSITIONAL[0]:-bnb4}"
                case "${quant_type}" in
                    bnb4)    submit_job "${D2}/quantization/quat_bitsandbytes" "job_bnb_4bit.sh" ;;
                    bnb8)    submit_job "${D2}/quantization/quat_bitsandbytes" "job_bnb_8bit.sh" ;;
                    torchao) submit_job "${D2}/quantization/quat_torchao" "job_torchao.sh" ;;
                    gptq)    submit_job "${D2}/quantization/quat_gptqmodel" "job_gptq.sh" ;;
                    *) die "Unknown quantization type: ${quant_type}. Choose bnb4, bnb8, torchao, or gptq." ;;
                esac
                ;;
            serve)
                serve_dir="${D2}/serving/${TYPE_CHOICE}_device/vllm_${ENV_CHOICE}"
                case "${QUANT_CHOICE}" in
                    none)             serve_script="serve_vllm" ;;
                    bnb|gptq|torchao) serve_script="serve_vllm_quantized_${QUANT_CHOICE}" ;;
                    *) die "--quant must be none, bnb, gptq or torchao" ;;
                esac
                [ "${ENV_CHOICE}" == "eessi" ] && serve_script="${serve_script}_eessi"
                submit_job "${serve_dir}" "${serve_script}.sh"
                echo ""
                echo -e "When the server is up, chat with it: ${BOLD}./run.sh day2 chat --type=${TYPE_CHOICE} --env=${ENV_CHOICE}${NC}"
                ;;
            chat)
                chat_dir="${D2}/serving/${TYPE_CHOICE}_device/vllm_${ENV_CHOICE}"
                exec bash "${chat_dir}/chat.sh"
                ;;
            inference)
                if [ "${ENV_CHOICE}" == "eessi" ]; then
                    submit_job "${D2}/inference" "slurm_vllm_distributed_eessi.sh"
                else
                    submit_job "${D2}/inference" "slurm_vllm_distributed_container.sh"
                fi
                ;;
            *)
                echo -e "${YELLOW}Unknown Day 2 action: ${ACTION}${NC}"
                print_usage
                exit 1
                ;;
        esac
        ;;

    monitor)
        job_id="${ACTION}"
        [ -n "${job_id}" ] || die "Please provide a Slurm JobID: ./run.sh monitor <JobID>"
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
