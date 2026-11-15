#!/bin/bash
# ==============================================================================
# Interactive Terminal Menu Launcher
# LLM Fine-Tuning & Inference on HPC
# ==============================================================================

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

clear_screen() {
    clear 2>/dev/null || echo -e "\n----------------------------------------\n"
}

# Prints the menu on stderr so that only the answer is captured by $(choose_env)
choose_env() {
    {
        echo ""
        echo -e "${BOLD}Select Execution Environment:${NC}"
        echo "  1) Apptainer Container (Recommended)"
        echo "  2) Native EESSI Modules"
    } >&2
    read -rp "Choice [1-2] (default 1): " env_choice
    if [ "$env_choice" == "2" ]; then
        echo "eessi"
    else
        echo "container"
    fi
}

while true; do
    echo -e "${BOLD}${BLUE}================================================================${NC}"
    echo -e "${BOLD}${BLUE}     LLM Fine-Tuning, Serving & Inference on HPC — Menu         ${NC}"
    echo -e "${BOLD}${BLUE}================================================================${NC}"
    echo -e "${BOLD}--- DAY 1 (Single GPU) ---${NC}"
    echo "  1) Single-GPU LoRA Fine-Tuning (QA)"
    echo "  2) Single-GPU LoRA Fine-Tuning with Auto-Evaluation"
    echo "  3) Single-GPU 4-bit QLoRA Fine-Tuning (QA)"
    echo "  4) Evaluate LoRA Model (MMLU Benchmark)"
    echo "  5) Evaluate Base Model (MMLU Benchmark Baseline)"
    echo "  6) PyTorch Memory & Kernel Profiling"
    echo ""
    echo -e "${BOLD}--- DAY 2 (Multi-GPU Distributed & Serving) ---${NC}"
    echo "  7) 4-GPU Distributed LoRA Fine-Tuning (Single Node)"
    echo "  8) Multi-Node Distributed LoRA Fine-Tuning (FSDP)"
    echo "  9) Multi-GPU Distributed Evaluation (MMLU)"
    echo " 10) Multi-GPU Distributed Profiling"
    echo " 11) Post-Training Quantization (BitsAndBytes 4-bit / 8-bit)"
    echo " 12) Launch vLLM OpenAI-Compatible Server"
    echo " 13) Connect to vLLM Server via Terminal Chat"
    echo " 14) Distributed High-Throughput Batch Inference"
    echo ""
    echo -e "${BOLD}--- UTILITIES ---${NC}"
    echo " 15) Monitor an Active Job (Real-Time GPU Utilization)"
    echo " 16) View Active Slurm Queue (squeue --me)"
    echo " 17) Run Pre-Flight Environment & Directory Health Check"
    echo "  q) Quit"
    echo -e "${BOLD}${BLUE}================================================================${NC}"
    read -rp "Enter choice [1-17 or q]: " choice

    case "$choice" in
        1)
            backend=$(choose_env)
            "${SCRIPT_ROOT}/run.sh" day1 lora --env="${backend}"
            ;;
        2)
            backend=$(choose_env)
            "${SCRIPT_ROOT}/run.sh" day1 lora --env="${backend}" --with-eval
            ;;
        3)
            backend=$(choose_env)
            "${SCRIPT_ROOT}/run.sh" day1 qlora --env="${backend}"
            ;;
        4)
            backend=$(choose_env)
            "${SCRIPT_ROOT}/run.sh" day1 eval --env="${backend}"
            ;;
        5)
            backend=$(choose_env)
            "${SCRIPT_ROOT}/run.sh" day1 eval --env="${backend}" --base
            ;;
        6)
            echo ""
            echo "1) LoRA Profiling  2) QLoRA Profiling"
            read -rp "Choice [1-2] (default 1): " p_choice
            if [ "$p_choice" == "2" ]; then
                "${SCRIPT_ROOT}/run.sh" day1 profile --method=qlora
            else
                "${SCRIPT_ROOT}/run.sh" day1 profile --method=lora
            fi
            ;;
        7)
            backend=$(choose_env)
            "${SCRIPT_ROOT}/run.sh" day2 ft-single --env="${backend}"
            ;;
        8)
            backend=$(choose_env)
            "${SCRIPT_ROOT}/run.sh" day2 ft-multi --env="${backend}"
            ;;
        9)
            backend=$(choose_env)
            "${SCRIPT_ROOT}/run.sh" day2 eval --env="${backend}"
            ;;
        10)
            "${SCRIPT_ROOT}/run.sh" day2 profile
            ;;
        11)
            echo ""
            echo "1) BitsAndBytes 4-bit  2) BitsAndBytes 8-bit  3) TorchAO  4) GPTQModel"
            read -rp "Choice [1-4] (default 1): " q_choice
            case "$q_choice" in
                2) "${SCRIPT_ROOT}/run.sh" day2 quantize bnb8 ;;
                3) "${SCRIPT_ROOT}/run.sh" day2 quantize torchao ;;
                4) "${SCRIPT_ROOT}/run.sh" day2 quantize gptq ;;
                *) "${SCRIPT_ROOT}/run.sh" day2 quantize bnb4 ;;
            esac
            ;;
        12)
            backend=$(choose_env)
            "${SCRIPT_ROOT}/run.sh" day2 serve --env="${backend}"
            ;;
        13)
            "${SCRIPT_ROOT}/run.sh" day2 chat
            ;;
        14)
            backend=$(choose_env)
            "${SCRIPT_ROOT}/run.sh" day2 inference --env="${backend}"
            ;;
        15)
            read -rp "Enter Slurm JobID: " jid
            if [ -n "$jid" ]; then
                "${SCRIPT_ROOT}/run.sh" monitor "$jid"
            fi
            ;;
        16)
            echo ""
            squeue --me 2>/dev/null || echo "squeue command not found (not on Slurm cluster)."
            ;;
        17)
            "${SCRIPT_ROOT}/check_env.sh"
            ;;
        q|Q|exit)
            echo -e "\nExiting launcher. Good luck with your course runs!\n"
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Invalid option. Please try again.${NC}"
            ;;
    esac

    echo ""
    read -rp "Press [Enter] to return to the menu..." dummy
    clear_screen
done

