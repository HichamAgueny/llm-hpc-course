# Parameter‑Efficient Fine‑Tuning (PEFT) & Evaluation Configurations

This directory contains YAML configuration files for fine‑tuning and evaluating LLMs using **LoRA** (Low‑Rank Adaptation), **QLoRA** (Quantized LoRA), and **QAT-LoRA** (Quantization-Aware Training with LoRA). All training configurations are tailored for Question-Answering (QA) on the Alpaca GPT-4 dataset.

## Directory Structure

| Folder | Description |
| :--- | :--- |
| `lora/` | LoRA adaptation configurations in BF16 precision for single-device, multi-device, and multi-node training, plus profiling. |
| `qlora/` | Quantized 4‑bit configurations using `bitsandbytes` NF4 quantization for memory‑efficient fine-tuning and profiling. |
| `qat-lora/` | Quantization-Aware Training (QAT) configurations for multi-device and multi-node setups. |
| `eval_cfg/` | Evaluation configuration (`eval_config.yaml`) for benchmarking models on MMLU tasks. |

---

## Configuration Files Overview

### 1. Single-Device (Single GPU — Day 1)
- `lora/llama3_2_1B_lora_single_device_QA.yaml`: Standard LoRA fine-tuning for LLaMA 3.2-1B.
- `lora/llama3_2_1B_lora_single_device_profiling.yaml`: LoRA run with PyTorch Profiler enabled.
- `qlora/llama3_2_1B_qlora_single_device_QA.yaml`: 4-bit QLoRA fine-tuning.
- `qlora/llama3_2_1B_qlora_single_device_profiling.yaml`: 4-bit QLoRA run with PyTorch Profiler enabled.

### 2. Multi-Device Distributed (Single Node, Multiple GPUs — Day 2)
- `lora/llama3_2_1B_lora_multi_device_QA.yaml`: Distributed LoRA fine-tuning using FSDP.
- `lora/llama3_2_1B_lora_multi_device_profiling.yaml`: Distributed LoRA with profiler enabled.
- `qlora/llama3_2_1B_qlora_multi_device_QA.yaml`: Distributed QLoRA fine-tuning with FSDP.
- `qlora/llama3_2_1B_qlora_multi_device_profiling.yaml`: Distributed QLoRA with profiler enabled.
- `qat-lora/llama3_2_1B_qat-lora_multi_device_QA.yaml`: Distributed Quantization-Aware Training (QAT).
- `qat-lora/llama3_2_1B_qat-lora_multi_device_profiling.yaml`: Distributed QAT with profiler enabled.

### 3. Multi-Node Distributed (Multiple Nodes, Multiple GPUs — Day 2)
- `lora/llama3_2_1B_lora_multi_node_QA.yaml`: Multi-node LoRA fine-tuning across Slurm nodes.
- `qlora/llama3_2_1B_qlora_multi_node_QA.yaml`: Multi-node QLoRA fine-tuning across Slurm nodes.
- `qat-lora/llama3_2_1B_qat-lora_multi_node_QA.yaml`: Multi-node QAT-LoRA fine-tuning across Slurm nodes.

### 4. Evaluation
- `eval_cfg/eval_config.yaml`: Configuration for EleutherAI LM Evaluation Harness evaluating on MMLU.

---

## Usage

These YAML configurations are typically passed to recipes via Slurm batch scripts or directly through the CLI:

```bash
# Example: Running single-device LoRA fine-tuning
python recipes/single_device/lora_finetune_single_device.py \
    --config configs/lora/llama3_2_1B_lora_single_device_QA.yaml \
    --checkpoint_dir /cluster/work/projects/nn9971k/$USER/llm-hpc-course/shared/models/Llama-3.2-1B-Instruct \
    --tokenizer_dir /cluster/work/projects/nn9971k/$USER/llm-hpc-course/shared/models/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --dataset_dir /cluster/work/projects/nn9971k/$USER/llm-hpc-course/shared/datasets/GPT-4-LLM/alpaca_gpt4_data.json \
    --output_dir /cluster/work/projects/nn9971k/$USER/llm-hpc-course/results/checkpoint_results/my_run/checkpoints \
    --log_dir /cluster/work/projects/nn9971k/$USER/llm-hpc-course/results/checkpoint_results/my_run/logs \
    --profiling_dir /cluster/work/projects/nn9971k/$USER/llm-hpc-course/results/checkpoint_results/my_run/profiles
```

> [!TIP]
> Slurm scripts in `day1_single_gpu/` and `day2_multi_gpu/` automate passing the appropriate `--config`, input model directories, and output result directories (`results/checkpoint_results`, `results/profiling_results`, `results/evaluation_results`).
