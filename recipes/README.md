# Python Recipes for LLM Workloads

This directory contains standalone Python recipes for fine‑tuning, quantization, model merging, evaluation, and inference used across the course.

## Directory Structure

| Folder | Description |
| :--- | :--- |
| `single_device/` | Single-GPU training and quantization recipes (LoRA / QLoRA). |
| `distributed/` | Distributed training recipes using PyTorch FSDP (LoRA, QLoRA, QAT-LoRA). |
| `merge/` | Utility scripts (`merge_and_save.py`) to merge trained LoRA adapter weights with base model checkpoints for evaluation and serving. |
| `eval/` | Benchmark evaluation recipes (`eleuther_eval.py`) integrating EleutherAI LM Evaluation Harness (e.g., MMLU). |
| `quanti/` | Standalone post-training quantization recipes (BitsAndBytes 4-bit/8-bit, TorchAO, GPTQModel). |
| `inference/` | High-throughput batch and distributed inference recipes powered by vLLM. |
| `bench/` | Benchmarking utilities and log parsers. |

---

## Core Recipes

### 1. Fine-Tuning
- **`recipes/single_device/lora_finetune_single_device.py`**: Entry point for single-GPU fine-tuning (LoRA / QLoRA) driven by torchtune.
- **`recipes/distributed/lora_finetune_distributed.py`**: Multi-GPU and multi-node distributed fine-tuning with PyTorch FSDP.
- **`recipes/distributed/qat_lora_finetune_distributed.py`**: Distributed Quantization-Aware Training with LoRA.

### 2. Model Merging & Evaluation
- **`recipes/merge/merge_and_save.py`**: Merges LoRA adapters from `results/checkpoint_results/` back into the base LLaMA model and saves standard Hugging Face safetensors.
- **`recipes/eval/eleuther_eval.py`**: Runs evaluation tasks against merged models or unadapted base models.

### 3. Post-Training Quantization
- **`recipes/quanti/quat_bnb_4bit.py`** & **`quat_bnb_8bit.py`**: BitsAndBytes weight-only quantization.
- **`recipes/quanti/quat_gptq_with_filter.py`**: GPTQ calibration and quantization.
- **`recipes/quanti/qat_torchoa.py`**: PyTorch TorchAO quantization integration.

### 4. High-Throughput Inference
- **`recipes/inference/vllm_inference.py`**: Single-GPU batch inference with vLLM and optional LoRA adapter loading.
- **`recipes/inference/vllm_distributed.py`**: Multi-GPU distributed inference with tensor parallelism in vLLM.

---

## Usage Examples

### Single-Device QA Fine-Tuning
```bash
python recipes/single_device/lora_finetune_single_device.py \
    --config configs/lora/llama3_2_1B_lora_single_device_QA.yaml
```

### Merging LoRA Adapter with Base Model
```bash
python recipes/merge/merge_and_save.py \
    --base_model /cluster/projects/nn9997k/$USER/llm-hpc-course/shared/models/Llama-3.2-1B-Instruct \
    --adapter_dir /cluster/projects/nn9997k/$USER/llm-hpc-course/results/checkpoint_results/llama3.2-1b_lora_single_device/checkpoints/epoch_0 \
    --output_dir /cluster/projects/nn9997k/$USER/llm-hpc-course/results/checkpoint_results/Llama-3.2-1B-Instruct_merged
```

### Serving / Inference with vLLM
```bash
python recipes/inference/vllm_inference.py \
    --model_id /cluster/projects/nn9997k/$USER/llm-hpc-course/shared/models/Llama-3.2-1B-Instruct \
    --prompt "What is High-Performance Computing?"
```
