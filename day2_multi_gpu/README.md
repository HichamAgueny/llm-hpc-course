# Day 2: Multi‑GPU & Distributed Systems

This directory focuses on scaling LLM workloads across multiple GPUs, including distributed fine‑tuning and high‑throughput inference.

## Learning Objectives

- Scale fine‑tuning to multiple GPUs using **FSDP** (Fully Sharded Data Parallel) or other distributed strategies.
- Profile your distributed workloads to identify communication bottlenecks. (See the [Multi-GPU Profiling Guide](./profiling/README.md)).
- Deploy LLMs for inference using **vLLM** for optimized throughput.
- Compare performance between single‑GPU and multi‑GPU setups.

---

## Directory Contents

| Folder | Description |
| :--- | :--- |
| `fine-tuning/` | Slurm job scripts for distributed LoRA and QLoRA. |
| `inference/` | Scripts for high‑performance inference using vLLM, including specific tasks like QA and XSum. |

---

## Getting Started

### 1. Distributed Fine-tuning
Navigate to the `fine-tuning` subfolder and submit a multi‑GPU job:

```bash
cd fine-tuning/lora_slurm
sbatch job_multiGPU_LoRA.sh
```

### 2. High-Performance Inference (vLLM)
Use the `inference` directory to run high‑throughput inference. We provide setup scripts for interactive and batch mode:

```bash
cd inference
# For interactive session inside the container
bash srun_interactive
bash launch_shell.sh
```

To run a specific task (e.g., Question Answering):
```bash
cd task_QA
sbatch slurm_vllm_distributed.sh
```

---

## Technical Notes

- **Distributed Strategy**: Check the job scripts to see how GPUs are allocated (`--gpus-per-node`) and how distributed training is launched.
- **vLLM**: The inference scripts utilize vLLM module, specifically `vLLM/0.11.0` to serve models efficiently. Ensure you first load the `NRIS/GPU` environment.
