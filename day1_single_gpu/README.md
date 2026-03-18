# Day 1: Single GPU Workflow

This directory contains labs and scripts for performing LLM fine‑tuning and performance profiling on a single GPU.

## Learning Objectives

- Learn how to submit Slurm jobs for LLM fine‑tuning.
- Compare **LoRA** and **QLoRA** techniques in terms of memory usage and speed.
- Profile your training jobs to understand resource bottlenecks.

---

## Directory Contents

| Folder | Description |
| :--- | :--- |
| `finetuning/` | Slurm job scripts for single‑GPU fine‑tuning on the XSum dataset. |
| `profiling/` | Job scripts configured to measure compute and memory metrics. |

---

## Getting Started

### 1. Fine-tuning
Navigate to the `finetuning` directory and choose a script (LoRA or QLoRA):

```bash
For LoRA
cd finetuning/lora_slurm
sbatch job_singleGPU_Xsum_LoRA.sh

For QLoRA
cd finetuning/qlora_slurm
sbatch job_singleGPU_Xsum_QLoRA.sh
```

### 2. Profiling
To analyze the performance of your training, use the profiling scripts:

```bash
cd profiling
sbatch job_singleGPU_profiling_LoRA.sh
```

---

## Resource Usage Tips

- Use `squeue --me` to monitor your job status.
- Once the job is running, run `./gpu_monitor.sh your-jobID`  from this `/cluster/work/projects/nn9997k/hicham/llm-hpc-course/utils` to check GPU utilization.
- Review the `.out` files in the `out/` subdirectories for logs and performance summaries.
