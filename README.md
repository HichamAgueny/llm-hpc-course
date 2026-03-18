# LLM Fine‑tuning and Inference on HPC

Welcome to the **LLM Fine‑tuning and Inference on HPC** course! This repository provides a hands‑on guide to training and deploying Large Language Models (LLMs) on High‑Performance Computing (HPC) clusters.

## Course Overview

This course covers the full lifecycle of working with LLMs in an HPC environment, from setting up the environment to distributed training and high‑throughput inference.

## Course Program

### Day 1 — Single-GPU Fine-Tuning & HPC Foundations
**Theme:** Build an efficient single-GPU fine-tuning workflow on an HPC system.

#### Morning Session (09:30–12:00) — HPC Fundamentals & Fine-Tuning Optimization
- **HPC Foundations for LLM Workloads**
  - Overview of Olivia Supercomputer
  - Storage hierarchy strategy
  - Containerized environments
- **LLM Fine-Tuning Fundamentals**
  - Parameter-efficient fine-tuning (LoRA, QLoRA)
  - Quantization within QLoRA (FP4, FP8, BF16)
  - Memory–throughput trade-offs

#### Afternoon Session (13:00–15:30) — Hands-On: Single-GPU Workflow
- End-to-end LoRA fine-tuning workflow
- Quantized fine-tuning: FP4 vs FP8 vs BF16 comparison
- GPU monitoring and memory profiling

#### Wrap-Up & Discussion (15:30–16:00)
**Outcome:** Participants implement and optimize a complete single-GPU fine-tuning pipeline with performance diagnostics on an HPC system.

---

### Day 2 — Distributed Training & Optimized Inference
**Theme:** Scale fine-tuning and inference across multiple GPUs while minimizing communication overhead.

#### Morning Session (09:30–12:00) — Distributed Fine-Tuning
- **Distributed Training Concepts**
  - DDP vs FSDP
  - Communication overhead and scaling efficiency
- **Hands-On: Multi-GPU Fine-Tuning**
  - Multi-GPU LoRA & QLoRA fine-tuning
  - Profiling distributed workloads
  - Throughput and scaling efficiency analysis

#### Afternoon Session (13:00–15:30) — Hands-On: Optimized Inference
- Introduction to the vLLM inference engine
- Single-GPU inference benchmarking
- Multi-GPU inference scaling
- Latency vs throughput trade-offs

#### Wrap-Up & Discussion (15:30–16:00)
**Outcome:** Participants scale fine-tuned models and inference across multiple GPUs, interpret performance metrics, and apply optimization strategies suitable for HPC allocations.

---

## Environment Setup

To initialize your project environment on the cluster, run the following setup script from the repository root. This will copy shared resources (datasets, models) and create the necessary directory structures for results and logs.

```bash
# Run the setup script
bash setup_env.sh
```

> [!IMPORTANT]
> This script is configured for the **Olivia Supercomputer** and handles the migration of resources from shared project directories.

---

## Repository Layout

```text
llm-hpc-course/
├── apptainer/       # Container definitions and launch scripts
├── configs/         # LoRA/QLoRA configuration files (YAML)
├── day1_single_gpu/ # Labs for Day 1 (Fine-tuning & Profiling)
├── day2_multi_gpu/  # Labs for Day 2 (Multi-GPU FT & Inference)
├── recipes/        # Reusable Python scripts for FT and Inference
├── utils/          # Utility scripts (monitoring, etc.)
├── LICENSE
└── README.md        # ← You are here
```

---

## Getting Started

### 0. SSH to Olivia

First, connect to **Olivia** via SSH:
```bash
ssh username@olivia.sigma2.no
(username@olivia.sigma2.no) One-time password (OATH) for `username': 
(username@olivia.sigma2.no) Password: 
username@uan02:~> 
```
Then run the following commands:

```bash
mkdir /cluster/work/projects/nn9970k/$USER
cd /cluster/work/projects/nn9970k/$USER
git clone https://github.com/HichamAgueny/llm-hpc-workshop.git
cd llm-hpc-workshop
````

### Setup Script

Run the setup script:

```bash
chmod u+x my_script.sh
./my_script.sh
```

This script will:

* Copy the Apptainer (Singularity) image and dataset from a local project to your project work area.
* Update paths in the configuration files automatically.
  
### 1. Environment Setup

We use **Apptainer** to provide a consistent environment across the cluster.

```bash
cd apptainer
# Build or use the existing image
bash launch_shell.sh
```
Follow the instructions in [apptainer/README.md](./apptainer/README.md) for more details.

### 2. Running your first job

Navigate to the Day 1 labs to start your first single‑GPU fine‑tuning job.

```bash
cd day1_single_gpu/finetuning
sbatch job_singleGPU_LoRA.sh
```

---

## Resources & Documentation

- [Day 1: Single GPU Fine-tuning](./day1_single_gpu/README.md)
- [Day 2: Multi-GPU & Inference](./day2_multi_gpu/README.md)
- [Profiling Guide (Single GPU)](./day1_single_gpu/profiling/README.md)
- [Profiling Guide (Multi-GPU)](./day2_multi_gpu/profiling/README.md)
- [Configuration Guide](./configs/README.md)
- [Python Recipes Documentation](./recipes/README.md)
- [Utility Scripts Documentation](./utils/README.md)

---

## Contact
**Hicham Agueny**
[GitHub Profile](https://github.com/HichamAgueny)
