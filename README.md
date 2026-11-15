# LLM Fine‑tuning, Serving, and Inference on HPC

Welcome to the **LLM Fine‑tuning, Serving, and Inference on HPC** course! This repository provides a hands‑on guide to training, quantizing, evaluating, and deploying Large Language Models (LLMs) on High‑Performance Computing (HPC) clusters (specifically the **Olivia Supercomputer**).

All fine-tuning labs focus on **Question-Answering (QA)** using the Alpaca GPT-4 dataset.

---

## Course Overview

This course covers the full lifecycle of working with LLMs in an HPC environment—from single-GPU parameter-efficient fine-tuning (PEFT) to multi-node distributed training, post-training quantization, benchmark evaluation, and high-throughput vLLM serving.

```mermaid
flowchart TD
    subgraph Day1["Day 1: Single-GPU Foundations"]
        A["Base Model<br>(LLaMA 3.2-1B)"] --> B["PEFT Fine-Tuning<br>(LoRA / QLoRA)"]
        B --> C["Performance Profiling<br>(PyTorch Profiler Traces)"]
        B --> D["Adapter Merging & Eval<br>(MMLU Benchmark)"]
    end

    subgraph Day2["Day 2: Distributed Scale & Serving"]
        D --> E["Distributed FSDP Training<br>(Single-Node 4x GPU / Multi-Node)"]
        E --> F["Post-Training Quantization<br>(BitsAndBytes / TorchAO / GPTQ)"]
        E --> G["vLLM High-Throughput Serving<br>(Interactive Chat Client)"]
        E --> H["Distributed Batch Inference<br>(Tensor Parallelism)"]
    end
```

## Course Program

### Day 1 — Single-GPU Fine-Tuning, Evaluation & Profiling
**Theme:** Build an efficient, reproducible single-GPU fine-tuning and evaluation workflow on an HPC system.

#### Morning Session (09:30–12:00) — HPC Fundamentals & Fine-Tuning Principles
- **HPC Foundations for LLM Workloads**
  - Architecture of the Olivia Supercomputer (NVIDIA Grace Hopper GH200 architecture)
  - Storage hierarchy: shared project shares, local `/tmp` scratch, and cache management
  - Software environments: Apptainer containers vs. native EESSI software modules
- **Parameter-Efficient Fine-Tuning (PEFT) Fundamentals**
  - Low-Rank Adaptation (LoRA): mathematical foundations, rank selection, and parameter efficiency
  - Quantized LoRA (QLoRA): 4-bit NormalFloat (NF4), double quantization, and dequantization
  - The Question-Answering (QA) benchmark task on the Alpaca GPT-4 dataset

#### Afternoon Session (13:00–15:30) — Hands-On: Single-GPU Workflow
- **End-to-End Fine-Tuning Workflow:**
  - Launching single-GPU LoRA and QLoRA fine-tuning jobs (Apptainer vs. EESSI)
  - Quantized fine-tuning trade-offs: FP4/NF4 memory savings vs. BF16 training throughput
- **Model Merging & Benchmark Evaluation:**
  - Merging trained LoRA adapter weights with base LLaMA weights into standalone Hugging Face checkpoints
  - Running MMLU benchmark evaluation with EleutherAI LM Evaluation Harness (comparing base model vs. fine-tuned adapter)
- **Performance Diagnostics & Monitoring:**
  - Real-time GPU monitoring with `gpu.monitor.sh` and interactive `nvitop`
  - Deep-dive PyTorch Profiler traces: memory timeline visualization and kernel analysis

#### Wrap-Up & Discussion (15:30–16:00)
**Outcome:** Participants implement, evaluate, and optimize a complete single-GPU QA fine-tuning pipeline with diagnostics, outputting structured artifacts to `results/checkpoint_results`, `results/profiling_results`, and `results/evaluation_results`.

---

### Day 2 — Distributed Training, Quantization, Serving & Inference
**Theme:** Scale fine-tuning across multiple GPUs and nodes, apply post-training quantization, and deploy high-throughput serving pipelines.

#### Morning Session (09:30–12:00) — Distributed Fine-Tuning, Evaluation & Profiling
- **Distributed Training Concepts & Interconnects**
  - PyTorch Fully Sharded Data Parallel (FSDP) vs. Distributed Data Parallel (DDP)
  - Inter-GPU interconnect topology: NVLink intra-node and HPE Slingshot / Cray Fabric inter-node
- **Hands-On: Multi-GPU & Multi-Node Fine-Tuning**
  - Single-node 4-GPU distributed LoRA, QLoRA, and Quantization-Aware Training (QAT-LoRA) with FSDP
  - Multi-node scaling across Slurm nodes with optimized AWS-OFI-NCCL network bindings
- **Hands-On: Distributed Evaluation & Profiling**
  - Multi-GPU MMLU evaluation of distributed checkpoints
  - Multi-rank profiling: identifying inter-GPU communication bottlenecks (all-reduce, reduce-scatter)

#### Afternoon Session (13:00–15:30) — Hands-On: Quantization, Serving & High-Throughput Inference
- **Hands-On: Post-Training Quantization (PTQ)**
  - Weight-only 4-bit and 8-bit quantization with BitsAndBytes
  - Modern kernel quantization using PyTorch TorchAO
  - Calibration-driven quantization with GPTQModel
- **Hands-On: LLM Serving with vLLM**
  - Deploying OpenAI-compatible vLLM API servers on single and multiple GPUs
  - Connecting and interacting with models via the terminal chat client (`chat.sh`)
- **Hands-On: High-Throughput Distributed Batch Inference**
  - Batch inference scaling with tensor parallelism in vLLM
  - Dynamic loading of trained LoRA adapters from `results/checkpoint_results/`

#### Wrap-Up & Discussion (15:30–16:00)
**Outcome:** Participants scale fine-tuned QA models across multiple GPUs and nodes, quantize models for edge/production constraints, evaluate performance on standard benchmarks, and deploy high-throughput vLLM serving on HPC clusters.

---

## Repository Layout

```text
llm-hpc-course/
├── run.sh           # Unified CLI runner for all Day 1 & Day 2 jobs
├── menu.sh          # Interactive terminal menu launcher
├── check_env.sh     # Pre-flight environment & directory health check
├── course_env.sh    # Centralized Slurm account & path configuration
├── apptainer/       # Container definitions (.def), build guides, and launch scripts
├── configs/         # YAML configurations (LoRA, QLoRA, QAT-LoRA, and Evaluation)
├── day1_single_gpu/ # Single-GPU labs: Fine-Tuning, Evaluation, and Profiling
├── day2_multi_gpu/  # Multi-GPU labs: Distributed FT, Evaluation, Profiling, Quantization, Serving & Inference
├── recipes/         # Standalone Python recipes for training, merging, evaluation, quantization, and vLLM
├── utils/           # Cluster utilities (GPU monitoring, interactive nvitop)
├── run_course.sh    # Multi-node / single-point execution launcher
└── README.md        # ← You are here
```

---

## Output Directory Structure (`results/`)

All training, profiling, and evaluation runs save outputs into organized subfolders under `results/`:

```text
results/
├── checkpoint_results/  # Checkpoints and adapter weights produced by fine-tuning
├── profiling_results/   # PyTorch Profiler JSON/HTML traces and training logs
└── evaluation_results/  # EleutherAI LM Evaluation Harness benchmark outputs (MMLU)
```

---

## Getting Started on Olivia

### 1. SSH to Olivia & Clone Repo

Connect to the **Olivia Supercomputer** via SSH:
```bash
ssh username@olivia.sigma2.no
```

Navigate to your user directory on the project share and clone the repository:
```bash
cd /cluster/projects/nn9997k/$USER
git clone https://github.com/HichamAgueny/llm-hpc-course.git
cd llm-hpc-course
```

---

### 2. Pre-Flight Health Check

Verify cluster paths, permissions, container images, and initialize result folders:
```bash
./check_env.sh
```

---

### 3. Running Jobs: Three Simple Ways

You can run any lab using either the **Interactive Menu**, the **Unified CLI Runner**, or directly via standard `sbatch`.

#### Option A: Interactive Menu (Easiest)
Simply run the interactive text menu and pick any Day 1 or Day 2 lab by number:
```bash
./menu.sh
```

#### Option B: Unified CLI Runner (`./run.sh`)
Submit jobs directly from the repository root without navigating into deep directories:

```bash
# --- DAY 1 (Single GPU) ---
./run.sh day1 lora               # Submit single-GPU LoRA fine-tuning (Container)
./run.sh day1 lora --env=eessi   # Submit single-GPU LoRA fine-tuning (EESSI)
./run.sh day1 lora --with-eval   # Train AND automatically schedule evaluation when training succeeds!
./run.sh day1 qlora              # Submit single-GPU 4-bit QLoRA
./run.sh day1 eval               # Evaluate LoRA adapter on MMLU
./run.sh day1 profile            # Profile GPU memory and kernel execution

# --- DAY 2 (Multi-GPU Distributed) ---
./run.sh day2 ft-single          # 4-GPU distributed fine-tuning (single node)
./run.sh day2 ft-multi           # Multi-node distributed fine-tuning
./run.sh day2 eval               # Multi-GPU MMLU evaluation
./run.sh day2 quantize bnb4      # 4-bit BitsAndBytes quantization
./run.sh day2 serve              # Launch vLLM OpenAI-compatible server
./run.sh day2 chat               # Connect interactive chat client to vLLM server
./run.sh day2 inference          # Distributed high-throughput batch inference

# --- UTILITIES ---
./run.sh monitor <JobID>         # Attach real-time GPU monitor to your active job
./run.sh status                  # Check active Slurm queue (squeue --me)
```

#### Option C: Manual Slurm Submissions
If you prefer inspecting or running Slurm scripts directly:
```bash
# Day 1 LoRA (Container)
cd day1_single_gpu/finetuning/lora_slurm/job_container
sbatch job_singleGPU_QA_LoRA.sh

# Day 1 LoRA (EESSI)
cd day1_single_gpu/finetuning/lora_slurm/job_eessi
sbatch job_singleGPU_QA_LoRA_eessi.sh
```

---

## Documentation Links

- [Day 1 Guide: Single-GPU Workflow](./day1_single_gpu/README.md)
  - [Day 1 Profiling Guide](./day1_single_gpu/profiling/README.md)
- [Day 2 Guide: Multi-GPU & Distributed Workflow](./day2_multi_gpu/README.md)
  - [Day 2 Profiling Guide](./day2_multi_gpu/profiling/README.md)
- [Configuration Guide (`configs/`)](./configs/README.md)
- [Python Recipes Guide (`recipes/`)](./recipes/README.md)
- [Apptainer Container Guide (`apptainer/`)](./apptainer/README.md)
- [Utilities & Monitoring Guide (`utils/`)](./utils/README.md)

---

## Contact
**Hicham Agueny** — [GitHub Profile](https://github.com/HichamAgueny)
