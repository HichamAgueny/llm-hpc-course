# Day 1: Single GPU Workflow

This directory contains labs and scripts for performing LLM fine‑tuning, evaluation, and performance profiling on a single GPU using the Alpaca GPT-4 Question-Answering (QA) dataset.

## Learning Objectives

- Learn how to submit Slurm jobs for LLM fine‑tuning using both Apptainer containers and EESSI software modules.
- Compare **LoRA** and **QLoRA** techniques in terms of memory usage, throughput, and accuracy.
- Evaluate base and fine-tuned models on benchmarks such as MMLU.
- Profile single-GPU training jobs to analyze memory allocation and compute bottlenecks.

---

## Directory Contents

| Folder | Description |
| :--- | :--- |
| `finetuning/` | Slurm job scripts for single‑GPU fine‑tuning on the QA dataset (LoRA and QLoRA). Supports both containerized (`job_container/`) and EESSI-based (`job_eessi/`) execution. |
| `evaluation/` | Slurm job scripts to evaluate base and fine-tuned models (e.g., MMLU benchmark) using EleutherAI LM Evaluation Harness. |
| `profiling/` | Job scripts configured to measure compute and memory metrics using PyTorch Profiler. |

---

## Output Directory Organization

All outputs are written into structured subdirectories under `${PROJECT_DIR}/results`:

- **`${PROJECT_DIR}/results/checkpoint_results/`**: Model checkpoints and adapter weights generated during fine-tuning (e.g., `llama3.2-1b_lora_single_device/checkpoints/epoch_0`).
- **`${PROJECT_DIR}/results/profiling_results/`**: PyTorch Profiler traces (`profiles/`) and training logs (`logs/`).
- **`${PROJECT_DIR}/results/evaluation_results/`**: Benchmark metrics, evaluation logs, and score summaries.

---

## Guided Hands-On Lab Worksheets (Day 1)

### **Lab 1.1: Single-GPU LoRA Fine-Tuning**
* **Objective:** Fine-tune LLaMA 3.2-1B on the Question-Answering (QA) task using BF16 Low-Rank Adaptation (LoRA).
* **Launch Commands:**
  * Fast launcher (from repo root):
    ```bash
    ./run.sh day1 lora               # Using Apptainer container
    ./run.sh day1 lora --env=eessi   # Using EESSI modules
    ```
  * Direct Slurm submission:
    ```bash
    # LoRA (Container):
    cd day1_single_gpu/finetuning/lora_slurm/job_container && sbatch job_singleGPU_QA_LoRA.sh
    # LoRA (EESSI):
    cd day1_single_gpu/finetuning/lora_slurm/job_eessi && sbatch job_singleGPU_QA_LoRA_eessi.sh
    ```
* **Real-Time Monitoring:** `./run.sh monitor <JobID>` or `./utils/gpu.monitor.sh <JobID>`
* **Where to Inspect Outputs:**
  * **Slurm Job Logs (stdout/stderr):** `day1_single_gpu/finetuning/lora_slurm/job_container/out/*.out`
  * **Training Loss & Step Metrics:** `results/checkpoint_results/llama3.2-1b_lora_single_device/logs/`
  * **Saved Model Checkpoints:** `results/checkpoint_results/llama3.2-1b_lora_single_device/checkpoints/epoch_0/`

---

### **Lab 1.2: Quantized Fine-Tuning with QLoRA**
* **Objective:** Fine-tune LLaMA 3.2-1B using 4-bit NormalFloat (NF4) base-model weight quantization with LoRA adapters.
* **Launch Commands:**
  * Fast launcher (from repo root):
    ```bash
    ./run.sh day1 qlora               # Using Apptainer container
    ./run.sh day1 qlora --env=eessi   # Using EESSI modules
    ```
  * Direct Slurm submission:
    ```bash
    # QLoRA (Container):
    cd day1_single_gpu/finetuning/qlora_slurm/job_container && sbatch job_singleGPU_QA_QLoRA.sh
    # QLoRA (EESSI):
    cd day1_single_gpu/finetuning/qlora_slurm/job_eessi && sbatch job_singleGPU_QA_QLoRA_eessi.sh
    ```
* **Where to Inspect Outputs:**
  * **Slurm Job Logs (stdout/stderr):** `day1_single_gpu/finetuning/qlora_slurm/job_container/out/*.out`
  * **Training Loss & Step Metrics:** `results/checkpoint_results/llama3.2-1b_qlora_single_device/logs/`
  * **Saved Model Checkpoints:** `results/checkpoint_results/llama3.2-1b_qlora_single_device/checkpoints/`
* **Student Focus:** Compare training step throughput (tokens/sec) between LoRA (BF16) and QLoRA (NF4 dequantization overhead) in the stdout logs.

---

### **Lab 1.3: GPU Memory & Kernel Profiling**
* **Objective:** Profile memory allocation, activation footprint, and kernel execution timelines using PyTorch Profiler.
* **Launch Commands:**
  * Fast launcher (from repo root):
    ```bash
    ./run.sh day1 profile --method=lora    # Profile LoRA
    ./run.sh day1 profile --method=qlora   # Profile QLoRA
    ```
  * Direct Slurm submission:
    ```bash
    cd day1_single_gpu/profiling && sbatch job_singleGPU_QA_LoRA_profiling.sh
    # or for QLoRA:
    cd day1_single_gpu/profiling && sbatch job_singleGPU_QA_QLoRA_profiling.sh
    ```
* **Where to Inspect Outputs:**
  * **Slurm Job Logs:** `day1_single_gpu/profiling/out/*.out`
  * **Profiler Traces & Memory Timelines:**
    * LoRA: `results/profiling_results/profiling_llama3.2-1b_lora_single_device/profiles/`
    * QLoRA: `results/profiling_results/profiling_llama3.2-1b_qlora_single_device/profiles/`
* **Viewing Traces Locally on Your Laptop:**
  Download the memory timeline HTML trace to inspect in your browser:
  ```bash
  scp USERNAME@olivia.sigma2.no:/cluster/projects/nn9997k/USERNAME/llm-hpc-course/results/profiling_results/profiling_llama3.2-1b_lora_single_device/profiles/rank0_memory-timeline.html .
  open rank0_memory-timeline.html   # macOS (or xdg-open on Linux)
  ```
* **Key Analytical Questions for Students:**
  * How much peak VRAM was saved by 4-bit base weight quantization in QLoRA compared to LoRA?
  * Inspect the memory timeline HTML (`rank0_memory-timeline.html`): What proportion of memory is occupied by model weights vs. activations during the forward and backward passes?
  * What fraction of iteration time is spent on GPU compute vs. CPU host dispatching?

---

### **Lab 1.4: Model Merging & Benchmark Evaluation (MMLU)**
* **Objective:** Merge your trained LoRA adapter weights with base LLaMA weights and measure benchmark reasoning accuracy on MMLU tasks.
* **Prerequisite:** Lab 1.4 evaluates the fine-tuned adapter from Lab 1.1. Ensure Lab 1.1 has completed and saved its checkpoint to `results/checkpoint_results/llama3.2-1b_lora_single_device/checkpoints/epoch_0` before submitting.
  > [!TIP]
  > You can train and automatically evaluate in a single step using `./run.sh day1 lora --with-eval`, which chains the evaluation job to run automatically when training succeeds!
* **Launch Commands:**
  * Fast launcher (from repo root):
    ```bash
    ./run.sh day1 eval         # Evaluate fine-tuned LoRA model
    ./run.sh day1 eval --base  # Evaluate base model (baseline)
    ```
  * Direct Slurm submission:
    ```bash
    cd day1_single_gpu/evaluation && sbatch job_singleGPU_LoRA.sh
    # or for base model:
    cd day1_single_gpu/evaluation && sbatch job_singleGPU_No_LoRA.sh
    ```
* **Where to Inspect Outputs:**
  * **Slurm Job Logs:** `day1_single_gpu/evaluation/out/*.out`
  * **Benchmark Scores & Summary Tables:** `results/evaluation_results/llama3.2-1b-mmlu_singlegpu/`
* **Student Focus:** Compare accuracy metrics between the unadapted base model and your fine-tuned model on question-answering subjects.

---

## Resource Usage & Monitoring

- Monitor job status with `squeue --me`.
- While a job is running, inspect real-time GPU metrics:
  ```bash
  ./evaluation/gpu.monitor.sh <JobID>
  # or from utils:
  ../utils/gpu.monitor.sh <JobID>
  ```
- Alternatively, connect directly to the allocated GPU node: `ssh <gpu-node-name>` and run `nvidia-smi` or `nvitop`.
- Check stdout/stderr log files in the `out/` directory for run progress and training statistics.
