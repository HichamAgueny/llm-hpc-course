# Day 1: Single GPU Workflow

This directory contains labs and scripts for performing LLM fine‑tuning, evaluation, and performance profiling on a single GPU using the Alpaca GPT-4 Question-Answering (QA) dataset.

## Learning Objectives

- Learn how to submit Slurm jobs for LLM fine‑tuning using both Apptainer containers and EESSI software modules.
- Compare **LoRA** and **QLoRA** techniques in terms of memory usage, throughput, and accuracy.
- Evaluate base and fine-tuned models on benchmarks such as MMLU.
- Profile single-GPU training jobs to analyze memory allocation and compute bottlenecks.

---

## Before You Start (once)

From the course folder (where `run.sh` is):
```bash
./setup_env.sh          # first time only: copies models/datasets/containers, creates folders, runs checks
source course_env.sh    # once per login: sets the Slurm account & reservation and $COURSE_ROOT
```

**You never need to edit paths or accounts in the job scripts.** Every lab can be started in two ways:

- **Fast launcher** — run from the course folder: `./run.sh day1 ...` (or pick it in `./menu.sh`)
- **Direct Slurm submission** — the commands below use `$COURSE_ROOT`, so they work from any folder
  once you have run `source course_env.sh`.

Each job prints at the end where its results are and what to run next.

---

## Directory Contents

| Folder | Description |
| :--- | :--- |
| `finetuning/` | Slurm job scripts for single‑GPU fine‑tuning on the QA dataset (LoRA and QLoRA). Supports both containerized (`job_container/`) and EESSI-based (`job_eessi/`) execution. |
| `evaluation/` | Slurm job scripts to evaluate base and fine-tuned models (e.g., MMLU benchmark) using EleutherAI LM Evaluation Harness. |
| `profiling/` | Job scripts configured to measure compute and memory metrics using PyTorch Profiler. |

---

## Where Things Are Written

- **Slurm log of each job:** the `out/` folder next to the job script (e.g. `finetuning/lora_slurm/job_container/out/`).
- **Results:** under `results/` in the course folder:
  - **`results/checkpoint_results/`**: Model checkpoints and adapter weights generated during fine-tuning (e.g., `llama3.2-1b_lora_single_device/checkpoints/epoch_0`). EESSI runs use the prefix `eessi_`.
  - **`results/profiling_results/`**: PyTorch Profiler traces (`profiles/`) and training logs (`logs/`).
  - **`results/evaluation_results/`**: Benchmark metrics, evaluation logs, and score summaries.

---

## Guided Hands-On Lab Worksheets (Day 1)

### **Lab 1.1: Single-GPU LoRA Fine-Tuning**
* **Objective:** Fine-tune LLaMA 3.2-1B on the Question-Answering (QA) task using BF16 Low-Rank Adaptation (LoRA).
* **Launch Commands:**
  * Fast launcher (from the course folder):
    ```bash
    ./run.sh day1 lora               # Using Apptainer container
    ./run.sh day1 lora --env=eessi   # Using EESSI modules
    ./run.sh day1 lora --with-eval   # Train, then evaluate automatically when training succeeds (Lab 1.4)
    ```
  * Direct Slurm submission:
    ```bash
    # LoRA (Container):
    cd $COURSE_ROOT/day1_single_gpu/finetuning/lora_slurm/job_container && sbatch job_singleGPU_QA_LoRA.sh
    # LoRA (EESSI):
    cd $COURSE_ROOT/day1_single_gpu/finetuning/lora_slurm/job_eessi && sbatch job_singleGPU_QA_LoRA_eessi.sh
    ```
* **Real-Time Monitoring:** `./run.sh monitor <JobID>` (see [Monitoring](#resource-usage--monitoring))
* **Where to Inspect Outputs:**
  * **Slurm Job Logs (stdout/stderr):** `day1_single_gpu/finetuning/lora_slurm/job_container/out/*.out` (EESSI: `.../job_eessi/out/`)
  * **Training Loss & Step Metrics:** `results/checkpoint_results/llama3.2-1b_lora_single_device/logs/`
  * **Saved Model Checkpoints:** `results/checkpoint_results/llama3.2-1b_lora_single_device/checkpoints/epoch_0/`
  * EESSI run: same paths with `eessi_llama3.2-1b_lora_single_device`.

---

### **Lab 1.2: Quantized Fine-Tuning with QLoRA**
* **Objective:** Fine-tune LLaMA 3.2-1B using 4-bit NormalFloat (NF4) base-model weight quantization with LoRA adapters.
* **Launch Commands:**
  * Fast launcher (from the course folder):
    ```bash
    ./run.sh day1 qlora               # Using Apptainer container
    ./run.sh day1 qlora --env=eessi   # Using EESSI modules
    ```
  * Direct Slurm submission:
    ```bash
    # QLoRA (Container):
    cd $COURSE_ROOT/day1_single_gpu/finetuning/qlora_slurm/job_container && sbatch job_singleGPU_QA_QLoRA.sh
    # QLoRA (EESSI):
    cd $COURSE_ROOT/day1_single_gpu/finetuning/qlora_slurm/job_eessi && sbatch job_singleGPU_QA_QLoRA_eessi.sh
    ```
* **Where to Inspect Outputs:**
  * **Slurm Job Logs (stdout/stderr):** `day1_single_gpu/finetuning/qlora_slurm/job_container/out/*.out` (EESSI: `.../job_eessi/out/`)
  * **Training Loss & Step Metrics:** `results/checkpoint_results/llama3.2-1b_qlora_single_device/logs/`
  * **Saved Model Checkpoints:** `results/checkpoint_results/llama3.2-1b_qlora_single_device/checkpoints/`
* **Student Focus:** Compare training step throughput (tokens/sec) between LoRA (BF16) and QLoRA (NF4 dequantization overhead) in the stdout logs.

---

### **Lab 1.3: GPU Memory & Kernel Profiling**
* **Objective:** Profile memory allocation, activation footprint, and kernel execution timelines using PyTorch Profiler.
* **Launch Commands:**
  * Fast launcher (from the course folder):
    ```bash
    ./run.sh day1 profile --method=lora    # Profile LoRA
    ./run.sh day1 profile --method=qlora   # Profile QLoRA
    ```
  * Direct Slurm submission:
    ```bash
    cd $COURSE_ROOT/day1_single_gpu/profiling && sbatch job_singleGPU_QA_LoRA_profiling.sh
    # or for QLoRA:
    cd $COURSE_ROOT/day1_single_gpu/profiling && sbatch job_singleGPU_QA_QLoRA_profiling.sh
    ```
* **Where to Inspect Outputs:**
  * **Slurm Job Logs:** `day1_single_gpu/profiling/out/*.out`
  * **Profiler Traces & Memory Timelines:**
    * LoRA: `results/profiling_results/profiling_llama3.2-1b_lora_single_device/profiles/`
    * QLoRA: `results/profiling_results/profiling_llama3.2-1b_qlora_single_device/profiles/`
* **Viewing Traces Locally on Your Laptop:**
  On Olivia, print the full path of the traces with `echo $PROFILING_RESULTS_DIR` (after `source course_env.sh`).
  Then, on your laptop (shown for the recommended clone location):
  ```bash
  scp USERNAME@olivia.sigma2.no:/cluster/work/projects/nn9971k/USERNAME/llm-hpc-course/results/profiling_results/profiling_llama3.2-1b_lora_single_device/profiles/rank0_memory-timeline.html .
  open rank0_memory-timeline.html   # macOS (or xdg-open on Linux)
  ```
  More details: [profiling/README.md](./profiling/README.md)
* **Key Analytical Questions for Students:**
  * How much peak VRAM was saved by 4-bit base weight quantization in QLoRA compared to LoRA?
  * Inspect the memory timeline HTML (`rank0_memory-timeline.html`): What proportion of memory is occupied by model weights vs. activations during the forward and backward passes?
  * What fraction of iteration time is spent on GPU compute vs. CPU host dispatching?

---

### **Lab 1.4: Model Merging & Benchmark Evaluation (MMLU)**
* **Objective:** Merge your trained LoRA adapter weights with base LLaMA weights and measure benchmark reasoning accuracy on MMLU tasks.
* **Prerequisite:** Lab 1.1 must have finished. The evaluation uses the Lab 1.1 adapter
  `results/checkpoint_results/llama3.2-1b_lora_single_device/checkpoints/epoch_0`
  (or the EESSI one if that is the only one you trained). If no adapter is found, the job stops
  immediately and tells you what to run.
  > [!TIP]
  > Train and evaluate in one step with `./run.sh day1 lora --with-eval`: the evaluation job starts automatically when training succeeds.
  >
  > Evaluate a different adapter (e.g. the QLoRA one) with `LORA_PATH`:
  > `LORA_PATH=$COURSE_ROOT/results/checkpoint_results/llama3.2-1b_qlora_single_device/checkpoints/epoch_0 ./run.sh day1 eval`
* **Launch Commands:**
  * Fast launcher (from the course folder):
    ```bash
    ./run.sh day1 eval                 # Evaluate fine-tuned LoRA model (merges the adapter first)
    ./run.sh day1 eval --base          # Evaluate base model (baseline)
    ./run.sh day1 eval --env=eessi     # Same with EESSI modules (add --base for the baseline)
    ```
  * Direct Slurm submission:
    ```bash
    cd $COURSE_ROOT/day1_single_gpu/evaluation && sbatch job_singleGPU_LoRA.sh        # LoRA (container)
    cd $COURSE_ROOT/day1_single_gpu/evaluation && sbatch job_singleGPU_No_LoRA.sh     # base model (container)
    cd $COURSE_ROOT/day1_single_gpu/evaluation && sbatch job_singleGPU_LoRA_eessi.sh  # LoRA (EESSI)
    cd $COURSE_ROOT/day1_single_gpu/evaluation && sbatch job_singleGPU_No_LoRA_eessi.sh
    ```
* **Where to Inspect Outputs:**
  * **Slurm Job Logs:** `day1_single_gpu/evaluation/out/*.out`
  * **Merged model:** next to the adapter, in `.../checkpoints/epoch_0_merged/` (created once, reused afterwards)
  * **Benchmark Scores & Summary Tables:**
    * Fine-tuned: `results/evaluation_results/llama3.2-1b-mmlu_singlegpu/` (EESSI: `..._singlegpu_eessi/`)
    * Base model: `results/evaluation_results/llama3.2-1b-base_mmlu_singlegpu/` (EESSI: `..._singlegpu_eessi/`)
* **Student Focus:** Compare accuracy metrics between the unadapted base model and your fine-tuned model on question-answering subjects.

---

## Resource Usage & Monitoring

- Monitor job status with `./run.sh status` (same as `squeue --me`).
- While a job is running, inspect real-time GPU metrics:
  ```bash
  ./run.sh monitor <JobID>      # from the course folder
  gpu.monitor.sh <JobID>        # from any folder, after: source course_env.sh
  run_nvitop.sh <JobID>         # interactive nvitop view (installs nvitop on first use)
  ```
- Alternatively, connect directly to the allocated GPU node: `ssh <gpu-node-name>` and run `nvidia-smi`.
- Check stdout/stderr log files in the `out/` directory for run progress and training statistics.

## If a Job Fails

1. Open the newest log (inside the lab folder): `tail -50 $(ls -t out/*.out | head -1)`
2. Messages starting with `ERROR: ... not found` say exactly what is missing and how to fix it
   (usually: run `./setup_env.sh`, or finish the previous lab first).
3. `sbatch: error: ... account` → run `source course_env.sh` and submit again (or use `./run.sh`).
4. `./check_env.sh` re-checks models, datasets, container images and folders.
