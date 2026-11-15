# Day 1: Single GPU Workflow

This directory contains labs and scripts for performing LLM fine‑tuning, evaluation, and performance profiling on a single GPU using the Alpaca GPT-4 Question-Answering (QA) dataset.

## Learning Objectives

- Learn how to submit Slurm jobs for LLM fine‑tuning using both Apptainer containers and EESSI software modules.
- Compare **LoRA** and **QLoRA** techniques in terms of memory usage, throughput, and accuracy.
- Evaluate base and fine-tuned models on benchmarks such as MMLU.
- Profile single-GPU training jobs to analyze memory allocation and compute bottlenecks.

---

## Before You Start

You should have done the [Getting Started](../README.md#getting-started-on-olivia) steps once:
the course is in **`/cluster/work/projects/nn9971k/$USER/llm-hpc-course`** and `./setup_env.sh` has run.

Every lab works the same way:

1. `cd` into the lab folder (copy the full `cd` line below),
2. `sbatch <script>.sh` — note the **JobID** it prints,
3. follow the job with `squeue --me` and `tail -f out/*.out`.

**Container or EESSI?** Each fine-tuning lab comes in two versions that do the same work, side by side
in the same folder: the plain script runs inside an Apptainer container, the `_eessi.sh` script uses EESSI
software modules.
Start with the container version; try EESSI afterwards if you have time.

---

## Lab 1.1: Single-GPU LoRA Fine-Tuning
**Goal:** Fine-tune LLaMA 3.2-1B on the Question-Answering (QA) task using BF16 Low-Rank Adaptation (LoRA).

**Run it (container):**
```bash
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day1_single_gpu/finetuning/lora_slurm
sbatch job_singleGPU_QA_LoRA.sh
```

**Run it (EESSI):**
```bash
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day1_single_gpu/finetuning/lora_slurm
sbatch job_singleGPU_QA_LoRA_eessi.sh
```

**Watch the GPU while it runs** (in the same folder): `./gpu.monitor.sh <JobID>`

**Results** (in `llm-hpc-course/results/checkpoint_results/`):
| | Container | EESSI |
| :--- | :--- | :--- |
| Job log | `lora_slurm/out/` | `lora_slurm/out/` |
| Training loss | `llama3.2-1b_lora_single_device/logs/` | `eessi_llama3.2-1b_lora_single_device/logs/` |
| Fine-tuned adapter | `llama3.2-1b_lora_single_device/checkpoints/epoch_0/` | `eessi_llama3.2-1b_lora_single_device/checkpoints/epoch_0/` |

---

## Lab 1.2: Quantized Fine-Tuning with QLoRA
**Goal:** Fine-tune LLaMA 3.2-1B using 4-bit NormalFloat (NF4) base-model weight quantization with LoRA adapters.

**Run it (container):**
```bash
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day1_single_gpu/finetuning/qlora_slurm
sbatch job_singleGPU_QA_QLoRA.sh
```

**Run it (EESSI):**
```bash
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day1_single_gpu/finetuning/qlora_slurm
sbatch job_singleGPU_QA_QLoRA_eessi.sh
```

**Results:** job log in `out/`; training loss and adapter in
`results/checkpoint_results/llama3.2-1b_qlora_single_device/` (EESSI: `eessi_llama3.2-1b_qlora_single_device/`).

**Question:** Compare the training speed (tokens/sec in the log) of LoRA (BF16) and QLoRA (NF4). Which is faster, and why?

---

## Lab 1.3: GPU Memory & Kernel Profiling
**Goal:** Profile memory allocation, activation footprint, and kernel execution timelines using PyTorch Profiler.

**Run it:**
```bash
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day1_single_gpu/profiling
sbatch job_singleGPU_QA_LoRA_profiling.sh      # profile LoRA
sbatch job_singleGPU_QA_QLoRA_profiling.sh     # profile QLoRA
```

**Results:** job log in `profiling/out/`; profiler traces in
`results/profiling_results/profiling_llama3.2-1b_lora_single_device/profiles/` (QLoRA: `..._qlora_single_device/`).

**Look at the memory timeline on your laptop** (run this on your laptop, not on Olivia):
```bash
scp <username>@olivia.sigma2.no:/cluster/work/projects/nn9971k/<username>/llm-hpc-course/results/profiling_results/profiling_llama3.2-1b_lora_single_device/profiles/rank0_memory-timeline.html .
open rank0_memory-timeline.html     # macOS  (Linux: xdg-open, Windows: start)
```
More details: [profiling/README.md](./profiling/README.md)

**Questions:**
* How much peak GPU memory did QLoRA's 4-bit base weights save compared to LoRA?
* In the memory timeline, how much memory is taken by model weights vs. activations during the forward and backward passes?
* What fraction of iteration time is spent on GPU compute vs. CPU host dispatching?

---

## Lab 1.4: Model Merging & Benchmark Evaluation (MMLU)
**Goal:** Merge your trained LoRA adapter with the base model and measure its accuracy on MMLU tasks.

> [!NOTE]
> Lab 1.1 must be finished first: this lab evaluates the adapter it produced.
> The container script uses the container adapter, the EESSI script uses the EESSI adapter.

**Run it (container):**
```bash
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day1_single_gpu/evaluation
sbatch job_singleGPU_LoRA.sh         # your fine-tuned model
sbatch job_singleGPU_No_LoRA.sh      # the base model, for comparison
```

**Run it (EESSI):** same folder, `sbatch job_singleGPU_LoRA_eessi.sh` and `sbatch job_singleGPU_No_LoRA_eessi.sh`.

**Results** (in `llm-hpc-course/results/evaluation_results/`):
| | Container | EESSI |
| :--- | :--- | :--- |
| Fine-tuned model | `llama3.2-1b-mmlu_singlegpu/` | `llama3.2-1b-mmlu_singlegpu_eessi/` |
| Base model | `llama3.2-1b-base_mmlu_singlegpu/` | `llama3.2-1b-base_mmlu_singlegpu_eessi/` |

The merged model is saved once in `results/checkpoint_results/Llama-3.2-1B-Instruct_merged/`
(EESSI: `..._merged_eessi/`) and reused by later runs.

**Question:** Compare the accuracy of the base model and your fine-tuned model. Did fine-tuning on QA help?

> [!TIP]
> To evaluate another adapter (for example the QLoRA one), open `job_singleGPU_LoRA.sh`, change
> `LORA_ADAPTER_DIR`, and also give `MERGED_MODEL_DIR` a new name — otherwise the old merged model is reused.

---

## Monitoring Your Jobs

```bash
squeue --me                                   # your jobs: PD = waiting, R = running
tail -f out/*.out                             # follow the logs of this lab folder (Ctrl+C to stop)
./gpu.monitor.sh <JobID>                      # live GPU usage (in folders that contain it)
/cluster/work/projects/nn9971k/$USER/llm-hpc-course/utils/gpu.monitor.sh <JobID>   # same, from any folder
scancel <JobID>                               # cancel a job
```

## If a Job Fails

1. Read the end of the log: `tail -50 out/<job-name>-<JobID>.out`
2. `No such file or directory` for a model, dataset or `.sif` image → run `./setup_env.sh` again from the course folder.
3. Evaluation fails because the adapter is missing → finish Lab 1.1 (same version: container or EESSI) first.
4. Still stuck? Ask an instructor and show them the log file.
