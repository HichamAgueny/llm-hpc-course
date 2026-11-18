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
An alternative is to use `nvitop`: `/cluster/work/projects/nn9971k/hicham/llm-hpc-course/utils/run_nvitop.sh`

**Results** (in `llm-hpc-course/results/checkpoint_results/`):
| | Container | EESSI |
| :--- | :--- | :--- |
| Job log | `lora_slurm/out/` | `lora_slurm/out/` |
| Training loss | `llama3.2-1b_lora_single_device/logs/` | `eessi_llama3.2-1b_lora_single_device/logs/` |
| Fine-tuned adapter | `llama3.2-1b_lora_single_device/checkpoints/epoch_1/` | `eessi_llama3.2-1b_lora_single_device/checkpoints/epoch_1/` |

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

## Lab 1.3: GPU Memory Profiling — LoRA vs. QLoRA
**Goal:** Understand how LoRA and QLoRA affect GPU memory use and training speed, and what activation checkpointing changes.

The jobs use these configuration files (in your course folder):
- LoRA: `configs/lora/llama3_2_1B_lora_single_device_profiling.yaml`
- QLoRA: `configs/qlora/llama3_2_1B_qlora_single_device_profiling.yaml`

### Exercise 1 — GPU Memory Profiling: LoRA vs. QLoRA
1. Run both profiling jobs:
   ```bash
   cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day1_single_gpu/profiling
   sbatch job_singleGPU_QA_LoRA_profiling.sh      # profile LoRA
   sbatch job_singleGPU_QA_QLoRA_profiling.sh     # profile QLoRA
   ```
2. Open the memory timeline of each run on your laptop (see [below](#look-at-the-memory-timeline-on-your-laptop)).
3. Find the peak GPU memory and the main sources of memory use (model weights, activations, gradients, optimizer state).
4. Compare LoRA and QLoRA.

**Questions:** Which method uses less GPU memory? What explains the difference?

### Exercise 2 — Impact of Activation Checkpointing
1. Keep the Exercise 1 results by renaming their folders (otherwise the new runs overwrite them):
   ```bash
   cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/results/profiling_results
   mv profiling_llama3.2-1b_lora_single_device  profiling_llama3.2-1b_lora_single_device_no_ac
   mv profiling_llama3.2-1b_qlora_single_device profiling_llama3.2-1b_qlora_single_device_no_ac
   ```
2. In **both** configuration files, find the line `enable_activation_checkpointing: False` and change it to:
   ```yaml
   enable_activation_checkpointing: True
   ```
3. Rerun both profiling jobs (same `cd` and `sbatch` commands as in Exercise 1).
4. Compare the peak GPU memory with and without activation checkpointing, and the training step time
   (in the training log in `logs/` and the job log in `profiling/out/`).
5. When you are done, set `enable_activation_checkpointing` back to `False` in both files.

**Questions:** How much memory is saved? Does activation checkpointing add compute time? Why?

### Analyze and Summarize Your Results
Fill in this table with your own measurements (copy it into your notes):

| Configuration | Peak GPU memory | Step time |
| :--- | :--- | :--- |
| LoRA | | |
| QLoRA | | |
| LoRA + Activation Checkpointing | | |
| QLoRA + Activation Checkpointing | | |

### Results
* Job logs: `day1_single_gpu/profiling/out/`
* Training logs and profiler traces: `results/profiling_results/profiling_llama3.2-1b_<method>_single_device/logs/` and `.../profiles/`
  (`<method>` = `lora` or `qlora`; the Exercise 1 results are in the folders ending in `_no_ac`).

### Look at the memory timeline on your laptop
Run this on your laptop, not on Olivia (replace `<username>`; change `lora` to `qlora`, or add `_no_ac`, for the other runs):
```bash
scp <username>@olivia.sigma2.no:/cluster/work/projects/nn9971k/<username>/llm-hpc-course/results/profiling_results/profiling_llama3.2-1b_lora_single_device/profiles/rank0_memory-timeline.html lora_memory-timeline.html
open lora_memory-timeline.html     # macOS  (Linux: xdg-open, Windows: start)
```
More details: [profiling/README.md](./profiling/README.md)

**Extra questions (if you have time):**
* In the memory timeline, how much memory is taken by model weights vs. activations during the forward and backward passes?

---

## Lab 1.4: Evaluating Fine-Tuned LLMs — LoRA vs. Base Model
**Goal:** Measure how LoRA fine-tuning changes the model's performance compared with the original base model,
using standard benchmarks: MMLU, HellaSwag and TruthfulQA.

> [!NOTE]
> Lab 1.1 must be finished first: the LoRA job evaluates the adapter it produced.
> The container script uses the container adapter, the EESSI script uses the EESSI adapter.

Two jobs, in the same folder:
- `job_singleGPU_No_LoRA.sh` — the original Llama 3.2 1B model (no adapter)
- `job_singleGPU_LoRA.sh` — the same model merged with your LoRA adapter from Lab 1.1

The benchmarks are chosen on the line starting with `tasks=` in each script.
The scores are printed at the end of the job log in `out/` — copy them from there
(each new run overwrites the previous `results.json`).

> [!NOTE]
> To keep the jobs short, each benchmark uses only 150 questions (`limit: 150` in
> `configs/eval_cfg/eval_config.yaml`). Small differences between the two models may be noise.

### Exercise 1 — Base Model on MMLU
1. Run the base model on four MMLU subjects (already set in the script):
   ```bash
   cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day1_single_gpu/evaluation
   sbatch job_singleGPU_No_LoRA.sh
   ```
2. Record the accuracy (`acc`) for each subject, and calculate the average of the four.

**Questions:** Does the base model perform equally well in all subjects? Which subject is the most difficult?

### Exercise 2 — LoRA Model on MMLU
1. Run the fine-tuned model on the same four subjects (same folder):
   ```bash
   sbatch job_singleGPU_LoRA.sh
   ```
2. Compare its accuracy with the base model. In which subjects did fine-tuning help, and in which did it hurt?

**Questions:** Does LoRA fine-tuning improve general knowledge and reasoning in all subjects? Why might it get worse in some?

### Exercise 3 — Commonsense Reasoning (HellaSwag)
1. In **both** scripts, replace the `tasks=` line with:
   ```bash
         tasks=["hellaswag"] \
   ```
2. Run both jobs again (`sbatch job_singleGPU_No_LoRA.sh` and `sbatch job_singleGPU_LoRA.sh`)
   and compare the scores (`acc_norm`).

**Question:** Does better performance on the fine-tuning task mean better commonsense reasoning?

### Exercise 4 — Truthfulness (TruthfulQA)
1. In **both** scripts, replace the `tasks=` line with:
   ```bash
         tasks=["truthfulqa_mc2"] \
   ```
2. Run both jobs again and compare the scores (`acc`).

**Question:** Can fine-tuning improve one benchmark while reducing truthfulness on another?

When you are done, set the `tasks=` line back to the four MMLU subjects:
```bash
      tasks=["mmlu_philosophy","mmlu_sociology","mmlu_business_ethics","mmlu_college_physics"] \
```

> [!TIP]
> Short on time? Run Exercises 3 and 4 together: `tasks=["hellaswag","truthfulqa_mc2"] \` — 2 jobs instead of 4.

### Exercise 5 — Compare and Analyze
Fill in this table with your measurements (copy it into your notes):

| Benchmark | Base model | LoRA model | Difference |
| :--- | :--- | :--- | :--- |
| MMLU – Philosophy | | | |
| MMLU – Sociology | | | |
| MMLU – Business Ethics | | | |
| MMLU – College Physics | | | |
| HellaSwag | | | |
| TruthfulQA (MC2) | | | |

### Run it with EESSI
Same steps with `job_singleGPU_No_LoRA_eessi.sh` and `job_singleGPU_LoRA_eessi.sh` (same folder).

### Results
Job logs (with the score tables): `day1_single_gpu/evaluation/out/`.
The latest `results.json` of each job is in `llm-hpc-course/results/evaluation_results/`:
| | Container | EESSI |
| :--- | :--- | :--- |
| LoRA model | `llama3.2-1b-mmlu_singlegpu/` | `llama3.2-1b-mmlu_singlegpu_eessi/` |
| Base model | `llama3.2-1b-base_mmlu_singlegpu/` | `llama3.2-1b-base_mmlu_singlegpu_eessi/` |

The merged model is saved once in `results/checkpoint_results/Llama-3.2-1B-Instruct_merged/`
(EESSI: `..._merged_eessi/`) and reused by later runs.

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
/cluster/work/projects/nn9971k/hicham/llm-hpc-course/utils/run_nvitop.sh
```

## If a Job Fails

1. Read the end of the log: `tail -50 out/<job-name>-<JobID>.out`
2. `No such file or directory` for a model, dataset or `.sif` image → run `./setup_env.sh` again from the course folder.
3. Evaluation fails because the adapter is missing → finish Lab 1.1 (same version: container or EESSI) first.
