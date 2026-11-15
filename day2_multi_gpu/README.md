# Day 2: Distributed Training, Quantization, Serving & Inference

This directory contains labs, configurations, and job scripts for scaling LLM workloads across multiple GPUs and multiple nodes on the Olivia supercomputer.

## Learning Objectives

- Scale fine-tuning across multiple GPUs (single-node) and multiple nodes using PyTorch FSDP.
- Compare **LoRA**, **QLoRA**, and **QAT-LoRA** (Quantization-Aware Training) for Question-Answering (QA) tasks.
- Profile distributed workloads to analyze inter-GPU communication and memory scaling.
- Apply post-training quantization techniques using **BitsAndBytes**, **TorchAO**, and **GPTQModel**.
- Deploy high-throughput LLM serving and inference pipelines using **vLLM** (single-device and multi-device tensor parallel).

---

## Before You Start

The course is in **`/cluster/work/projects/nn9971k/$USER/llm-hpc-course`** (see [Getting Started](../README.md#getting-started-on-olivia)).
As on Day 1, every lab works the same way:

1. `cd` into the lab folder (copy the full `cd` line below),
2. `sbatch <script>.sh` — note the **JobID** it prints,
3. follow the job with `squeue --me` and `tail -f out/*.out`.

---

## Lab 2.1: Multi-GPU Fine-Tuning (1 node, 4 GPUs)
**Goal:** Scale fine-tuning across 4 GPUs on a single node using PyTorch Fully Sharded Data Parallel (FSDP).

**Run it** — pick one method:
```bash
# LoRA (container)
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day2_multi_gpu/fine-tuning/ft_single_node/lora_slurm/job_container
sbatch job_multiGPU_QA_LoRA.sh

# LoRA (EESSI)
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day2_multi_gpu/fine-tuning/ft_single_node/lora_slurm/job_eessi
sbatch job_multiGPU_QA_LoRA_eessi.sh

# QLoRA (container)
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day2_multi_gpu/fine-tuning/ft_single_node/qlora_slurm
sbatch job_multiGPU_QA_QLoRA.sh

# QAT-LoRA (container)
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day2_multi_gpu/fine-tuning/ft_single_node/qat-lora_slurm
sbatch job_multiGPU_QA_qat-LoRA.sh
```

**Watch all 4 GPUs** (in the same folder): `./gpu.monitor.sh <JobID>` — is the work balanced across the GPUs?

**Results:** job log in `out/`; training loss and adapter in
`results/checkpoint_results/llama3.2-1b_<method>_multi_device/` (`<method>` = `lora`, `qlora` or `qat-lora`;
EESSI: `eessi_llama3.2-1b_lora_multi_device/`).

---

## Lab 2.2: Multi-Node Fine-Tuning (2 nodes, 8 GPUs)
**Goal:** Scale training across several nodes over the Slingshot network (AWS-OFI-NCCL).

**Run it** — pick one method:
```bash
# LoRA (container)
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day2_multi_gpu/fine-tuning/ft_multi_node/lora_slurm/job_container
sbatch job_multiNode_QA_LoRA.sh

# LoRA (EESSI)
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day2_multi_gpu/fine-tuning/ft_multi_node/lora_slurm/job_eessi
sbatch job_multiNode_QA_LoRA_eessi.sh

# QLoRA (container)
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day2_multi_gpu/fine-tuning/ft_multi_node/qlora_slurm
sbatch job_multiNode_QA_QLoRA.sh

# QAT-LoRA (container)
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day2_multi_gpu/fine-tuning/ft_multi_node/qat-lora_slurm
sbatch job_multiNode_QA_qat-LoRA.sh
```

**Results:** job log in `out/`; adapter in `results/checkpoint_results/llama3.2-1b_<method>_multi_node/`.
The network settings used by these jobs are in `nccl_env.sh`, next to each script.

**Question:** Compare the training time per step with Lab 2.1. How well does it scale from 4 to 8 GPUs?

---

## Lab 2.3: Distributed Profiling
**Goal:** Profile 4-GPU training to see how much time goes into communication between GPUs.

**Run it:**
```bash
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day2_multi_gpu/profiling
sbatch job_multiGPU_QA_LoRA.sh        # LoRA
sbatch job_multiGPU_QA_QLoRA.sh       # QLoRA
sbatch job_multiGPU_QA_qat-LoRA.sh    # QAT-LoRA
```

**Results:** job log in `profiling/out/`; one trace per GPU (`rank0_...`, `rank1_...`) in
`results/profiling_results/profiling_llama3.2-1b_<method>_multi_device/profiles/`.

**Look at the traces on your laptop** (run this on your laptop, not on Olivia):
```bash
scp -r <username>@olivia.sigma2.no:/cluster/work/projects/nn9971k/<username>/llm-hpc-course/results/profiling_results/profiling_llama3.2-1b_lora_multi_device/profiles .
open profiles/rank0_memory-timeline.html     # macOS  (Linux: xdg-open, Windows: start)
```

**Questions:**
* What proportion of iteration time is spent in NCCL collective calls (`all_reduce`, `reduce_scatter`) vs. matrix multiplication?
* How does this compare with the single-GPU profile from Lab 1.3?

---

## Lab 2.4: Evaluation on 4 GPUs (MMLU)
**Goal:** Evaluate the fine-tuned model on MMLU using 4 GPUs.

**Run it:**
```bash
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day2_multi_gpu/evaluation
sbatch job_multiGPU_LoRA.sh          # fine-tuned model (adapter from Lab 1.1, container)
sbatch job_multiGPU_No_LoRA.sh       # base model, for comparison
```
EESSI versions: `job_multiGPU_LoRA_eessi.sh` (adapter from the EESSI run of Lab 1.1) and `job_multiGPU_No_LoRA_eessi.sh`.

**Results** (in `results/evaluation_results/`): `llama3.2-1b-mmlu_multigpu/` and `llama3.2-1b-base-mmlu/`
(EESSI: `llama3.2-1b-mmlu_eessi/`, `llama3.2-1b-base-mmlu_eessi/`).

> [!TIP]
> To evaluate your Day 2 adapter instead, open `job_multiGPU_LoRA.sh` and set
> `LORA_ADAPTER_DIR` to `.../checkpoint_results/llama3.2-1b_lora_multi_device/checkpoints/epoch_0`.
> Also give `MERGED_MODEL_DIR` a new name, otherwise the old merged model is reused.

---

## Lab 2.5: Post-Training Quantization
**Goal:** Make the model smaller by quantizing its weights, without retraining. (These jobs use EESSI modules.)

**Run it** — one job per method:
```bash
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day2_multi_gpu/quantization/quat_bitsandbytes
sbatch job_bnb_4bit.sh               # BitsAndBytes 4-bit
sbatch job_bnb_8bit.sh               # BitsAndBytes 8-bit (used by the serving lab)

cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day2_multi_gpu/quantization/quat_torchao
sbatch job_torchao.sh                # TorchAO INT8

cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day2_multi_gpu/quantization/quat_gptqmodel
sbatch job_gptq.sh                   # GPTQ 4-bit (job_gptq_with_filter.sh: filtered calibration data)
```

**Results:** the quantized models are saved in your course folder, in
`shared/models/Llama-3.2-1B-Instruct-<method>` (`bnb_4bit`, `bnb_8bit`, `torchao_int8`, `gptq_4bit`).

**Question:** Compare the size on disk:
```bash
du -sh /cluster/work/projects/nn9971k/$USER/llm-hpc-course/shared/models/*
```

---

## Lab 2.6: Serving with vLLM + Terminal Chat
**Goal:** Start an OpenAI-compatible vLLM server with your LoRA adapter, then chat with it from the terminal.

**1. Start the server** (it runs as a Slurm job for up to 30 minutes):
```bash
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day2_multi_gpu/serving/single_device/vllm_container
sbatch serve_vllm.sh
```

**2. Wait until it is ready** — the log shows `Application startup complete`:
```bash
tail -f out/*.out          # Ctrl+C to stop following
```

**3. Chat with it** (same folder; type `exit` to quit):
```bash
./chat.sh
```

**4. Stop the server** when you are done: `scancel <JobID>`

**Other servers** — same steps, other folder or script:

| Folder (in `day2_multi_gpu/serving/`) | Script | What it serves |
| :--- | :--- | :--- |
| `single_device/vllm_container` | `serve_vllm.sh` | base model + your LoRA adapter, 1 GPU |
| `single_device/vllm_container` | `serve_vllm_quantized_bnb.sh` / `_gptq.sh` / `_torchao.sh` | quantized model (Lab 2.5) + adapter |
| `multi_device/vllm_container` | same script names | the same on 4 GPUs (tensor parallel) |
| `single_device/vllm_eessi`, `multi_device/vllm_eessi` | `serve_vllm_eessi.sh`, `serve_vllm_quantized_*_eessi.sh` | EESSI versions (use the EESSI adapter) |

Always run `./chat.sh` in the **same folder** as the server you started.
The quantized servers need the matching model from Lab 2.5 (`bnb_8bit`, `gptq_4bit` or `torchao_int8`).

**Options** (put them in front of `sbatch`):
```bash
LORA_PATH="" sbatch serve_vllm.sh        # serve the base model only, without adapter
PORT=8001 sbatch serve_vllm.sh           # use another port if 8000 is busy
```

**Optional — benchmark the running single-GPU container server:**
```bash
/cluster/work/projects/nn9971k/$USER/llm-hpc-course/recipes/bench/run_benchmark.sh
```

---

## Lab 2.7: Batch Inference with vLLM
**Goal:** Measure batch inference throughput with vLLM tensor parallelism on 4 GPUs.

**Run it:**
```bash
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day2_multi_gpu/inference
sbatch slurm_vllm_distributed_container.sh     # 4 GPUs, container
sbatch slurm_vllm_distributed_eessi.sh         # 4 GPUs, EESSI (uses the EESSI adapter)
sbatch slurm_vllm_lora_singlegpu.sh            # 1 GPU, for comparison
```
The prompts are in `prompt_QA.json` in this folder.

**Results:** the generated answers, tokens/second and latency are printed in the job log (`out/`).

**Question:** How much faster is 4-GPU inference than 1-GPU inference for this small model?

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
2. A model, dataset or adapter is missing → check that the lab it comes from has finished
   (Lab 1.1 for adapters, Lab 2.5 for quantized models).
3. `chat.sh` keeps waiting → the server is still starting, or it stopped: check the server log in `out/`.
4. Still stuck? Ask an instructor and show them the log file.
