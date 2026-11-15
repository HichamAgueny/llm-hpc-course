# Multi-GPU & Distributed Profiling

This directory contains job scripts for profiling multi-GPU and distributed LLM fine-tuning on the Olivia supercomputer.

## Overview

Distributed profiling captures inter-GPU communication metrics, NCCL collective synchronization bottlenecks (e.g., all-reduce and reduce-scatter overhead in FSDP), along with compute and memory utilization per rank.

---

## Profiling Scripts

| Script | Method | Description |
| :--- | :--- | :--- |
| `job_multiGPU_QA_LoRA.sh` | LoRA (4 GPUs) | Profiles distributed LoRA fine-tuning using PyTorch FSDP. |
| `job_multiGPU_QA_QLoRA.sh` | QLoRA (4 GPUs) | Profiles distributed 4-bit QLoRA with FSDP model sharding. |
| `job_multiGPU_QA_qat-LoRA.sh` | QAT-LoRA (4 GPUs) | Profiles distributed Quantization-Aware Training with LoRA. |

---

## Submitting a Profiling Job

```bash
cd day2_multi_gpu/profiling

# Run LoRA distributed profiling
sbatch job_multiGPU_QA_LoRA.sh

# Run QLoRA distributed profiling
sbatch job_multiGPU_QA_QLoRA.sh

# Run QAT-LoRA distributed profiling
sbatch job_multiGPU_QA_qat-LoRA.sh
```

---

## Output and Visualization

- Each rank generates its own timeline and trace (e.g., `rank0_...`, `rank1_...`, `rank2_...`, `rank3_...`).
- Download trace artifacts to your workstation using `scp` and visualize using Chrome Tracing (`chrome://tracing`), [Perfetto](https://ui.perfetto.dev/), or the PyTorch Profiler plugin for TensorBoard.

